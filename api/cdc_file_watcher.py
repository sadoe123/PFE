"""
OnePilot – CDC File Watcher §2.2.5
Delta detection pour sources fichiers (CSV, Excel, JSON)
Checksums MD5 + timestamps + taille
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import asyncpg
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/uploads")


# ── Calcul checksum fichier ───────────────────────────────────

def compute_file_checksum(filepath: str, chunk_size: int = 65536) -> Optional[str]:
    """MD5 checksum d'un fichier en streaming (gros fichiers)."""
    try:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"[FileWatcher] Checksum error {filepath}: {e}")
        return None


def get_file_meta(filepath: str) -> Optional[dict]:
    """Retourne taille + mtime d'un fichier."""
    try:
        stat = os.stat(filepath)
        return {
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        }
    except Exception as e:
        logger.warning(f"[FileWatcher] Stat error {filepath}: {e}")
        return None


# ── Moteur File CDC ───────────────────────────────────────────

class FileCDCEngine:
    """
    CDC pour sources de type fichier (file_csv, file_excel, file_json).
    Détecte les changements via checksum MD5 + timestamp + taille.
    S'intègre avec CDCEngine pour créer des versions de schéma.
    """

    def __init__(self, pg_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.pg    = pg_pool
        self.redis = redis_client

    # ── 1. Snapshot fichier ──────────────────────────────────

    async def snapshot_file(self, source_id: UUID) -> Optional[dict]:
        """
        Récupère le chemin du fichier depuis les options de la source
        et calcule checksum + meta.
        """
        row = await self.pg.fetchrow("""
            SELECT name, connector_type, options
            FROM   data_sources
            WHERE  id = $1
        """, source_id)

        if not row:
            return None

        options = row["options"] or {}
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = {}

        # Cherche le chemin dans les options
        filepath = (
            options.get("file_path") or
            options.get("filepath") or
            options.get("path")
        )

        if not filepath:
            # Cherche dans le dossier uploads par nom de source
            name = row["name"]
            for ext in [".csv", ".xlsx", ".xls", ".json"]:
                candidate = os.path.join(UPLOAD_DIR, f"{name}{ext}")
                if os.path.exists(candidate):
                    filepath = candidate
                    break

        if not filepath or not os.path.exists(filepath):
            logger.warning(
                f"[FileWatcher] Fichier introuvable pour source {source_id}"
            )
            return None

        checksum = compute_file_checksum(filepath)
        meta     = get_file_meta(filepath)

        if not checksum or not meta:
            return None

        return {
            "filepath":    filepath,
            "checksum":    checksum,
            "size_bytes":  meta["size_bytes"],
            "modified_at": meta["modified_at"],
            "source_name": row["name"],
            "connector_type": row["connector_type"],
        }

    # ── 2. Détecter changement fichier ───────────────────────

    async def detect_file_change(self, source_id: UUID) -> dict:
        """
        Compare le checksum actuel avec la dernière valeur connue.
        Persiste dans file_cdc_history si changement détecté.
        Publie une notification Redis.
        """
        current = await self.snapshot_file(source_id)

        if not current:
            return {
                "status":  "error",
                "message": "Fichier introuvable ou inaccessible",
            }

        # Dernière entrée connue
        last = await self.pg.fetchrow("""
            SELECT checksum, size_bytes, modified_at, detected_at
            FROM   file_cdc_history
            WHERE  source_id = $1
            ORDER  BY detected_at DESC
            LIMIT  1
        """, source_id)

        if last and last["checksum"] == current["checksum"]:
            logger.info(
                f"[FileWatcher] source {source_id} — fichier inchangé "
                f"(checksum={current['checksum'][:8]})"
            )
            return {
                "status":    "no_change",
                "checksum":  current["checksum"],
                "filepath":  current["filepath"],
                "size_bytes": current["size_bytes"],
            }

        # Changement détecté
        change_type = "CREATED" if not last else "MODIFIED"
        size_delta  = None
        if last:
            size_delta = current["size_bytes"] - last["size_bytes"]

        await self.pg.execute("""
            INSERT INTO file_cdc_history
                (source_id, filepath, checksum, size_bytes,
                 file_modified_at, change_type, size_delta, detected_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
        """,
            source_id,
            current["filepath"],
            current["checksum"],
            current["size_bytes"],
            current["modified_at"],
            change_type,
            size_delta,
            datetime.now(timezone.utc),
        )

        # Publie notification Redis
        payload = json.dumps({
            "event":        "FILE_CHANGED",
            "source_id":    str(source_id),
            "change_type":  change_type,
            "filepath":     current["filepath"],
            "checksum":     current["checksum"],
            "size_bytes":   current["size_bytes"],
            "size_delta":   size_delta,
            "modified_at":  current["modified_at"],
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        })

        try:
            await self.redis.publish(f"cdc:file:{source_id}", payload)
            key = f"cdc:file:notifications:{source_id}"
            await self.redis.lpush(key, payload)
            await self.redis.ltrim(key, 0, 49)
            await self.redis.expire(key, 60 * 60 * 24 * 7)
        except Exception as e:
            logger.warning(f"[FileWatcher] Redis publish error: {e}")

        logger.info(
            f"[FileWatcher] source {source_id} — fichier {change_type} "
            f"(size_delta={size_delta})"
        )

        return {
            "status":       "changed",
            "change_type":  change_type,
            "filepath":     current["filepath"],
            "checksum":     current["checksum"],
            "size_bytes":   current["size_bytes"],
            "size_delta":   size_delta,
            "modified_at":  current["modified_at"],
            "needs_resync": True,
        }

    # ── 3. Historique fichier ────────────────────────────────

    async def get_file_history(
        self, source_id: UUID, limit: int = 20
    ) -> list[dict]:
        """Retourne l'historique des changements de fichier."""
        rows = await self.pg.fetch("""
            SELECT
                checksum, size_bytes, file_modified_at,
                change_type, size_delta, detected_at
            FROM   file_cdc_history
            WHERE  source_id = $1
            ORDER  BY detected_at DESC
            LIMIT  $2
        """, source_id, limit)

        return [
            {
                "checksum":       r["checksum"][:12] + "…",
                "size_bytes":     r["size_bytes"],
                "file_modified":  r["file_modified_at"],
                "change_type":    r["change_type"],
                "size_delta":     r["size_delta"],
                "detected_at":    r["detected_at"].isoformat(),
            }
            for r in rows
        ]

    # ── 4. Notifications fichier Redis ───────────────────────

    async def get_file_notifications(
        self, source_id: UUID, limit: int = 20
    ) -> list[dict]:
        """Notifications de changements fichier depuis Redis."""
        try:
            key   = f"cdc:file:notifications:{source_id}"
            items = await self.redis.lrange(key, 0, limit - 1)
            return [json.loads(i) for i in items]
        except Exception:
            return await self.get_file_history(source_id, limit)