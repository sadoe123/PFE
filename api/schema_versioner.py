"""
OnePilot – Schema Versioner §2.2.5
Historique Git-like des versions de schéma : log, diff, tag, checkout
"""
from __future__ import annotations

import json
import logging
from typing import Optional
from uuid import UUID

import asyncpg
import redis.asyncio as aioredis

from .cdc_engine import diff_schemas

logger = logging.getLogger(__name__)


class SchemaVersioner:
    """
    Couche lecture/écriture sur l'historique des versions de schéma.
    Analogie Git :
      get_log()          → git log
      get_version()      → git show <commit>
      diff_versions()    → git diff v1 v2
      tag_version()      → git tag
      get_notifications()→ git log --grep="BREAKING"
    """

    def __init__(self, pg_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.pg    = pg_pool
        self.redis = redis_client

    # ── Log ───────────────────────────────────────────────────

    async def get_log(self, source_id: UUID, limit: int = 50) -> list[dict]:
        """
        Retourne l'historique des versions, plus récente en premier.
        Enrichit chaque version avec ses tags éventuels.
        """
        rows = await self.pg.fetch("""
            SELECT
                sv.id,
                sv.version_number,
                sv.fingerprint,
                sv.has_breaking_changes,
                sv.is_rollback,
                sv.rollback_from_version,
                sv.change_summary,
                sv.created_at,
                jsonb_array_length(sv.changes_delta) AS change_count,
                COALESCE(
                    (SELECT json_agg(t.tag)
                     FROM   schema_version_tags t
                     WHERE  t.source_id      = sv.source_id
                       AND  t.version_number = sv.version_number),
                    '[]'::json
                ) AS tags
            FROM   schema_versions sv
            WHERE  sv.source_id = $1
            ORDER  BY sv.version_number DESC
            LIMIT  $2
        """, source_id, limit)

        result = []
        for r in rows:
            summary = {}
            try:
                raw = r["change_summary"]
                summary = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                pass

            tags = []
            try:
                raw_tags = r["tags"]
                tags = json.loads(raw_tags) if isinstance(raw_tags, str) else (raw_tags or [])
            except Exception:
                pass

            result.append({
                "id":                  str(r["id"]),
                "version":             r["version_number"],
                "fingerprint":         r["fingerprint"][:12] + "…",
                "fingerprint_full":    r["fingerprint"],
                "has_breaking":        r["has_breaking_changes"],
                "is_rollback":         r["is_rollback"],
                "rollback_from":       r["rollback_from_version"],
                "change_count":        r["change_count"] or 0,
                "summary":             summary,
                "tags":                tags,
                "created_at":          r["created_at"].isoformat(),
            })
        return result

    # ── Détail d'une version ──────────────────────────────────

    async def get_version(self, source_id: UUID, version: int) -> Optional[dict]:
        """Retourne le détail complet d'une version (snapshot + deltas)."""
        row = await self.pg.fetchrow("""
            SELECT *
            FROM   schema_versions
            WHERE  source_id = $1 AND version_number = $2
        """, source_id, version)

        if not row:
            return None

        tags = await self.pg.fetch("""
            SELECT tag, note, created_at
            FROM   schema_version_tags
            WHERE  source_id = $1 AND version_number = $2
        """, source_id, version)

        snap = row["schema_snapshot"]
        delt = row["changes_delta"]
        summ = row["change_summary"]

        return {
            "id":               str(row["id"]),
            "source_id":        str(source_id),
            "version":          row["version_number"],
            "fingerprint":      row["fingerprint"],
            "has_breaking":     row["has_breaking_changes"],
            "is_rollback":      row["is_rollback"],
            "rollback_from":    row["rollback_from_version"],
            "schema_snapshot":  json.loads(snap) if isinstance(snap, str) else snap,
            "changes_delta":    json.loads(delt) if isinstance(delt, str) else (delt or []),
            "summary":          json.loads(summ) if isinstance(summ, str) else (summ or {}),
            "created_at":       row["created_at"].isoformat(),
            "tags": [
                {
                    "tag":        t["tag"],
                    "note":       t["note"],
                    "created_at": t["created_at"].isoformat(),
                }
                for t in tags
            ],
        }

    # ── Diff entre deux versions ──────────────────────────────

    async def diff_versions(
        self, source_id: UUID, v1: int, v2: int
    ) -> dict:
        """
        Diff entre deux versions arbitraires.
        v1 = version de base, v2 = version cible.
        """
        r1 = await self.get_version(source_id, v1)
        r2 = await self.get_version(source_id, v2)

        if not r1:
            raise ValueError(f"Version {v1} introuvable pour source {source_id}")
        if not r2:
            raise ValueError(f"Version {v2} introuvable pour source {source_id}")

        changes  = diff_schemas(r1["schema_snapshot"], r2["schema_snapshot"])
        breaking = [c for c in changes if c.get("breaking")]

        tables_affected = sorted({c["table"] for c in changes})

        return {
            "source_id":       str(source_id),
            "from_version":    v1,
            "to_version":      v2,
            "from_fp":         r1["fingerprint"][:12] + "…",
            "to_fp":           r2["fingerprint"][:12] + "…",
            "total_changes":   len(changes),
            "breaking_count":  len(breaking),
            "tables_affected": tables_affected,
            "breaking":        breaking,
            "changes":         changes,
        }

    # ── Tags ──────────────────────────────────────────────────

    async def tag_version(
        self,
        source_id: UUID,
        version:   int,
        tag:       str,
        note:      str = "",
    ) -> dict:
        """
        Pose un tag nommé sur une version (équivalent git tag).
        Upsert : si le tag existe déjà, le déplace sur la nouvelle version.
        """
        # Vérifie que la version existe
        exists = await self.pg.fetchval("""
            SELECT 1 FROM schema_versions
            WHERE source_id=$1 AND version_number=$2
        """, source_id, version)
        if not exists:
            raise ValueError(f"Version {version} introuvable pour source {source_id}")

        await self.pg.execute("""
            INSERT INTO schema_version_tags
                (source_id, version_number, tag, note, created_at)
            VALUES ($1,$2,$3,$4,NOW())
            ON CONFLICT (source_id, tag)
            DO UPDATE SET version_number=$2, note=$4, created_at=NOW()
        """, source_id, version, tag, note)

        return {
            "success":  True,
            "tag":      tag,
            "version":  version,
            "note":     note,
        }

    async def delete_tag(self, source_id: UUID, tag: str) -> dict:
        """Supprime un tag."""
        result = await self.pg.execute("""
            DELETE FROM schema_version_tags
            WHERE source_id=$1 AND tag=$2
        """, source_id, tag)

        if result == "DELETE 0":
            raise ValueError(f"Tag '{tag}' introuvable pour source {source_id}")
        return {"success": True, "deleted_tag": tag}

    async def list_tags(self, source_id: UUID) -> list[dict]:
        """Liste tous les tags d'une source."""
        rows = await self.pg.fetch("""
            SELECT tag, version_number, note, created_at
            FROM   schema_version_tags
            WHERE  source_id = $1
            ORDER  BY version_number DESC
        """, source_id)
        return [
            {
                "tag":        r["tag"],
                "version":    r["version_number"],
                "note":       r["note"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]

    # ── Notifications breaking changes ────────────────────────

    async def get_notifications(
        self, source_id: UUID, limit: int = 50
    ) -> list[dict]:
        """
        Récupère les dernières notifications de breaking changes
        depuis Redis (liste LIFO, TTL 7 jours).
        Fallback sur PostgreSQL si Redis indisponible.
        """
        # 1. Essaie Redis
        try:
            key   = f"cdc:notifications:{source_id}"
            items = await self.redis.lrange(key, 0, limit - 1)
            return [json.loads(i) for i in items]
        except Exception as e:
            logger.warning(f"[Versioner] Redis unavailable, fallback PG: {e}")

        # 2. Fallback PostgreSQL
        rows = await self.pg.fetch("""
            SELECT version_number, changes_delta, created_at
            FROM   schema_versions
            WHERE  source_id            = $1
              AND  has_breaking_changes = TRUE
            ORDER  BY version_number DESC
            LIMIT  $2
        """, source_id, limit)

        result = []
        for r in rows:
            delta = r["changes_delta"]
            changes = json.loads(delta) if isinstance(delta, str) else (delta or [])
            breaking = [c for c in changes if c.get("breaking")]
            result.append({
                "event":     "BREAKING_CHANGE",
                "source_id": str(source_id),
                "version":   r["version_number"],
                "timestamp": r["created_at"].isoformat(),
                "count":     len(breaking),
                "changes":   breaking,
            })
        return result

    # ── Impact analysis ───────────────────────────────────────

    async def impact_analysis(
        self, source_id: UUID, version: int
    ) -> dict:
        """
        Analyse l'impact des breaking changes d'une version.
        Retourne les entités potentiellement affectées (relations, jointures).
        """
        v = await self.get_version(source_id, version)
        if not v:
            raise ValueError(f"Version {version} introuvable")

        breaking = [c for c in v["changes_delta"] if c.get("breaking")]
        if not breaking:
            return {
                "version":  version,
                "breaking": 0,
                "impact":   [],
            }

        # Cherche les relations impliquant les tables/colonnes affectées
        affected_tables = {c["table"] for c in breaking}
        impact_rows = await self.pg.fetch("""
            SELECT DISTINCT
                se1.name AS from_table,
                er.source_field,
                se2.name AS to_table,
                er.target_field,
                er.relation_type,
                er.confidence
            FROM   entity_relations er
            JOIN   source_entities  se1 ON se1.id = er.source_entity_id
            JOIN   source_entities  se2 ON se2.id = er.target_entity_id
            WHERE  er.source_id = $1
              AND  (se1.name = ANY($2) OR se2.name = ANY($2))
        """, source_id, list(affected_tables))

        return {
            "version":         version,
            "breaking":        len(breaking),
            "affected_tables": sorted(affected_tables),
            "impacted_joins":  [
                {
                    "from_table":   r["from_table"],
                    "source_field": r["source_field"],
                    "to_table":     r["to_table"],
                    "target_field": r["target_field"],
                    "relation_type": r["relation_type"],
                    "confidence":   r["confidence"],
                }
                for r in impact_rows
            ],
            "breaking_changes": breaking,
        }