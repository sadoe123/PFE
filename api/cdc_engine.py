"""
OnePilot – CDC Engine §2.2.5
Change Data Capture : détection, diff, rollback, notifications
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID

import asyncpg
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


# ── Types de changements ──────────────────────────────────────

class ChangeType(str, Enum):
    CREATE_TABLE       = "CREATE_TABLE"
    DROP_TABLE         = "DROP_TABLE"
    ALTER_ADD          = "ALTER_ADD_COLUMN"
    ALTER_DROP         = "ALTER_DROP_COLUMN"
    ALTER_TYPE         = "ALTER_COLUMN_TYPE"
    ALTER_NULLABLE     = "ALTER_COLUMN_NULLABLE"
    ALTER_DEFAULT      = "ALTER_COLUMN_DEFAULT"
    CONSTRAINT_ADDED   = "CONSTRAINT_ADDED"
    CONSTRAINT_DROPPED = "CONSTRAINT_DROPPED"
    INDEX_ADDED        = "INDEX_ADDED"
    INDEX_REMOVED      = "INDEX_REMOVED"
    ROW_COUNT_CHANGE   = "ROW_COUNT_CHANGE"


BREAKING_CHANGE_TYPES = {
    ChangeType.DROP_TABLE,
    ChangeType.ALTER_DROP,
    ChangeType.ALTER_TYPE,
    ChangeType.ALTER_NULLABLE,
    ChangeType.CONSTRAINT_DROPPED,
}


# ── Utilitaires ───────────────────────────────────────────────

def compute_schema_fingerprint(schema: dict) -> str:
    canonical = json.dumps(schema, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()


def diff_schemas(old: dict, new: dict) -> list[dict]:
    changes: list[dict] = []
    old_tables = set(old.keys())
    new_tables = set(new.keys())

    for t in sorted(new_tables - old_tables):
        changes.append({"type": ChangeType.CREATE_TABLE, "table": t,
                         "breaking": False, "detail": {"columns": new[t].get("columns", [])}})

    for t in sorted(old_tables - new_tables):
        changes.append({"type": ChangeType.DROP_TABLE, "table": t,
                         "breaking": True, "detail": {"columns": old[t].get("columns", [])}})

    for t in sorted(old_tables & new_tables):
        old_t = old[t]
        new_t = new[t]
        old_cols = {c["column_name"]: c for c in old_t.get("columns", [])}
        new_cols = {c["column_name"]: c for c in new_t.get("columns", [])}

        for col in sorted(set(new_cols) - set(old_cols)):
            changes.append({"type": ChangeType.ALTER_ADD, "table": t, "column": col,
                             "breaking": False, "detail": new_cols[col]})

        for col in sorted(set(old_cols) - set(new_cols)):
            changes.append({"type": ChangeType.ALTER_DROP, "table": t, "column": col,
                             "breaking": True, "detail": old_cols[col]})

        for col in sorted(set(old_cols) & set(new_cols)):
            o = old_cols[col]
            n = new_cols[col]
            if o.get("data_type") != n.get("data_type"):
                changes.append({"type": ChangeType.ALTER_TYPE, "table": t, "column": col,
                                 "breaking": True,
                                 "detail": {"old_type": o.get("data_type"), "new_type": n.get("data_type")}})
            if o.get("is_nullable") != n.get("is_nullable"):
                changes.append({"type": ChangeType.ALTER_NULLABLE, "table": t, "column": col,
                                 "breaking": n.get("is_nullable") == "NO",
                                 "detail": {"old_nullable": o.get("is_nullable"), "new_nullable": n.get("is_nullable")}})
            if o.get("column_default") != n.get("column_default"):
                changes.append({"type": ChangeType.ALTER_DEFAULT, "table": t, "column": col,
                                 "breaking": False,
                                 "detail": {"old_default": o.get("column_default"), "new_default": n.get("column_default")}})

        old_idx = set(old_t.get("indexes", []))
        new_idx = set(new_t.get("indexes", []))
        for idx in sorted(new_idx - old_idx):
            changes.append({"type": ChangeType.INDEX_ADDED,   "table": t, "breaking": False, "detail": {"index": idx}})
        for idx in sorted(old_idx - new_idx):
            changes.append({"type": ChangeType.INDEX_REMOVED, "table": t, "breaking": False, "detail": {"index": idx}})

        old_cst = set(old_t.get("constraints", []))
        new_cst = set(new_t.get("constraints", []))
        for cst in sorted(new_cst - old_cst):
            changes.append({"type": ChangeType.CONSTRAINT_ADDED,   "table": t, "breaking": False, "detail": {"constraint": cst}})
        for cst in sorted(old_cst - new_cst):
            changes.append({"type": ChangeType.CONSTRAINT_DROPPED, "table": t, "breaking": True,  "detail": {"constraint": cst}})

        old_rc = old_t.get("row_count")
        new_rc = new_t.get("row_count")
        if old_rc is not None and new_rc is not None and old_rc != new_rc:
            delta_pct = round((new_rc - old_rc) / max(old_rc, 1) * 100, 2)
            if abs(delta_pct) >= 10:
                changes.append({"type": ChangeType.ROW_COUNT_CHANGE, "table": t, "breaking": False,
                                 "detail": {"old_count": old_rc, "new_count": new_rc, "delta_pct": delta_pct}})

    return changes


def _build_summary(changes: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for c in changes:
        ct = c["type"]
        counts[ct] = counts.get(ct, 0) + 1
    tables_affected = {c["table"] for c in changes}
    breaking        = [c for c in changes if c.get("breaking")]
    return {
        "total":           len(changes),
        "breaking":        len(breaking),
        "tables_affected": sorted(tables_affected),
        "by_type":         counts,
    }


# ── Moteur principal ──────────────────────────────────────────

class CDCEngine:
    def __init__(self, pg_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.pg    = pg_pool
        self.redis = redis_client

    async def snapshot_schema(self, source_id: UUID) -> dict:
        """
        Snapshot du schéma courant depuis source_entities + entity_fields.
        CORRECTION : entity_relations utilise source_entity / target_entity (VARCHAR)
        et non source_entity_id / target_entity_id (UUID).
        """
        col_rows = await self.pg.fetch("""
            SELECT
                se.name               AS table_name,
                ef.name               AS column_name,
                ef.data_type,
                ef.native_type,
                ef.is_nullable,
                ef.is_primary_key,
                ef.is_foreign_key,
                ef.position           AS ordinal_position
            FROM   source_entities se
            JOIN   entity_fields   ef ON ef.entity_id = se.id
            WHERE  se.source_id  = $1
              AND  se.entity_type IN ('table', 'view', 'materialized_view')
            ORDER  BY se.name, ef.position
        """, source_id)

        schema: dict = {}
        for r in col_rows:
            t = r["table_name"]
            if t not in schema:
                schema[t] = {"columns": [], "indexes": [], "constraints": [], "row_count": None}
            schema[t]["columns"].append({
                "column_name":      r["column_name"],
                "data_type":        r["data_type"],
                "native_type":      r["native_type"],
                "is_nullable":      "YES" if r["is_nullable"] else "NO",
                "is_primary_key":   r["is_primary_key"],
                "is_foreign_key":   r["is_foreign_key"],
                "ordinal_position": r["ordinal_position"],
                "column_default":   None,
            })

        entity_rows = await self.pg.fetch("""
            SELECT name, row_count, indexes
            FROM   source_entities
            WHERE  source_id  = $1
              AND  entity_type IN ('table', 'view', 'materialized_view')
        """, source_id)

        for r in entity_rows:
            t = r["name"]
            if t not in schema:
                continue
            if r["row_count"] is not None:
                schema[t]["row_count"] = int(r["row_count"])
            if r["indexes"]:
                try:
                    raw = r["indexes"]
                    idx_list = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(idx_list, list):
                        schema[t]["indexes"] = [
                            i.get("name", str(i)) if isinstance(i, dict) else str(i)
                            for i in idx_list
                        ]
                except Exception:
                    pass

        # Contraintes FK — colonnes source_entity/target_entity (VARCHAR noms)
        try:
            rel_rows = await self.pg.fetch("""
                SELECT
                    er.source_entity AS table_name,
                    er.source_field,
                    er.target_entity AS target_table,
                    er.target_field
                FROM   entity_relations er
                WHERE  er.source_id    = $1
                  AND  er.is_confirmed = TRUE
            """, source_id)

            for r in rel_rows:
                t = r["table_name"]
                if t in schema:
                    cst_name = f"fk_{t}_{r['source_field']}__{r['target_table']}"
                    if cst_name not in schema[t]["constraints"]:
                        schema[t]["constraints"].append(cst_name)
        except Exception as e:
            logger.warning(f"[CDC] Contraintes FK ignorées (non-bloquant): {e}")

        return schema

    async def detect_and_record(self, source_id: UUID) -> dict:
        current_schema = await self.snapshot_schema(source_id)
        current_fp     = compute_schema_fingerprint(current_schema)

        last = await self.pg.fetchrow("""
            SELECT version_number, schema_snapshot, fingerprint
            FROM   schema_versions
            WHERE  source_id = $1
            ORDER  BY version_number DESC
            LIMIT  1
        """, source_id)

        if last and last["fingerprint"] == current_fp:
            logger.info(f"[CDC] source {source_id} — aucun changement (fp={current_fp[:8]})")
            return {
                "status":          "no_change",
                "fingerprint":     current_fp,
                "current_version": last["version_number"],
            }

        old_schema  = json.loads(last["schema_snapshot"]) if last else {}
        old_version = last["version_number"]               if last else 0
        new_version = old_version + 1

        changes  = diff_schemas(old_schema, current_schema)
        breaking = [c for c in changes if c.get("breaking")]
        summary  = _build_summary(changes)

        await self.pg.execute("""
            INSERT INTO schema_versions (
                source_id, version_number, fingerprint,
                schema_snapshot, changes_delta, has_breaking_changes,
                change_summary, created_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
        """,
            source_id, new_version, current_fp,
            json.dumps(current_schema, default=str),
            json.dumps(changes,        default=str),
            len(breaking) > 0,
            json.dumps(summary),
            datetime.now(timezone.utc),
        )

        try:
            keys = await self.redis.keys(f"onepilot:*{source_id}*")
            if keys:
                await self.redis.delete(*keys)
        except Exception:
            pass

        if breaking:
            await self._publish_breaking(source_id, new_version, breaking)

        logger.info(f"[CDC] source {source_id} → v{new_version} ({len(changes)} changements, {len(breaking)} breaking)")

        return {
            "status":         "changed",
            "old_version":    old_version,
            "new_version":    new_version,
            "fingerprint":    current_fp,
            "total_changes":  len(changes),
            "breaking_count": len(breaking),
            "summary":        summary,
            "changes":        changes,
        }

    async def rollback_to_version(self, source_id: UUID, target_version: int) -> dict:
        target = await self.pg.fetchrow("""
            SELECT version_number, fingerprint, schema_snapshot
            FROM   schema_versions
            WHERE  source_id = $1 AND version_number = $2
        """, source_id, target_version)

        if not target:
            raise ValueError(f"Version {target_version} introuvable pour la source {source_id}")

        current_max = await self.pg.fetchval("""
            SELECT COALESCE(MAX(version_number), 0)
            FROM   schema_versions WHERE source_id = $1
        """, source_id)

        rollback_version = current_max + 1

        await self.pg.execute("""
            INSERT INTO schema_versions (
                source_id, version_number, fingerprint,
                schema_snapshot, changes_delta, has_breaking_changes,
                change_summary, is_rollback, rollback_from_version, created_at
            ) VALUES ($1,$2,$3,$4,'[]'::jsonb,FALSE,$5,TRUE,$6,$7)
        """,
            source_id, rollback_version, target["fingerprint"],
            target["schema_snapshot"],
            json.dumps({"message": f"Rollback vers la version {target_version}"}),
            current_max,
            datetime.now(timezone.utc),
        )

        try:
            payload = json.dumps({
                "event": "ROLLBACK", "source_id": str(source_id),
                "rollback_version": rollback_version, "restored_to": target_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await self.redis.publish(f"cdc:events:{source_id}", payload)
        except Exception:
            pass

        return {"status": "rolled_back", "rollback_version": rollback_version, "restored_to": target_version}

    async def _publish_breaking(self, source_id: UUID, version: int, breaking: list) -> None:
        payload = json.dumps({
            "event": "BREAKING_CHANGE", "source_id": str(source_id),
            "version": version, "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(breaking), "changes": breaking,
        }, default=str)

        try:
            await self.redis.publish(f"cdc:breaking:{source_id}", payload)
            key = f"cdc:notifications:{source_id}"
            await self.redis.lpush(key, payload)
            await self.redis.ltrim(key, 0, 99)
            await self.redis.expire(key, 60 * 60 * 24 * 7)
            logger.warning(f"[CDC] ⚠️  Breaking changes source {source_id} v{version}: {len(breaking)} publiés")
        except Exception as e:
            logger.error(f"[CDC] Redis publish error: {e}")

    async def incremental_sync(self, source_id: UUID) -> dict:
        return await self.detect_and_record(source_id)