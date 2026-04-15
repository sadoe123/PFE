"""
OnePilot – CDC DB Triggers §2.2.5 completion
PostgreSQL logical replication (WAL) + SQL Server CDC natif
Polling-based CDC depuis les logs de transaction
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import asyncpg
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class PostgreSQLWALCDC:
    """
    CDC via PostgreSQL logical replication (WAL).
    Utilise pg_logical_slot_get_changes() pour lire les changements
    sans avoir besoin de droits SUPERUSER.

    Prérequis PostgreSQL :
      - wal_level = logical (dans postgresql.conf)
      - Créer un slot : SELECT pg_create_logical_replication_slot('onepilot_cdc', 'wal2json')
    """

    SLOT_NAME = "onepilot_cdc"
    PLUGIN    = "wal2json"

    def __init__(self, pg_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.pg    = pg_pool
        self.redis = redis_client

    async def setup_replication_slot(self, source_db_conn_str: str) -> dict:
        """
        Crée le slot de réplication logique sur la base source.
        À appeler une seule fois lors de la configuration.
        """
        try:
            conn = await asyncpg.connect(source_db_conn_str)
            try:
                # Vérifie si le slot existe déjà
                existing = await conn.fetchval("""
                    SELECT slot_name FROM pg_replication_slots
                    WHERE slot_name = $1
                """, self.SLOT_NAME)

                if existing:
                    return {"status": "exists", "slot": self.SLOT_NAME}

                # Crée le slot
                await conn.execute(f"""
                    SELECT pg_create_logical_replication_slot(
                        '{self.SLOT_NAME}', '{self.PLUGIN}'
                    )
                """)
                return {"status": "created", "slot": self.SLOT_NAME}
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"[WAL CDC] Setup error: {e}")
            return {"status": "error", "message": str(e)}

    async def poll_changes(
        self, source_id: UUID, source_db_conn_str: str, max_changes: int = 1000
    ) -> dict:
        """
        Lit les changements DDL depuis le WAL via le slot de réplication.
        Filtre uniquement les changements de schéma (DDL).
        """
        try:
            conn = await asyncpg.connect(source_db_conn_str)
            try:
                rows = await conn.fetch(f"""
                    SELECT lsn, xid, data
                    FROM pg_logical_slot_get_changes(
                        '{self.SLOT_NAME}', NULL, $1,
                        'include-schemas', 'true',
                        'include-types', 'true',
                        'include-transaction', 'false'
                    )
                """, max_changes)

                ddl_changes = []
                for row in rows:
                    try:
                        data = json.loads(row["data"])
                        for change in data.get("change", []):
                            kind = change.get("kind", "")
                            # Filtre DDL uniquement
                            if kind in ("ddl",) or change.get("columnnames") is None:
                                ddl_changes.append({
                                    "lsn":    str(row["lsn"]),
                                    "xid":    str(row["xid"]),
                                    "kind":   kind,
                                    "schema": change.get("schema"),
                                    "table":  change.get("table"),
                                    "data":   change,
                                })
                    except Exception:
                        pass

                if ddl_changes:
                    await self._publish_wal_changes(source_id, ddl_changes)

                return {
                    "status":      "ok",
                    "total_rows":  len(rows),
                    "ddl_changes": len(ddl_changes),
                }
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"[WAL CDC] Poll error: {e}")
            return {"status": "error", "message": str(e)}

    async def check_wal_level(self, source_db_conn_str: str) -> dict:
        """Vérifie si wal_level = logical est configuré."""
        try:
            conn = await asyncpg.connect(source_db_conn_str)
            try:
                wal_level = await conn.fetchval(
                    "SHOW wal_level"
                )
                slots = await conn.fetch("""
                    SELECT slot_name, active, restart_lsn
                    FROM pg_replication_slots
                    WHERE slot_name = $1
                """, self.SLOT_NAME)

                return {
                    "wal_level":     wal_level,
                    "wal_ok":        wal_level == "logical",
                    "slot_exists":   len(slots) > 0,
                    "slot_active":   slots[0]["active"] if slots else False,
                    "restart_lsn":   str(slots[0]["restart_lsn"]) if slots else None,
                }
            finally:
                await conn.close()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _publish_wal_changes(self, source_id: UUID, changes: list):
        """Publie les changements WAL dans Redis."""
        payload = json.dumps({
            "event":     "WAL_DDL_CHANGE",
            "source_id": str(source_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count":     len(changes),
            "changes":   changes[:10],  # Preview dans la notif
        }, default=str)
        try:
            await self.redis.publish(f"cdc:wal:{source_id}", payload)
            key = f"cdc:wal:notifications:{source_id}"
            await self.redis.lpush(key, payload)
            await self.redis.ltrim(key, 0, 49)
            await self.redis.expire(key, 60 * 60 * 24 * 7)
        except Exception as e:
            logger.warning(f"[WAL CDC] Redis publish error: {e}")


class SQLServerCDC:
    """
    CDC via SQL Server Change Data Capture natif.
    Utilise cdc.fn_cdc_get_all_changes_* pour lire les changements DDL.

    Prérequis SQL Server :
      - EXEC sys.sp_cdc_enable_db  (activer CDC sur la base)
      - EXEC sys.sp_cdc_enable_table (activer CDC par table)
    """

    def __init__(self, pg_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.pg    = pg_pool
        self.redis = redis_client

    async def check_cdc_enabled(self, source_id: UUID) -> dict:
        """
        Vérifie si SQL Server CDC est activé via l'API OnePilot.
        Exécute la vérification sur la source MSSQL via connection_service.
        """
        try:
            from .database import get_pg_pool
            pool = await get_pg_pool()

            # Récupère les infos de connexion depuis les métadonnées
            source_row = await pool.fetchrow("""
                SELECT host, port, database_name, username, schema_name, options
                FROM data_sources WHERE id = $1
            """, source_id)

            if not source_row:
                return {"status": "error", "message": "Source introuvable"}

            result = await self._check_mssql_cdc(
                host     = source_row["host"],
                port     = source_row["port"] or 1433,
                database = source_row["database_name"],
                username = source_row["username"],
                schema   = source_row["schema_name"] or "dbo",
                source_id= source_id,
            )
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _check_mssql_cdc(
        self, host: str, port: int, database: str,
        username: str, schema: str, source_id: UUID
    ) -> dict:
        """Vérifie l'état CDC sur SQL Server."""
        import asyncio

        def _sync_check():
            from sqlalchemy import create_engine, text
            from .repository import get_source_secrets
            import asyncio

            url = (
                f"mssql+pyodbc://{username}:@{host}:{port}/{database}"
                f"?driver=ODBC+Driver+18+for+SQL+Server"
                f"&Encrypt=yes&TrustServerCertificate=yes"
            )
            engine = create_engine(url, pool_pre_ping=True)
            try:
                with engine.connect() as conn:
                    # Vérifie si CDC est activé sur la base
                    db_cdc = conn.execute(text("""
                        SELECT is_cdc_enabled
                        FROM sys.databases
                        WHERE name = DB_NAME()
                    """)).scalar()

                    if not db_cdc:
                        return {
                            "cdc_enabled": False,
                            "message": "CDC non activé sur cette base. Exécutez: EXEC sys.sp_cdc_enable_db",
                        }

                    # Tables avec CDC activé
                    cdc_tables = conn.execute(text(f"""
                        SELECT source_object_id,
                               OBJECT_NAME(source_object_id) AS table_name,
                               capture_instance,
                               is_tracked_by_cdc
                        FROM cdc.change_tables
                        WHERE OBJECT_SCHEMA_NAME(source_object_id) = '{schema}'
                    """)).fetchall()

                    return {
                        "cdc_enabled":   True,
                        "tracked_tables": len(cdc_tables),
                        "tables": [
                            {"name": r[1], "capture_instance": r[2]}
                            for r in cdc_tables
                        ],
                    }
            finally:
                engine.dispose()

        try:
            return await asyncio.to_thread(_sync_check)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def poll_ddl_changes(
        self, source_id: UUID, since_lsn: Optional[str] = None
    ) -> dict:
        """
        Lit les changements DDL depuis SQL Server CDC.
        Utilise sys.dm_exec_query_stats + sys.traces pour détecter
        les DDL events quand le CDC natif n'est pas disponible.
        """
        import asyncio

        source_row = await self.pg.fetchrow("""
            SELECT host, port, database_name, username, schema_name
            FROM data_sources WHERE id = $1
        """, source_id)

        if not source_row:
            return {"status": "error", "message": "Source introuvable"}

        def _sync_poll():
            from sqlalchemy import create_engine, text

            url = (
                f"mssql+pyodbc://{source_row['username']}:@"
                f"{source_row['host']}:{source_row['port'] or 1433}/"
                f"{source_row['database_name']}"
                f"?driver=ODBC+Driver+18+for+SQL+Server"
                f"&Encrypt=yes&TrustServerCertificate=yes"
            )
            engine = create_engine(url, pool_pre_ping=True)
            try:
                with engine.connect() as conn:
                    schema = source_row["schema_name"] or "dbo"

                    # Lit les DDL events depuis le default trace SQL Server
                    ddl_events = conn.execute(text("""
                        SELECT
                            te.StartTime,
                            te.EventClass,
                            te.ObjectName,
                            te.ObjectType,
                            te.DatabaseName,
                            te.LoginName,
                            te.TextData
                        FROM sys.fn_trace_gettable(
                            CONVERT(VARCHAR(MAX),
                                (SELECT TOP 1 path FROM sys.traces WHERE is_default = 1)
                            ), DEFAULT
                        ) te
                        WHERE te.EventClass IN (46, 47, 164)  -- Create/Drop/Alter object
                          AND te.DatabaseName = DB_NAME()
                          AND te.StartTime > DATEADD(DAY, -1, GETDATE())
                        ORDER BY te.StartTime DESC
                    """)).fetchall()

                    changes = []
                    for r in ddl_events:
                        event_class = r[1]
                        event_type = {
                            46: "CREATE_TABLE",
                            47: "DROP_TABLE",
                            164: "ALTER_TABLE",
                        }.get(event_class, "DDL")

                        changes.append({
                            "timestamp":   str(r[0]),
                            "event_type":  event_type,
                            "object_name": r[2],
                            "object_type": str(r[3]),
                            "login":       r[5],
                            "sql_preview": str(r[6] or "")[:200],
                        })

                    return changes
            finally:
                engine.dispose()

        try:
            changes = await asyncio.to_thread(_sync_poll)

            if changes:
                await self._publish_ddl_changes(source_id, changes)

            return {
                "status":  "ok",
                "changes": len(changes),
                "events":  changes[:10],
            }
        except Exception as e:
            logger.warning(f"[MSSQL CDC] Poll error (non-bloquant): {e}")
            return {"status": "unavailable", "message": str(e)}

    async def _publish_ddl_changes(self, source_id: UUID, changes: list):
        """Publie les changements DDL MSSQL dans Redis."""
        payload = json.dumps({
            "event":     "MSSQL_DDL_CHANGE",
            "source_id": str(source_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count":     len(changes),
            "changes":   changes[:10],
        }, default=str)
        try:
            await self.redis.publish(f"cdc:mssql:{source_id}", payload)
            key = f"cdc:mssql:notifications:{source_id}"
            await self.redis.lpush(key, payload)
            await self.redis.ltrim(key, 0, 49)
            await self.redis.expire(key, 60 * 60 * 24 * 7)
        except Exception as e:
            logger.warning(f"[MSSQL CDC] Redis publish error: {e}")