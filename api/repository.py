"""
OnePilot – Source Repository
CRUD PostgreSQL + Cache Redis pour les data sources
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timezone

from .database import pg_conn, cache_get, cache_set, cache_invalidate
from .schemas  import (
    DataSourceCreate, DataSourceUpdate, DataSourceOut,
    DataSourceDetail, EntityOut, FieldOut, RelationOut,
    ConnectionTestResult, MetadataSyncResult,
    CONNECTOR_CATEGORY_MAP, SourceCategory, ConnectionStatus
)

logger = logging.getLogger(__name__)


def _row_to_source(row: dict) -> DataSourceOut:
    return DataSourceOut(
        id=row["id"],
        name=row["name"],
        description=row.get("description"),
        category=row["category"],
        connector_type=row["connector_type"],
        status=row["status"],
        host=row.get("host"),
        port=row.get("port"),
        database_name=row.get("database_name"),
        schema_name=row.get("schema_name"),
        base_url=row.get("base_url"),
        auth_type=row["auth_type"],
        username=row.get("username"),
        options=json.loads(row.get("options") or "{}") if isinstance(row.get("options"), str) else (row.get("options") or {}),
        tags=list(row.get("tags") or []),
        entity_count=row.get("entity_count") or 0,
        test_latency_ms=row.get("test_latency_ms"),
        error_message=row.get("error_message"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_tested_at=row.get("last_tested_at"),
        last_synced_at=row.get("last_synced_at"),
    )


# ── CREATE ────────────────────────────────────────────────────

async def create_source(data: DataSourceCreate) -> DataSourceOut:
    category = CONNECTOR_CATEGORY_MAP.get(data.connector_type, SourceCategory.DATABASE)

    async with pg_conn() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO data_sources
                (name, description, category, connector_type,
                 host, port, database_name, schema_name, base_url,
                 auth_type, username, options, tags)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            RETURNING *
            """,
            data.name, data.description,
            category.value, data.connector_type.value,
            data.host, data.port, data.database_name,
            data.schema_name, data.base_url,
            data.auth_type.value, data.username,
            json.dumps(data.options), data.tags,
        )

        source_id = row["id"]
        secrets: Dict[str, Optional[str]] = {
            "password":      data.password,
            "token":         data.token,
            "client_secret": data.client_secret,
            "api_key_value": data.api_key_value,
        }
        for key, value in secrets.items():
            if value:
                await conn.execute(
                    """
                    INSERT INTO connection_secrets (source_id, secret_key, secret_value)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (source_id, secret_key)
                    DO UPDATE SET secret_value=$3, updated_at=NOW()
                    """,
                    source_id, key, value
                )

    await cache_invalidate("sources")
    return _row_to_source(dict(row))


# ── READ ──────────────────────────────────────────────────────

async def list_sources(
    category: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
) -> List[DataSourceOut]:
    cache_key = f"sources:list:{category}:{status}:{search}"
    cached = await cache_get(cache_key)
    if cached:
        return [DataSourceOut(**s) for s in cached]

    conditions = []
    params: List[Any] = []
    i = 1
    if category:
        conditions.append(f"category = ${i}"); params.append(category); i += 1
    if status:
        conditions.append(f"status = ${i}");   params.append(status);   i += 1
    if search:
        conditions.append(f"(name ILIKE ${i} OR description ILIKE ${i})")
        params.append(f"%{search}%"); i += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT * FROM data_sources {where} ORDER BY created_at DESC"

    async with pg_conn() as conn:
        rows = await conn.fetch(sql, *params)

    sources = [_row_to_source(dict(r)) for r in rows]
    await cache_set(cache_key, [s.model_dump(mode="json") for s in sources], ttl=60)
    return sources


async def get_source(source_id: UUID) -> Optional[DataSourceOut]:
    cache_key = f"source:{source_id}"
    cached = await cache_get(cache_key)
    if cached:
        return DataSourceOut(**cached)

    async with pg_conn() as conn:
        row = await conn.fetchrow("SELECT * FROM data_sources WHERE id=$1", source_id)

    if not row:
        return None
    source = _row_to_source(dict(row))
    await cache_set(cache_key, source.model_dump(mode="json"))
    return source


async def get_source_with_entities(
    source_id: UUID,
    page: int = 1,
    page_size: int = 50,
    search: str = "",
) -> Optional[DataSourceDetail]:
    """
    Retourne une source avec ses entités paginées.
    page      : numéro de page (commence à 1)
    page_size : nombre d'entités par page (max 200)
    search    : filtre sur le nom de l'entité
    """
    source = await get_source(source_id)
    if not source:
        return None

    offset = (page - 1) * page_size
    search_filter = f"%{search}%" if search else "%"

    async with pg_conn() as conn:
        # Nombre total d'entités (pour la pagination)
        total_entities = await conn.fetchval(
            """SELECT COUNT(*) FROM source_entities
               WHERE source_id=$1 AND is_visible=TRUE AND name ILIKE $2""",
            source_id, search_filter
        )

        # Entités de la page courante
        entity_rows = await conn.fetch(
            """SELECT * FROM source_entities
               WHERE source_id=$1 AND is_visible=TRUE AND name ILIKE $2
               ORDER BY name LIMIT $3 OFFSET $4""",
            source_id, search_filter, page_size, offset
        )

        # Champs de toutes les entités de la page en une seule requête
        entity_ids = [er["id"] for er in entity_rows]
        field_rows = []
        if entity_ids:
            field_rows = await conn.fetch(
                """SELECT * FROM entity_fields
                   WHERE entity_id = ANY($1)
                   ORDER BY entity_id, position, name""",
                entity_ids
            )

    # Grouper les champs par entité
    fields_by_entity: Dict[UUID, List[FieldOut]] = {}
    for fr in field_rows:
        eid = fr["entity_id"]
        if eid not in fields_by_entity:
            fields_by_entity[eid] = []
        fields_by_entity[eid].append(FieldOut(
            id=fr["id"],
            name=fr["name"],
            display_name=fr.get("display_name"),
            data_type=fr["data_type"],
            native_type=fr.get("native_type"),
            is_nullable=fr["is_nullable"],
            is_primary_key=fr["is_primary_key"],
            is_foreign_key=fr["is_foreign_key"],
            position=fr["position"],
        ))

    entities = []
    for er in entity_rows:
        fields = fields_by_entity.get(er["id"], [])
        entities.append(EntityOut(
            id=er["id"],
            name=er["name"],
            display_name=er.get("display_name"),
            entity_type=er["entity_type"],
            description=er.get("description"),
            row_count=er.get("row_count"),
            field_count=len(fields),
            fields=fields,
        ))

    detail = DataSourceDetail(**source.model_dump(), entities=entities)
    # Ajouter les infos de pagination dans le modèle via un attribut dynamique
    detail.__dict__["_pagination"] = {
        "page": page,
        "page_size": page_size,
        "total_entities": total_entities,
        "total_pages": max(1, -(-total_entities // page_size)),  # ceil division
        "search": search,
    }
    return detail


async def get_source_secrets(source_id: UUID) -> Dict[str, str]:
    async with pg_conn() as conn:
        rows = await conn.fetch(
            "SELECT secret_key, secret_value FROM connection_secrets WHERE source_id=$1",
            source_id
        )
    return {r["secret_key"]: r["secret_value"] for r in rows}


# ── UPDATE ────────────────────────────────────────────────────

async def update_source(source_id: UUID, data: DataSourceUpdate) -> Optional[DataSourceOut]:
    updates: List[str] = []
    params: List[Any] = []
    i = 1

    fields_map = {
        "name":          data.name,
        "description":   data.description,
        "host":          data.host,
        "port":          data.port,
        "database_name": data.database_name,
        "schema_name":   data.schema_name,
        "base_url":      data.base_url,
        "auth_type":     data.auth_type.value if data.auth_type else None,
        "username":      data.username,
        "options":       json.dumps(data.options) if data.options else None,
        "tags":          data.tags,
    }
    for col, val in fields_map.items():
        if val is not None:
            updates.append(f"{col}=${i}"); params.append(val); i += 1

    if not updates:
        return await get_source(source_id)

    params.append(source_id)
    sql = f"UPDATE data_sources SET {', '.join(updates)} WHERE id=${i} RETURNING *"

    async with pg_conn() as conn:
        row = await conn.fetchrow(sql, *params)
        if data.password:
            await conn.execute(
                "INSERT INTO connection_secrets (source_id, secret_key, secret_value) "
                "VALUES ($1,'password',$2) ON CONFLICT (source_id, secret_key) "
                "DO UPDATE SET secret_value=$2, updated_at=NOW()",
                source_id, data.password
            )

    await cache_invalidate(f"source:{source_id}")
    await cache_invalidate("sources:list")
    return _row_to_source(dict(row)) if row else None


# ── DELETE ────────────────────────────────────────────────────

async def delete_source(source_id: UUID) -> bool:
    async with pg_conn() as conn:
        result = await conn.execute("DELETE FROM data_sources WHERE id=$1", source_id)
    await cache_invalidate(f"source:{source_id}")
    await cache_invalidate("sources:list")
    return result == "DELETE 1"


# ── CONNECTION TEST ───────────────────────────────────────────

async def save_test_result(source_id: UUID, success: bool, message: str, latency_ms: int) -> ConnectionTestResult:
    now = datetime.now(timezone.utc)
    async with pg_conn() as conn:
        await conn.execute(
            "INSERT INTO connection_tests (source_id, success, latency_ms, message) VALUES ($1,$2,$3,$4)",
            source_id, success, latency_ms, message
        )
        status = ConnectionStatus.ACTIVE if success else ConnectionStatus.ERROR
        await conn.execute(
            """UPDATE data_sources
               SET status=$1, last_tested_at=$2, test_latency_ms=$3, error_message=$4
               WHERE id=$5""",
            status.value, now, latency_ms if success else None,
            None if success else message, source_id
        )
    await cache_invalidate(f"source:{source_id}")
    await cache_invalidate("sources:list")
    return ConnectionTestResult(
        source_id=source_id, success=success,
        message=message, latency_ms=latency_ms, tested_at=now
    )


# ── METADATA SYNC ─────────────────────────────────────────────

async def save_metadata(source_id: UUID, entities_data: list) -> int:
    entity_count = 0
    async with pg_conn() as conn:
        # Collecter les noms des entités reçues
        incoming_names = {e.get("name", "") for e in entities_data}

        # Supprimer uniquement les entités qui n'existent plus dans la source
        await conn.execute("""
            DELETE FROM source_entities
            WHERE source_id = $1
              AND name NOT IN (SELECT unnest($2::text[]))
        """, source_id, list(incoming_names))

        for entity in entities_data:
            indexes_json  = json.dumps(entity.get("indexes", []))
            metadata_json = json.dumps(entity.get("metadata", {}))
            # UPSERT : préserve la description LLM existante si elle existe
            row = await conn.fetchrow(
                """INSERT INTO source_entities
                   (source_id, name, display_name, entity_type, description,
                    indexes, metadata, row_count)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                   ON CONFLICT (source_id, name)
                   DO UPDATE SET
                       display_name = EXCLUDED.display_name,
                       entity_type  = EXCLUDED.entity_type,
                       description  = CASE
                           WHEN source_entities.description IS NOT NULL
                            AND source_entities.description != ''
                           THEN source_entities.description
                           ELSE EXCLUDED.description
                       END,
                       indexes      = EXCLUDED.indexes,
                       metadata     = EXCLUDED.metadata,
                       row_count    = EXCLUDED.row_count
                   RETURNING id""",
                source_id,
                entity.get("name", ""),
                entity.get("name", "").replace("_", " ").title(),
                entity.get("entity_type", "table"),
                entity.get("description"),
                indexes_json,
                metadata_json,
                entity.get("row_count"),
            )
            entity_id = row["id"]
            entity_count += 1

            for pos, field in enumerate(entity.get("fields", [])):
                await conn.execute(
                    """INSERT INTO entity_fields
                       (entity_id, name, display_name, data_type, native_type,
                        is_nullable, is_primary_key, is_foreign_key, position,
                        description, default_value)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                       ON CONFLICT (entity_id, name) DO UPDATE
                       SET data_type=$4, native_type=$5, is_nullable=$6,
                           is_primary_key=$7, is_foreign_key=$8,
                           description=COALESCE($10, entity_fields.description),
                           default_value=COALESCE($11, entity_fields.default_value)""",
                    entity_id, field.get("name", ""),
                    field.get("name", "").replace("_", " ").title(),
                    field.get("type", "string"),
                    field.get("native_type"),
                    field.get("nullable", True),
                    field.get("primary_key", False),
                    field.get("foreign_key", False),
                    pos,
                    field.get("description"),
                    field.get("default_value"),
                )

        await conn.execute(
            "UPDATE data_sources SET entity_count=$1, last_synced_at=NOW() WHERE id=$2",
            entity_count, source_id
        )

    await cache_invalidate(f"source:{source_id}")
    await cache_invalidate("sources:list")
    return entity_count