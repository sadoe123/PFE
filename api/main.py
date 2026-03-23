"""
OnePilot – API FastAPI
Universal Data Access Layer – Phase 3
"""
from __future__ import annotations

import asyncio
import csv as csv_module
import io
import json as json_module
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .connection_service import sync_metadata, test_connection
from .database import close_connections, get_pg_pool, get_redis, init_schema
from .relationship_discovery import (
    discover_relationships,
    get_join_paths_for_source,
    get_relations_for_source,
    get_validation_stats,
    validate_relation,
)
from .repository import (
    create_source,
    delete_source,
    get_source,
    get_source_with_entities,
    list_sources,
    update_source,
)
from .schemas import (
    ConnectorType,
    ConnectionTestResult,
    DataSourceCreate,
    DataSourceDetail,
    DataSourceList,
    DataSourceOut,
    DataSourceUpdate,
    MetadataSyncResult,
    SourceCategory,
)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 OnePilot API démarrage...")
    try:
        await get_pg_pool()
        await init_schema()
        logger.info("✅ PostgreSQL connecté et schema initialisé")
    except Exception as e:
        logger.error(f"❌ PostgreSQL erreur: {e}")
    try:
        await get_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️  Redis non disponible: {e}")
    yield
    await close_connections()
    logger.info("👋 OnePilot API arrêté")


# ══════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="OnePilot API",
    description="Universal Data Access Layer – Phase 3",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# SCHÉMAS PYDANTIC
# ══════════════════════════════════════════════════════════════

class FkRelation(BaseModel):
    source_entity: str
    source_field:  str
    target_entity: str
    target_field:  str


class ImportFkRequest(BaseModel):
    relations: List[FkRelation]


class ImportViewsRequest(BaseModel):
    views: Optional[List[dict]] = None  # [{view_name, sql_definition}] mode manuel


class ValidateRelationRequest(BaseModel):
    is_confirmed:  bool
    validated_by:  Optional[str] = "expert"
    reject_reason: Optional[str] = None


# ══════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    return {"service": "OnePilot", "version": "3.0.0", "status": "running", "docs": "/docs"}


@app.get("/health", tags=["Health"])
async def health():
    checks = {}
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
    try:
        r = await get_redis()
        await r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"unavailable: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 207,
        content={"status": "healthy" if all_ok else "degraded", "checks": checks},
    )


# ══════════════════════════════════════════════════════════════
# META — CONNECTOR TYPES
# ══════════════════════════════════════════════════════════════

@app.get("/connector-types", tags=["Meta"])
async def get_connector_types():
    return {
        "database": {
            "label": "Base de données directe",
            "description": "Connexion directe via driver SQL",
            "icon": "database",
            "types": [
                {"id": "postgresql", "label": "PostgreSQL",  "icon": "🐘", "default_port": 5432},
                {"id": "mysql",      "label": "MySQL",       "icon": "🐬", "default_port": 3306},
                {"id": "mssql",      "label": "SQL Server",  "icon": "🪟", "default_port": 1433},
                {"id": "sqlite",     "label": "SQLite",      "icon": "📁", "default_port": None},
                {"id": "sage_100",   "label": "SAGE 100",    "icon": "🟢", "default_port": 1433},
            ],
        },
        "webservice": {
            "label": "Web Service / ERP",
            "description": "Connexion via protocole HTTP ou ERP",
            "icon": "globe",
            "types": [
                {"id": "rest",        "label": "REST API",            "icon": "🔗", "default_port": None},
                {"id": "odata",       "label": "OData",               "icon": "⚡", "default_port": None},
                {"id": "graphql",     "label": "GraphQL",             "icon": "◈",  "default_port": None},
                {"id": "soap",        "label": "SOAP / WSDL",         "icon": "📮", "default_port": None},
                {"id": "sap_rfc",     "label": "SAP RFC/BAPI",        "icon": "🔷", "default_port": 3300},
                {"id": "sap_odata",   "label": "SAP OData (S/4HANA)", "icon": "🔶", "default_port": 443},
                {"id": "dynamics365", "label": "Dynamics 365",        "icon": "🟦", "default_port": None},
                {"id": "sage_x3",     "label": "SAGE X3",             "icon": "🟩", "default_port": None},
                {"id": "sage_cloud",  "label": "SAGE Business Cloud", "icon": "☁️", "default_port": None},
            ],
        },
        "file": {
            "label": "Fichiers",
            "description": "Import de fichiers locaux ou upload",
            "icon": "file",
            "types": [
                {"id": "file_csv",     "label": "CSV",                    "icon": "📄", "default_port": None},
                {"id": "file_excel",   "label": "Excel (.xls)",            "icon": "📊", "default_port": None},
                {"id": "file_xlsx",    "label": "Excel multi-feuilles",    "icon": "📗", "default_port": None},
                {"id": "file_json",    "label": "JSON",                    "icon": "📋", "default_port": None},
                {"id": "file_parquet", "label": "Parquet (Big Data)",      "icon": "🗃", "default_port": None},
                {"id": "file_avro",    "label": "Avro (Streaming/Kafka)",  "icon": "🗄", "default_port": None},
            ],
        },
    }


# ══════════════════════════════════════════════════════════════
# SOURCES CRUD
# ══════════════════════════════════════════════════════════════

@app.post("/sources", response_model=DataSourceOut, status_code=201, tags=["Sources"])
async def create_data_source(data: DataSourceCreate):
    try:
        return await create_source(data)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"[API] create_source: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/sources", response_model=DataSourceList, tags=["Sources"])
async def list_data_sources(
    category: Optional[str] = Query(None),
    status:   Optional[str] = Query(None),
    search:   Optional[str] = Query(None),
):
    sources = await list_sources(category=category, status=status, search=search)
    return DataSourceList(total=len(sources), sources=sources)


@app.get("/sources/{source_id}", tags=["Sources"])
async def get_data_source(
    source_id: UUID,
    page:      int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search:    str = Query(""),
):
    source = await get_source_with_entities(
        source_id, page=page, page_size=page_size, search=search
    )
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    pagination = source.__dict__.get("_pagination", {})
    data = source.model_dump()
    data["pagination"] = pagination
    return data


@app.patch("/sources/{source_id}", response_model=DataSourceOut, tags=["Sources"])
async def update_data_source(source_id: UUID, data: DataSourceUpdate):
    source = await update_source(source_id, data)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    return source


@app.delete("/sources/{source_id}", status_code=204, tags=["Sources"])
async def delete_data_source(source_id: UUID):
    deleted = await delete_source(source_id)
    if not deleted:
        raise HTTPException(404, f"Source {source_id} introuvable")


# ══════════════════════════════════════════════════════════════
# TEST + SYNC
# ══════════════════════════════════════════════════════════════

@app.post("/sources/{source_id}/test", response_model=ConnectionTestResult, tags=["Connections"])
async def test_source_connection(source_id: UUID):
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    result = await test_connection(source_id)
    return ConnectionTestResult(
        source_id=source_id,
        success=result["success"],
        message=result["message"],
        latency_ms=result.get("latency_ms", -1),
        tested_at=result.get("tested_at") or datetime.utcnow(),
    )


async def _run_sync_background(source_id: UUID):
    """Sync complet en background — ne bloque pas la réponse HTTP."""
    logger.info(f"[Sync BG] Démarrage background sync — source {source_id}")
    try:
        result = await sync_metadata(source_id)
        entity_count = result.get("entity_count", 0)
        logger.info(f"[Sync BG] sync_metadata terminé — {entity_count} entités, success={result.get('success')}")
        if result.get("success") and entity_count > 0:
            if entity_count <= 100:
                try:
                    discovery = await discover_relationships(source_id)
                    logger.info(f"[Sync BG] Relations découvertes: {discovery.get('relations_found', 0)}")
                except Exception as e:
                    logger.warning(f"[Sync BG] Relationship discovery failed: {e}")
            else:
                logger.info(f"[Sync BG] {entity_count} entités — auto-discover ignoré (>100), utiliser /discover")
        logger.info(f"[Sync BG] ✅ Terminé — source {source_id}: {entity_count} entités")
    except Exception as e:
        logger.error(f"[Sync BG] ❌ Erreur — source {source_id}: {e}", exc_info=True)


@app.post("/sources/{source_id}/sync", status_code=202, tags=["Connections"])
async def sync_source_metadata(source_id: UUID, background_tasks: BackgroundTasks):
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")

    background_tasks.add_task(_run_sync_background, source_id)
    logger.info(f"[Sync] Tâche background enregistrée pour source {source_id}")

    return {
        "source_id":      str(source_id),
        "success":        True,
        "message":        "Sync lancé en arrière-plan — rafraîchis dans 2-3 minutes",
        "entity_count":   0,
        "field_count":    0,
        "relation_count": 0,
        "duration_ms":    0,
    }


@app.get("/sources/{source_id}/entities", tags=["Metadata"])
async def get_source_entities(
    source_id: UUID,
    page:      int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search:    str = Query(""),
):
    source = await get_source_with_entities(
        source_id, page=page, page_size=page_size, search=search
    )
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    pagination = source.__dict__.get("_pagination", {})
    return {
        "source_id":  source_id,
        "entities":   source.entities,
        "pagination": pagination,
    }


# ══════════════════════════════════════════════════════════════
# RELATIONS
# ══════════════════════════════════════════════════════════════

@app.post("/sources/{source_id}/discover", tags=["Relations"])
async def discover_source_relations(source_id: UUID):
    """Lance la détection automatique des relations pour une source."""
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    return await discover_relationships(source_id)


@app.get("/sources/{source_id}/relations/stats", tags=["Relations"])
async def get_relations_stats(source_id: UUID):
    """Statistiques de validation des relations."""
    return await get_validation_stats(source_id)


@app.get("/sources/{source_id}/relations", tags=["Relations"])
async def get_source_relations(source_id: UUID):
    """Retourne toutes les relations détectées pour une source."""
    try:
        source = await get_source(source_id)
        if not source:
            raise HTTPException(404, f"Source {source_id} introuvable")
        relations = await get_relations_for_source(source_id)
        return {
            "source_id": str(source_id),
            "total":     len(relations),
            "relations": relations,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Relations GET] {source_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Erreur chargement relations: {str(e)}")


@app.post("/relations/{relation_id}/validate", tags=["Relations"])
async def validate_single_relation(relation_id: int, req: ValidateRelationRequest):
    """
    Valide ou rejette une relation détectée.
    Body: { is_confirmed: bool, validated_by?: str, reject_reason?: str }
    """
    ok = await validate_relation(
        relation_id=relation_id,
        confirmed=req.is_confirmed,
        validated_by=req.validated_by or "expert",
        reject_reason=req.reject_reason,
    )
    if not ok:
        raise HTTPException(404, f"Relation {relation_id} introuvable")
    return {
        "success":      True,
        "relation_id":  relation_id,
        "is_confirmed": req.is_confirmed,
        "message":      "✅ Confirmée" if req.is_confirmed else "❌ Rejetée",
    }


@app.patch("/sources/{source_id}/relations/validate-bulk", tags=["Relations"])
async def validate_bulk_relations(source_id: UUID, body: dict):
    """
    Valide plusieurs relations en une fois.
    Body: { validations: [{id, is_confirmed, reject_reason?}], validated_by?: str }
    """
    validations  = body.get("validations", [])
    validated_by = body.get("validated_by", "expert")
    results = []
    for v in validations:
        ok = await validate_relation(
            relation_id=v["id"],
            confirmed=bool(v["is_confirmed"]),
            validated_by=validated_by,
            reject_reason=v.get("reject_reason"),
        )
        results.append({"id": v["id"], "success": ok})
    return {"validated": len(results), "results": results}


@app.get("/sources/{source_id}/join-paths", tags=["Relations"])
async def find_join_paths(
    source_id:  UUID,
    from_table: str = Query(...),
    to_table:   str = Query(...),
    max_depth:  int = Query(3, ge=1, le=5),
):
    """Trouve les chemins de jointure entre deux tables."""
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    paths = await get_join_paths_for_source(source_id, from_table, to_table, max_depth)
    return {"from": from_table, "to": to_table, "paths": paths}


# ══════════════════════════════════════════════════════════════
# IMPORT FK DEPUIS SSMS
# ══════════════════════════════════════════════════════════════

@app.post("/sources/{source_id}/import-fk", tags=["Relations"])
async def import_fk_from_ssms(source_id: UUID, req: ImportFkRequest):
    """
    Importe des FK réelles exportées depuis SSMS.
    Chaque relation est insérée avec confidence=1.0, method=explicit_fk, is_confirmed=TRUE.
    """
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        imported = updated = skipped = 0

        for rel in req.relations:
            src_exists = await conn.fetchval(
                "SELECT 1 FROM source_entities WHERE source_id=$1 AND name=$2",
                source_id, rel.source_entity,
            )
            tgt_exists = await conn.fetchval(
                "SELECT 1 FROM source_entities WHERE source_id=$1 AND name=$2",
                source_id, rel.target_entity,
            )
            if not src_exists or not tgt_exists:
                skipped += 1
                continue

            result = await conn.execute(
                """
                INSERT INTO entity_relations
                    (source_id, source_entity, source_field,
                     target_entity, target_field,
                     relation_type, confidence, detection_method, is_confirmed)
                VALUES ($1, $2, $3, $4, $5, 'many_to_one', 1.0, 'explicit_fk', TRUE)
                ON CONFLICT (source_id, source_entity, source_field, target_entity)
                DO UPDATE SET
                    target_field     = EXCLUDED.target_field,
                    confidence       = 1.0,
                    detection_method = 'explicit_fk',
                    is_confirmed     = TRUE
                """,
                source_id,
                rel.source_entity, rel.source_field,
                rel.target_entity, rel.target_field,
            )
            if "INSERT 0 1" in result:
                imported += 1
            else:
                updated += 1

    return {
        "success":  True,
        "imported": imported,
        "updated":  updated,
        "skipped":  skipped,
        "message": (
            f"{imported} FK importées, {updated} mises à jour, "
            f"{skipped} ignorées (tables inconnues)"
        ),
    }


# ══════════════════════════════════════════════════════════════
# IMPORT VUES SQL
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# IMPORT VUES SQL
# ══════════════════════════════════════════════════════════════

from .view_parser import parse_multiple_views


@app.post("/sources/{source_id}/import-views", tags=["Relations"])
async def import_views_joins(
    source_id: UUID,
    req: ImportViewsRequest = ImportViewsRequest(),
):
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")

    views_list = req.views
    if not views_list:
        try:
            raw_views = await _fetch_views_from_db(source_id)
            # _fetch_views_from_db retourne {"view_name":..., "sql_definition":...}
            # parse_multiple_views attend {"name":..., "definition":...}
            views_list = [
                {"name": v.get("view_name", v.get("name", "")),
                 "definition": v.get("sql_definition", v.get("definition", ""))}
                for v in raw_views
                if v.get("view_name") or v.get("name")
            ]
            logger.info(f"[ImportViews] {len(views_list)} vues récupérées depuis DB")
        except Exception as e:
            raise HTTPException(500, f"Impossible de lire les vues: {str(e)}")

    if not views_list:
        return {
            "success": False,
            "views_analyzed": 0,
            "joins_extracted": 0,
            "imported": 0,
            "updated": 0,
            "skipped": 0,
            "message": "Aucune vue trouvée ou fournie"
        }

    extracted_relations = parse_multiple_views(views_list)
    views_analyzed = len(views_list)
    joins_extracted = len(extracted_relations)

    logger.info(f"[ImportViews auto] {views_analyzed} vues → {joins_extracted} jointures")

    if joins_extracted == 0:
        return {
            "success": True,
            "views_analyzed": views_analyzed,
            "joins_extracted": 0,
            "imported": 0,
            "updated": 0,
            "skipped": 0,
            "message": f"{views_analyzed} vues analysées mais aucune jointure explicite trouvée"
        }

    pool = await get_pg_pool()
    imported = updated = skipped = 0

    async with pool.acquire() as conn:
        async with conn.transaction():
            for rel in extracted_relations:
                src_entity = rel["source_entity"]
                tgt_entity = rel["target_entity"]
                src_field  = rel["source_field"]
                tgt_field  = rel["target_field"]

                src_exists = await conn.fetchval(
                    "SELECT 1 FROM source_entities WHERE source_id=$1 AND LOWER(name)=LOWER($2)",
                    source_id, src_entity
                )
                tgt_exists = await conn.fetchval(
                    "SELECT 1 FROM source_entities WHERE source_id=$1 AND LOWER(name)=LOWER($2)",
                    source_id, tgt_entity
                )
                if not src_exists or not tgt_exists:
                    skipped += 1
                    continue

                existing = await conn.fetchval(
                    """
                    SELECT detection_method FROM entity_relations
                    WHERE source_id=$1 AND LOWER(source_entity)=LOWER($2)
                      AND LOWER(source_field)=LOWER($3) AND LOWER(target_entity)=LOWER($4)
                    """,
                    source_id, src_entity, src_field, tgt_entity
                )
                if existing == "explicit_fk":
                    skipped += 1
                    continue

                result = await conn.execute(
                    """
                    INSERT INTO entity_relations
                        (source_id, source_entity, source_field, target_entity, target_field,
                         relation_type, confidence, detection_method, is_confirmed)
                    VALUES ($1,$2,$3,$4,$5,'many_to_one',1.0,'view_join',TRUE)
                    ON CONFLICT (source_id, source_entity, source_field, target_entity)
                    DO UPDATE SET target_field=EXCLUDED.target_field,
                                  confidence=1.0,
                                  detection_method='view_join',
                                  is_confirmed=TRUE
                    """,
                    source_id, src_entity, src_field, tgt_entity, tgt_field
                )

                if result == "INSERT 0 1":
                    imported += 1
                else:
                    updated += 1

    return {
        "success": True,
        "views_analyzed": views_analyzed,
        "joins_extracted": joins_extracted,
        "imported": imported,
        "updated": updated,
        "skipped": skipped,
        "message": f"{views_analyzed} vues → {joins_extracted} jointures | {imported} nouvelles | {updated} mises à jour | {skipped} ignorées"
    }


@app.get("/sources/{source_id}/import-views/stats", tags=["Relations"])
async def get_view_join_stats(source_id: UUID):
    """
    Retourne le nombre de relations 'view_join' déjà importées pour cette source.
    Utilisé par l'interface pour afficher le compteur.
    """
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, detail=f"Source {source_id} introuvable")

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            """
            SELECT COUNT(*) 
            FROM entity_relations
            WHERE source_id = $1 
              AND detection_method = 'view_join'
            """,
            source_id
        )

    return {
        "source_id": str(source_id),
        "total": int(total or 0)
    }


async def _fetch_views_from_db(source_id: UUID) -> List[dict]:
    """Récupère les définitions des vues depuis SQL Server (sys.views + sys.sql_modules)."""
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        source_row = await conn.fetchrow(
            """
            SELECT connector_type, host, port, database_name, username, options
            FROM data_sources WHERE id = $1
            """,
            source_id,
        )
        if not source_row:
            logger.warning(f"Source {source_id} non trouvée pour fetch vues")
            return []

        source_dict = dict(source_row)
        secret_rows = await conn.fetch(
            "SELECT secret_key, secret_value FROM connection_secrets WHERE source_id = $1",
            source_id,
        )
        for sr in secret_rows:
            if sr["secret_key"] == "password":
                source_dict["password"] = sr["secret_value"]

    ct = (source_dict.get("connector_type") or "").lower()
    # Accepter toutes les variantes MSSQL/SQL Server
    is_mssql = any(x in ct for x in ["mssql", "sqlserver", "sql_server", "sql server", "sql-server"])
    if not is_mssql:
        logger.warning(f"Source {source_id} connector_type='{ct}' → pas MSSQL, fetch vues ignoré")
        return []
    logger.info(f"[FetchViews] Source {source_id} connector_type='{ct}' → OK")

    # Exécution synchrone dans un thread (pyodbc n'est pas async natif)
    def _sync_fetch(sd: dict) -> List[dict]:
        import pyodbc
        import json

        opts = sd.get("options") or {}
        if isinstance(opts, str):
            try:
                opts = json.loads(opts)
            except:
                opts = {}

        host = sd.get("host") or opts.get("host", "localhost")
        port = sd.get("port") or opts.get("port", 1433)
        db   = sd.get("database_name") or ""
        user = sd.get("username") or ""
        pwd  = sd.get("password", "")

        # Choix du driver ODBC disponible
        drivers = pyodbc.drivers()
        driver = next(
            (d for d in drivers if "SQL Server" in d or "ODBC Driver" in d),
            None
        )
        if not driver:
            raise Exception("Aucun driver ODBC SQL Server trouvé")

        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={host},{port};"
            f"DATABASE={db};"
            f"UID={user};PWD={pwd};"
            "TrustServerCertificate=yes;Encrypt=no"
        )

        try:
            with pyodbc.connect(conn_str, timeout=15) as py_conn:
                cursor = py_conn.cursor()
                cursor.execute("""
                    SELECT v.name AS view_name,
                           m.definition AS sql_definition
                    FROM sys.views v
                    JOIN sys.sql_modules m ON v.object_id = m.object_id
                    WHERE v.type = 'V'
                    ORDER BY v.name
                """)
                rows = cursor.fetchall()
                return [{"view_name": r[0], "sql_definition": r[1]} for r in rows]
        except pyodbc.Error as e:
            logger.error(f"Erreur connexion/fetch vues MSSQL: {e}")
            raise

    try:
        return await asyncio.to_thread(_sync_fetch, source_dict)
    except Exception as e:
        logger.error(f"Fetch vues failed: {e}", exc_info=True)
        raise



# ══════════════════════════════════════════════════════════════
# FILE UPLOAD
# ══════════════════════════════════════════════════════════════

@app.post("/upload", tags=["Files"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload un fichier et retourne un aperçu des données.
    Formats supportés : CSV, JSON, TXT, Excel (.xlsx/.xls), Parquet, Avro.
    Pour Excel multi-feuilles, retourne aussi sheets[] pour l'UI.
    """
    allowed = {".csv", ".json", ".txt", ".xlsx", ".xls", ".parquet", ".parq", ".avro"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Type non supporté : {ext}. Acceptés : {', '.join(sorted(allowed))}")

    raw = await file.read()
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(raw)

    try:
        # ── Excel (.xlsx / .xls) ─────────────────────────────────
        if ext in (".xlsx", ".xls"):
            try:
                import openpyxl
                wb     = openpyxl.load_workbook(dest, read_only=True, data_only=True)
                sheets = wb.sheetnames
                # Aperçu de la première feuille
                ws      = wb[sheets[0]]
                rows_raw = list(ws.iter_rows(values_only=True))
                wb.close()
                headers  = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows_raw[0])] if rows_raw else []
                preview  = [
                    {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                    for row in rows_raw[1:6]
                ]
                return {
                    "filename":      file.filename,
                    "uploaded_path": dest,
                    "size":          len(raw),
                    "format":        "excel",
                    "sheets":        sheets,
                    "sheet_count":   len(sheets),
                    "columns":       headers,
                    "column_count":  len(headers),
                    "preview":       preview,
                    "message":       f"Excel uploadé — {len(sheets)} feuille(s) · {len(headers)} colonnes",
                }
            except ImportError:
                raise HTTPException(500, "openpyxl non installé : pip install openpyxl")

        # ── Parquet ──────────────────────────────────────────────
        elif ext in (".parquet", ".parq"):
            try:
                import pyarrow.parquet as pq
                pf       = pq.ParquetFile(dest)
                schema   = pf.schema_arrow
                meta     = pf.metadata
                columns  = [schema.field(i).name for i in range(len(schema))]
                col_types = {schema.field(i).name: str(schema.field(i).type) for i in range(len(schema))}
                # Aperçu des 5 premières lignes
                preview_tbl = pf.read_row_group(0).slice(0, 5).to_pydict()
                preview = [
                    {c: preview_tbl[c][j] for c in columns if j < len(preview_tbl.get(c, []))}
                    for j in range(min(5, meta.row_group(0).num_rows))
                ]
                return {
                    "filename":      file.filename,
                    "uploaded_path": dest,
                    "size":          len(raw),
                    "format":        "parquet",
                    "columns":       columns,
                    "column_count":  len(columns),
                    "col_types":     col_types,
                    "row_count":     meta.num_rows,
                    "row_groups":    meta.num_row_groups,
                    "preview":       preview,
                    "message":       f"Parquet uploadé — {meta.num_rows:,} lignes · {len(columns)} colonnes",
                }
            except ImportError:
                raise HTTPException(500, "pyarrow non installé : pip install pyarrow")

        # ── Avro ─────────────────────────────────────────────────
        elif ext == ".avro":
            try:
                import fastavro
                with open(dest, "rb") as avf:
                    reader  = fastavro.reader(avf)
                    schema  = reader.writer_schema or {}
                    records = []
                    for i, rec in enumerate(reader):
                        if i >= 5: break
                        records.append(rec)
                columns = list(records[0].keys()) if records else []
                if isinstance(schema, dict) and schema.get("fields"):
                    columns = [f["name"] for f in schema["fields"]]
                return {
                    "filename":      file.filename,
                    "uploaded_path": dest,
                    "size":          len(raw),
                    "format":        "avro",
                    "columns":       columns,
                    "column_count":  len(columns),
                    "schema_name":   schema.get("name", "") if isinstance(schema, dict) else "",
                    "preview":       records,
                    "message":       f"Avro uploadé — {len(columns)} champs · schéma intégré",
                }
            except ImportError:
                raise HTTPException(500, "fastavro non installé : pip install fastavro")

        # ── CSV / TXT ────────────────────────────────────────────
        elif ext in (".csv", ".txt"):
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            reader  = csv_module.DictReader(io.StringIO(text))
            rows    = [dict(row) for i, row in enumerate(reader) if i < 5]
            columns = list(rows[0].keys()) if rows else []
            return {
                "filename":      file.filename,
                "uploaded_path": dest,
                "size":          len(raw),
                "format":        "csv",
                "columns":       columns,
                "column_count":  len(columns),
                "preview":       rows,
                "message":       f"CSV uploadé — {len(columns)} colonnes détectées",
            }

        # ── JSON ─────────────────────────────────────────────────
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            data = json_module.loads(text)
            if isinstance(data, list):
                preview = data[:5]
                columns = list(preview[0].keys()) if preview else []
            else:
                preview = [data]
                columns = list(data.keys())
            return {
                "filename":      file.filename,
                "uploaded_path": dest,
                "size":          len(raw),
                "format":        "json",
                "columns":       columns,
                "column_count":  len(columns),
                "preview":       preview,
                "message":       f"JSON uploadé — {len(columns)} colonnes détectées",
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Upload] {file.filename}: {e}", exc_info=True)
        raise HTTPException(500, f"Erreur traitement fichier : {str(e)}")


@app.get("/uploads", tags=["Files"])
async def list_uploads():
    """Liste les fichiers uploadés disponibles."""
    if not os.path.exists(UPLOAD_DIR):
        return {"files": []}
    files = []
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.isfile(fpath):
            files.append({"filename": fname, "path": fpath, "size": os.path.getsize(fpath)})
    return {"files": files}




class ImportSchemaRequest(BaseModel):
    mode:          str             # "sql", "csv", "entities", "odata", "json", "graphql", "wsdl"
    schema_sql:    Optional[str]   = None   # DDL SQL (CREATE TABLE ...)
    schema_csv:    Optional[List[dict]] = None  # [{table_name, column_name, data_type, is_pk, is_fk}]
    entities:      Optional[List[str]]  = None  # noms de tables/entités (SAP, Dynamics, SAGE X3)
    odata_url:     Optional[str]   = None
    schema_json:   Optional[str]   = None
    schema_graphql:Optional[str]   = None
    wsdl_url:      Optional[str]   = None
    wsdl_xml:      Optional[str]   = None


@app.post("/sources/{source_id}/import-schema", tags=["Metadata"])
async def import_schema(source_id: UUID, req: ImportSchemaRequest):
    """
    Importe un schéma décrivant la structure d'une source sans connexion directe.
    Modes : sql (DDL), csv (table exportée), entities (noms SAP/Dynamics/SAGE),
            odata (URL $metadata), json (exemple réponse), graphql (SDL), wsdl.
    """
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")

    entities = []

    # ── Mode SQL DDL ─────────────────────────────────────────────────
    if req.mode == "sql" and req.schema_sql:
        entities = _parse_ddl_to_entities(req.schema_sql)

    # ── Mode CSV ─────────────────────────────────────────────────────
    elif req.mode == "csv" and req.schema_csv:
        from collections import defaultdict
        tables: dict = defaultdict(list)
        for row in req.schema_csv:
            tbl = row.get("table_name") or row.get("TABLE_NAME") or "unknown"
            tables[tbl].append({
                "name":        row.get("column_name") or row.get("COLUMN_NAME") or "",
                "type":        row.get("data_type")   or row.get("DATA_TYPE")   or "string",
                "native_type": row.get("data_type")   or "csv_import",
                "nullable":    str(row.get("is_nullable", "YES")).upper() != "NO",
                "primary_key": str(row.get("is_pk", "0")) in ("1", "YES", "True", "PRI"),
                "foreign_key": str(row.get("is_fk", "0")) in ("1", "YES", "True", "MUL"),
            })
        for tbl_name, fields in tables.items():
            entities.append({"name": tbl_name, "entity_type": "table", "fields": fields})

    # ── Mode Entités (SAP ABAP / Dynamics / SAGE X3) ─────────────────
    elif req.mode == "entities" and req.entities:
        for ent_name in req.entities:
            entities.append({
                "name":        ent_name.strip(),
                "entity_type": "entity",
                "fields":      [{"name": "id", "type": "string", "native_type": "auto",
                                 "nullable": False, "primary_key": True, "foreign_key": False}],
            })

    # ── Mode OData URL ────────────────────────────────────────────────
    elif req.mode == "odata" and req.odata_url:
        try:
            from .connection_service import _parse_odata_metadata
            entities = await _parse_odata_metadata(req.odata_url, {})
        except Exception as e:
            raise HTTPException(400, f"Erreur parsing OData $metadata : {e}")

    # ── Mode JSON exemple ─────────────────────────────────────────────
    elif req.mode == "json" and req.schema_json:
        import json as _json
        try:
            data = _json.loads(req.schema_json)
            from .connection_service import _parse_json
            name = source.name.replace(" ", "_").lower()
            entities = _parse_json(req.schema_json, name)
        except Exception as e:
            raise HTTPException(400, f"Erreur parsing JSON : {e}")

    else:
        raise HTTPException(400, f"Mode '{req.mode}' invalide ou données manquantes")

    if not entities:
        return {"success": False, "message": "Aucune entité extraite — vérifiez le contenu fourni", "entity_count": 0}

    from .repository import save_metadata
    entity_count = await save_metadata(source_id, entities)
    field_count  = sum(len(e.get("fields", [])) for e in entities)

    return {
        "success":      True,
        "entity_count": entity_count,
        "field_count":  field_count,
        "message":      f"{entity_count} entité(s) importée(s) · {field_count} champs",
        "mode":         req.mode,
    }


def _parse_ddl_to_entities(ddl: str) -> List[dict]:
    """Parse du DDL SQL (CREATE TABLE) et extrait les entités et leurs champs."""
    import re
    entities = []

    # Nettoyer les commentaires
    ddl_clean = re.sub(r'--[^\n]*', '', ddl)
    ddl_clean = re.sub(r'/\*.*?\*/', '', ddl_clean, flags=re.DOTALL)

    # Trouver les CREATE TABLE
    pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?'
        r'(?:\[?[\w\.]+\]?\.)?\[?([\w]+)\]?'
        r'\s*\(([^;]+?)\)',
        re.IGNORECASE | re.DOTALL
    )

    TYPE_MAP = {
        "int": "integer", "integer": "integer", "bigint": "integer",
        "smallint": "integer", "tinyint": "integer", "serial": "integer",
        "decimal": "decimal", "numeric": "decimal", "money": "decimal",
        "float": "float", "real": "float", "double": "float",
        "varchar": "string", "nvarchar": "string", "char": "string",
        "nchar": "string", "text": "string", "ntext": "string",
        "date": "date", "datetime": "datetime", "timestamp": "datetime",
        "datetime2": "datetime", "time": "string",
        "bit": "boolean", "bool": "boolean", "boolean": "boolean",
        "uuid": "uuid", "uniqueidentifier": "uuid",
        "json": "json", "jsonb": "json", "xml": "string",
    }

    for match in pattern.finditer(ddl_clean):
        table_name = match.group(1)
        body       = match.group(2)
        fields     = []
        pks        = set()

        # Chercher PRIMARY KEY inline ou en contrainte
        pk_constraint = re.search(r'PRIMARY\s+KEY\s*\(([^)]+)\)', body, re.IGNORECASE)
        if pk_constraint:
            for col in pk_constraint.group(1).split(","):
                pks.add(col.strip().strip("[]`\"'"))

        fk_refs = {}
        for fk_match in re.finditer(
            r'FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+[\[\`]?[\w\.]+[\]\`]?\s*\(([^)]+)\)',
            body, re.IGNORECASE
        ):
            for col in fk_match.group(1).split(","):
                fk_refs[col.strip().strip("[]`\"'")] = True

        # Parser chaque ligne de colonne
        col_pat = re.compile(
            r'^\s*\[?([\w]+)\]?\s+([\w]+)(?:\([^)]*\))?'
            r'(.*?)(?:,|$)',
            re.IGNORECASE
        )
        for line in body.split("\n"):
            line = line.strip().rstrip(",")
            if not line or re.match(r'(PRIMARY|FOREIGN|UNIQUE|CHECK|INDEX|KEY|CONSTRAINT)\b', line, re.IGNORECASE):
                continue
            m = col_pat.match(line)
            if not m:
                continue
            col_name  = m.group(1)
            col_type  = m.group(2).lower()
            col_rest  = m.group(3).upper()
            is_pk     = col_name in pks or "PRIMARY KEY" in col_rest
            is_fk     = col_name in fk_refs
            nullable  = "NOT NULL" not in col_rest and not is_pk

            if is_pk:
                pks.add(col_name)

            fields.append({
                "name":        col_name,
                "type":        TYPE_MAP.get(col_type, "string"),
                "native_type": col_type,
                "nullable":    nullable,
                "primary_key": is_pk,
                "foreign_key": is_fk,
            })

        if fields:
            entities.append({
                "name":        table_name,
                "entity_type": "table",
                "description": f"Importé depuis DDL SQL",
                "fields":      fields,
            })

    return entities

# ══════════════════════════════════════════════════════════════
# PROFILING
# ══════════════════════════════════════════════════════════════

@app.get("/sources/{source_id}/profile", tags=["Profiling"])
async def get_source_profile(source_id: UUID):
    """Résumé de profiling pour toutes les tables déjà profilées."""
    try:
        from .data_profiler import profile_source_summary
        return await profile_source_summary(source_id)
    except Exception as e:
        logger.error(f"[Profile] {source_id}: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/sources/{source_id}/profile/{table_name}", tags=["Profiling"])
async def get_table_profile(source_id: UUID, table_name: str, refresh: bool = False):
    """Profile une table spécifique (avec cache). Passe refresh=true pour forcer le recalcul."""
    try:
        from .data_profiler import get_cached_profile, profile_entity
        if not refresh:
            cached = await get_cached_profile(source_id, table_name)
            if cached:
                return cached
        return await profile_entity(source_id, table_name)
    except Exception as e:
        logger.error(f"[Profile] {source_id}/{table_name}: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/sources/{source_id}/profile/batch", tags=["Profiling"])
async def batch_profile_tables(source_id: UUID, body: dict):
    """Profile plusieurs tables en parallèle (max 10 par appel)."""
    from .data_profiler import profile_entity
    tables = (body.get("tables") or [])[:10]
    if not tables:
        raise HTTPException(400, "tables[] requis")
    results = await asyncio.gather(
        *[profile_entity(source_id, t) for t in tables],
        return_exceptions=True,
    )
    return {
        "results": [
            r if not isinstance(r, Exception) else {"error": str(r), "table": tables[i]}
            for i, r in enumerate(results)
        ]
    }


@app.post("/sources/{source_id}/profile/all", tags=["Profiling"])
async def profile_all_entities(
    source_id:   UUID,
    sample_size: int = Query(1000, ge=100, le=10000, description="Lignes à échantillonner par table"),
    batch_size:  int = Query(5,    ge=1,   le=20,    description="Tables en parallèle par micro-batch"),
):
    """
    Profile TOUTES les entités de la source en séquence par micro-batches.
    - Reprend automatiquement là où il s'est arrêté (skip entités déjà profilées)
    - Générique : fonctionne pour SQL Server, MySQL, PostgreSQL, OData, CSV...
    - Retourne un résumé : total, success, errors
    """
    from .data_profiler import profile_source_all
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    result = await profile_source_all(
        source_id,
        sample_size=sample_size,
        batch_size=batch_size,
    )
    return {"source_id": str(source_id), **result}




@app.post("/sources/{source_id}/circuit-breaker/reset", tags=["Connections"])
async def reset_circuit_breaker(source_id: UUID):
    """
    Réinitialise manuellement le circuit breaker pour une source.
    Utile après avoir résolu un problème de connexion.
    """
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    from .connection_service import CircuitBreaker
    cb = CircuitBreaker.get(str(source_id))
    cb.reset()
    return {
        "source_id": str(source_id),
        "success":   True,
        "message":   "🟢 Circuit Breaker réinitialisé — état : CLOSED",
        "state":     "closed",
    }


@app.get("/sources/{source_id}/circuit-breaker/status", tags=["Connections"])
async def get_circuit_breaker_status(source_id: UUID):
    """
    Retourne l'état actuel du circuit breaker pour une source.
    """
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    from .connection_service import CircuitBreaker
    cb = CircuitBreaker.get(str(source_id), source.options or {})
    import time
    seconds_until_reset = max(0, int(cb.reset_timeout - (time.time() - cb._last_failure_at))) if cb.state.value == "open" else 0
    return {
        "source_id":           str(source_id),
        "state":               cb.state.value,
        "state_label":         cb.state_label,
        "failure_count":       cb._failure_count,
        "failure_threshold":   cb.failure_threshold,
        "reset_timeout_s":     cb.reset_timeout,
        "seconds_until_reset": seconds_until_reset,
        "enabled":             cb.enabled,
    }

# ══════════════════════════════════════════════════════════════
# ML — TRAIN / PREDICT / STATUS
# ══════════════════════════════════════════════════════════════

@app.post("/sources/{source_id}/ml/train", tags=["ML"])
async def ml_train(source_id: UUID):
    """
    Entraîne un modèle ML pour détecter les relations de la source.
    - Utilise les relations existantes (explicit_fk, view_join) comme données d'entraînement
    - Sauvegarde le modèle .pkl dans MODEL_DIR/{source_id}.pkl
    - Retourne les métriques : model_type, F1, ROC-AUC, precision, recall, threshold
    """
    from .ml_detector import train_ml_model
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    try:
        pool = await get_pg_pool()
        result = await train_ml_model(source_id, pool)
        return {"source_id": str(source_id), **result}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"[ML Train] {source_id}: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/sources/{source_id}/ml/predict", tags=["ML"])
async def ml_predict(source_id: UUID):
    """
    Prédit les nouvelles relations ML pour la source.
    - Nécessite un modèle entraîné (POST /ml/train d'abord)
    - Supprime les anciennes prédictions ML et insère les nouvelles
    - Retourne : inserted, errors, threshold, avg_confidence, model_name
    """
    from .ml_detector import predict_ml_relations
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    try:
        pool = await get_pg_pool()
        result = await predict_ml_relations(source_id, pool)
        return {"source_id": str(source_id), **result}
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"[ML Predict] {source_id}: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/sources/{source_id}/ml/status", tags=["ML"])
async def ml_status(source_id: UUID):
    """
    Retourne le statut du modèle ML pour la source :
    - available: bool
    - model_name, trained_at, threshold, metrics (F1, AUC...)
    """
    from .ml_detector import get_ml_status
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    return {"source_id": str(source_id), **get_ml_status(source_id)}





# ══════════════════════════════════════════════════════════════
# FONCTION POUR FETCH AUTO DES VUES DEPUIS MSSQL
# ══════════════════════════════════════════════════════════════

async def _fetch_views_from_db(source_id: UUID) -> List[dict]:
    """Récupère les définitions des vues depuis SQL Server via pyodbc."""
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        source_row = await conn.fetchrow(
            """
            SELECT connector_type, host, port, database_name, username, options
            FROM data_sources WHERE id = $1
            """,
            source_id,
        )
        if not source_row:
            logger.warning(f"Source {source_id} introuvable pour fetch vues")
            return []

        source_dict = dict(source_row)
        secret_rows = await conn.fetch(
            "SELECT secret_key, secret_value FROM connection_secrets WHERE source_id = $1",
            source_id,
        )
        for sr in secret_rows:
            if sr["secret_key"] == "password":
                source_dict["password"] = sr["secret_value"]

    ct = (source_dict.get("connector_type") or "").lower()
    if not any(x in ct for x in ["mssql", "sqlserver", "sql server"]):
        logger.warning(f"Source {source_id} n'est pas MSSQL → fetch ignoré")
        return []

    # Exécution synchrone dans thread (pyodbc n'est pas async)
    def _sync_fetch(sd: dict) -> List[dict]:
        import pyodbc
        import json

        opts = sd.get("options") or {}
        if isinstance(opts, str):
            try:
                opts = json.loads(opts)
            except:
                pass

        host = sd.get("host") or opts.get("host", "localhost")
        port = sd.get("port") or opts.get("port", 1433)
        db   = sd.get("database_name") or ""
        user = sd.get("username") or ""
        pwd  = sd.get("password", "")

        drivers = pyodbc.drivers()
        driver = next(
            (d for d in drivers if "SQL Server" in d or "ODBC Driver" in d),
            None
        )
        if not driver:
            raise Exception("Aucun driver ODBC pour SQL Server trouvé")

        conn_str = (
            f"DRIVER={{{driver}}};SERVER={host},{port};DATABASE={db};"
            f"UID={user};PWD={pwd};TrustServerCertificate=yes;Encrypt=no"
        )

        try:
            with pyodbc.connect(conn_str, timeout=20) as py_conn:
                cursor = py_conn.cursor()
                cursor.execute("""
                    SELECT v.name AS view_name,
                           m.definition AS sql_definition
                    FROM sys.views v
                    JOIN sys.sql_modules m ON v.object_id = m.object_id
                    WHERE v.type = 'V'
                    ORDER BY v.name
                """)
                rows = cursor.fetchall()
                return [{"view_name": r.view_name, "sql_definition": r.sql_definition} for r in rows]
        except pyodbc.Error as e:
            logger.error(f"Erreur pyodbc fetch vues: {e}")
            raise

    try:
        return await asyncio.to_thread(_sync_fetch, source_dict)
    except Exception as e:
        logger.error(f"Fetch vues failed pour {source_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Erreur lors du fetch des vues: {str(e)}")
# ══════════════════════════════════════════════════════════════
# ENTRYPOINT (dev uniquement — en prod: uvicorn api.main:app)
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)