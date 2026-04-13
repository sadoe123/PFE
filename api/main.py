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
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .connection_service import sync_metadata, test_connection
from .database import close_connections, get_pg_pool, get_redis, init_schema
from .relationship_discovery import (
    discover_relationships,
    get_join_paths_for_source,
    get_relations_for_source,
    get_false_positive_pairs,
    remove_false_positive,
    get_validation_stats,
    get_alternative_suggestions,
    record_expert_feedback,
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
from .cdc_engine       import CDCEngine
from .schema_versioner import SchemaVersioner

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

def _safe_json(obj):
    """Nettoie un dict/list pour JSON — remplace NaN/Infinity par None."""
    import math
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)
    return obj

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

        # ── Migration : ajout colonne dimension_hierarchy si absente ─────────
        try:
            pool = await get_pg_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    ALTER TABLE entity_fields
                    ADD COLUMN IF NOT EXISTS dimension_hierarchy JSONB DEFAULT NULL
                """)
            logger.info("✅ Migration dimension_hierarchy OK")
        except Exception as e:
            logger.warning(f"Migration dimension_hierarchy (non-bloquant): {e}")
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

app = FastAPI(redirect_slashes=False, 
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


class ValidateRelationRequest(BaseModel):
    is_confirmed:  bool
    validated_by:  Optional[str] = "expert"
    reject_reason: Optional[str] = None


class FeedbackRequest(BaseModel):
    """PATCH P6 — Feedback expert sur une relation détectée."""
    source_entity:  str
    source_field:   str
    target_entity:  str
    target_field:   str
    feedback:       str            # 'confirmed' | 'rejected' | 'alternative'
    user_id:        Optional[str]  = None
    comment:        Optional[str]  = None
    alternative_target_entity: Optional[str] = None
    alternative_target_field:  Optional[str] = None


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
            # ── Étape 2 : Découverte des relations ───────────────────────────
            try:
                discovery = await discover_relationships(source_id)
                logger.info(f"[Sync BG] Relations découvertes: {discovery.get('relations_found', 0)}")
            except Exception as e:
                logger.warning(f"[Sync BG] Relationship discovery failed (non-bloquant): {e}")

            # ── Étape 3 : Enrichissement sémantique BERT ─────────────────────
            try:
                from .semantic_enricher import enrich_source as _enrich_source
                source = await get_source(source_id)
                if source:
                    enrich_result = await _enrich_source(
                        source_id,
                        source.name,
                        source.connector_type.value,
                    )
                    logger.info(
                        f"[Sync BG] Enrichissement sémantique: "
                        f"{enrich_result.get('enriched', 0)} entités | "
                        f"MeiliSearch: {enrich_result.get('meili_indexed', 0)} | "
                        f"Domaines: {enrich_result.get('domain_stats', {})}"
                    )
            except Exception as e:
                logger.warning(f"[Sync BG] Enrichissement sémantique failed (non-bloquant): {e}")

        logger.info(f"[Sync BG] ✅ Pipeline complet — source {source_id}: {entity_count} entités")
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


@app.get("/sources/{source_id}/db-objects", tags=["Metadata"])
async def get_db_objects(
    source_id: UUID,
    object_type: Optional[str] = Query(None, description="filter: stored_procedure, function, trigger, sequence, index"),
):
    """
    Retourne les objets DB avancés d'une source :
    procédures stockées, fonctions, triggers, séquences, et index.
    Disponible pour MSSQL et PostgreSQL après sync.
    """
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, name, entity_type, description, metadata
            FROM source_entities
            WHERE source_id = $1
              AND entity_type IN (
                  'stored_procedure','function','trigger','sequence',
                  'aggregate','window'
              )
            ORDER BY entity_type, name
        """, source_id)

        # Index : stockés dans metadata de chaque table
        idx_rows = await conn.fetch("""
            SELECT name, indexes
            FROM source_entities
            WHERE source_id = $1
              AND entity_type IN ('table','view','materialized_view')
              AND indexes IS NOT NULL
              AND indexes != '[]'
        """, source_id)

    objects = []
    for r in rows:
        etype = r["entity_type"]
        if object_type and etype != object_type:
            continue
        meta = {}
        try:
            meta = json_module.loads(r["metadata"] or "{}")
        except Exception:
            pass
        objects.append({
            "id":          r["id"],
            "name":        r["name"],
            "type":        etype,
            "description": r["description"],
            "metadata":    meta,
        })

    indexes = []
    if not object_type or object_type == "index":
        for r in idx_rows:
            try:
                idx_list = json_module.loads(r["indexes"] or "[]")
                for idx in idx_list:
                    indexes.append({
                        "table":   r["name"],
                        "name":    idx.get("name"),
                        "type":    idx.get("type"),
                        "unique":  idx.get("unique", False),
                        "pk":      idx.get("pk", False),
                        "columns": idx.get("columns", ""),
                    })
            except Exception:
                pass

    counts = {}
    for o in objects:
        counts[o["type"]] = counts.get(o["type"], 0) + 1
    if indexes:
        counts["index"] = len(indexes)

    return {
        "source_id":  str(source_id),
        "counts":     counts,
        "objects":    objects,
        "indexes":    indexes,
        "total":      len(objects) + len(indexes),
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
    Enregistre aussi dans relation_feedback pour le blacklisting des faux positifs.
    """
    ok = await validate_relation(
        relation_id=relation_id,
        confirmed=req.is_confirmed,
        validated_by=req.validated_by or "expert",
        reject_reason=req.reject_reason,
    )
    if not ok:
        raise HTTPException(404, f"Relation {relation_id} introuvable")

    # Enregistrer dans relation_feedback pour tracking faux positifs récurrents
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            rel = await conn.fetchrow(
                "SELECT source_id, source_entity, source_field, target_entity, target_field "
                "FROM entity_relations WHERE id=$1", relation_id)
            if rel:
                feedback_type = "confirmed" if req.is_confirmed else "rejected"
                await conn.execute("""
                    INSERT INTO relation_feedback
                        (source_id, relation_id, source_entity, source_field,
                         target_entity, target_field, feedback, user_id, comment)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                """, rel["source_id"], relation_id,
                    rel["source_entity"], rel["source_field"],
                    rel["target_entity"], rel["target_field"],
                    feedback_type, req.validated_by or "expert",
                    req.reject_reason)
    except Exception as _fe:
        logger.warning(f"[validate] feedback insert warning: {_fe}")

    return {
        "success":      True,
        "relation_id":  relation_id,
        "is_confirmed": req.is_confirmed,
        "message":      "✅ Confirmée" if req.is_confirmed else "❌ Rejetée",
    }


@app.patch("/sources/{source_id}/relations/validate-bulk", tags=["Relations"])
@app.post("/sources/{source_id}/relations/validate-bulk", tags=["Relations"])
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
                import fastavro  # type: ignore[import]
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
    """
    Profile une table spécifique (avec cache TTL intelligent).
    TTL : référentiel=24h | transactionnel=1h | inconnu=6h
    Passe refresh=true pour forcer le recalcul immédiat.
    """
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


@app.get("/sources/{source_id}/profile/cache/stats", tags=["Profiling"])
async def get_profile_cache_stats(source_id: UUID):
    """
    Retourne les statistiques du cache de profiling pour une source.
    Indique quelles tables sont fraîches, expirées ou jamais profilées.
    """
    from .database import get_pg_pool
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT entity_name,
                   profiled_at,
                   profile_data->>'table_class' AS table_class,
                   EXTRACT(EPOCH FROM (NOW() - profiled_at))/3600 AS age_hours
            FROM entity_profiles
            WHERE source_id = $1
            ORDER BY profiled_at DESC
        """, source_id)

    TTL = {"reference": 24, "transactional": 1}
    stats = {"fresh": 0, "stale": 0, "total": len(rows), "tables": []}

    for row in rows:
        tc      = row["table_class"] or "unknown"
        ttl     = TTL.get(tc, 6)
        age     = round(float(row["age_hours"]), 1)
        is_fresh = age <= ttl
        if is_fresh:
            stats["fresh"] += 1
        else:
            stats["stale"] += 1
        stats["tables"].append({
            "entity":      row["entity_name"],
            "table_class": tc,
            "age_hours":   age,
            "ttl_hours":   ttl,
            "fresh":       is_fresh,
            "profiled_at": row["profiled_at"].isoformat(),
        })

    stats["fresh_pct"] = round(stats["fresh"] / stats["total"] * 100, 1) if stats["total"] else 0
    return stats


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
    Nécessite ml_detector.py OU utiliser ml_relation_detector_v6.ipynb directement.
    """
    try:
        from .ml_detector import train_ml_model  # type: ignore[import]
    except ImportError:
        raise HTTPException(503, (
            "ml_detector.py non disponible. "
            "Utiliser ml_relation_detector_v6.ipynb pour générer best_model_xgboost.pkl, "
            "puis POST /discover utilisera le modèle automatiquement."
        ))
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    try:
        pool = await get_pg_pool()
        result = await train_ml_model(source_id, pool)
        return _safe_json({"source_id": str(source_id), **result})
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        msg = str(e)
        logger.error(f"[ML Train] {source_id}: {msg}", exc_info=True)
        # Message lisible pour les cas fréquents
        if "insufficient_data" in msg or "assez de relations" in msg.lower():
            raise HTTPException(400, {
                "status": "insufficient_data",
                "message": "Pas assez de relations pour entraîner le modèle. Lancez d'abord ↻ Relancer pour détecter les relations explicites."
            })
        raise HTTPException(500, msg)


@app.post("/sources/{source_id}/ml/predict", tags=["ML"])
async def ml_predict(source_id: UUID):
    """
    Prédit les nouvelles relations ML. Alternative : POST /discover charge le .pkl auto.
    """
    try:
        from .ml_detector import predict_ml_relations  # type: ignore[import]
    except ImportError:
        raise HTTPException(503, (
            "ml_detector.py non disponible. "
            "Utiliser POST /discover qui charge best_model_xgboost.pkl automatiquement."
        ))
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    try:
        pool = await get_pg_pool()
        result = await predict_ml_relations(source_id, pool)
        return _safe_json({"source_id": str(source_id), **result})
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
    return _safe_json({"source_id": str(source_id), **get_ml_status(source_id)})


@app.get("/sources/{source_id}/relations/false-positives", tags=["Relations"])
async def list_false_positives(
    source_id: UUID,
    min_rejections: int = Query(3, ge=1, le=20)
):
    """Retourne les paires rejetées >= min_rejections fois (blacklistées)."""
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    pairs = await get_false_positive_pairs(source_id, min_rejections)
    return {"source_id": str(source_id), "threshold": min_rejections, "total": len(pairs), "pairs": pairs}


@app.delete("/sources/{source_id}/relations/false-positives", tags=["Relations"])
async def rehabilitate_pair(
    source_id:     UUID,
    source_entity: str = Query(...),
    source_field:  str = Query(...),
    target_entity: str = Query(...),
):
    """Réhabilite une paire blacklistée."""
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    return await remove_false_positive(source_id, source_entity, source_field, target_entity)


@app.post("/sources/{source_id}/ml/retrain", tags=["ML"])
async def ml_retrain_feedback(source_id: UUID):
    """
    PATCH P12 — Re-entraîne le modèle ML avec les feedbacks experts accumulés.
    Déclenché automatiquement tous les 20 feedbacks, ou manuellement ici.
    """
    try:
        from .ml_detector import retrain_with_feedback  # type: ignore
    except ImportError:
        raise HTTPException(503, "ml_detector.py non disponible.")
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    try:
        pool   = await get_pg_pool()
        result = await retrain_with_feedback(source_id, pool)
        return {"source_id": str(source_id), **result}
    except Exception as e:
        logger.error(f"[ML Retrain] {source_id}: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════
# PATCH P5/P6 — Alternatives + Feedback expert
# ══════════════════════════════════════════════════════════════════════

@app.get("/sources/{source_id}/relations/alternatives", tags=["Relations"])
async def get_relation_alternatives(
    source_id:      UUID,
    source_entity:  str = Query(...),
    source_field:   str = Query(...),
    current_target: str = Query(...),
    top_k:          int = Query(3, ge=1, le=10),
):
    """
    PATCH P5 — Retourne les K meilleures relations alternatives pour un champ donné.
    Utilisé par l'UI de validation pour proposer des corrections.
    """
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    return await get_alternative_suggestions(
        source_id, source_entity, source_field, current_target, top_k
    )


@app.post("/sources/{source_id}/relations/{relation_id}/feedback", tags=["Relations"])
async def submit_relation_feedback(
    source_id:   UUID,
    relation_id: UUID,
    req:         FeedbackRequest,
):
    """
    PATCH P6 — Enregistre le feedback expert (confirmé/rejeté/alternative).
    Si 20 feedbacks accumulés → retourne should_retrain=True.
    """
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    result = await record_expert_feedback(
        source_id=source_id,
        relation_id=relation_id,
        source_entity=req.source_entity,
        source_field=req.source_field,
        target_entity=req.target_entity,
        target_field=req.target_field,
        feedback=req.feedback,
        user_id=req.user_id,
        comment=req.comment,
        alternative_target_entity=req.alternative_target_entity,
        alternative_target_field=req.alternative_target_field,
    )
    # Auto-trigger re-training si seuil atteint
    if result.get("should_retrain"):
        try:
            from .ml_detector import retrain_with_feedback  # type: ignore
            pool = await get_pg_pool()
            asyncio.create_task(retrain_with_feedback(source_id, pool))
            logger.info(f"[Feedback] Re-training ML déclenché en background — {source_id}")
        except Exception as _rt:
            logger.warning(f"[Feedback] Auto-retrain failed (non-bloquant): {_rt}")
    return result


# ══════════════════════════════════════════════════════════════════════
# PATCH P7/P8 — Histogrammes DB + Dépendances objets
# ══════════════════════════════════════════════════════════════════════

@app.get("/sources/{source_id}/entities/{entity_name}/columns/{column_name}/histogram",
         tags=["Profiling"])
async def get_column_histogram_endpoint(
    source_id:   UUID,
    entity_name: str,
    column_name: str,
):
    """
    PATCH P7 — Retourne l'histogramme d'une colonne depuis les stats moteur DB.
    PostgreSQL : pg_stats | MSSQL : sys.dm_db_stats_histogram
    """
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    from .data_profiler import get_column_histogram
    return await get_column_histogram(source_id, entity_name, column_name)


@app.get("/sources/{source_id}/dependencies", tags=["Metadata"])
async def get_source_dependencies(source_id: UUID):
    """
    PATCH P8 — Retourne les dépendances entre objets DB (vues→tables, procédures, triggers).
    """
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    from .data_profiler import get_object_dependencies
    return await get_object_dependencies(source_id)


@app.get("/sources/{source_id}/excel-metadata/{entity_name}", tags=["Metadata"])
async def get_excel_metadata_endpoint(source_id: UUID, entity_name: str):
    """
    PATCH P9 — Retourne les métadonnées avancées d'un fichier Excel
    (formules, auteur, headers/footers, plages nommées).
    """
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")
    # Récupérer le path du fichier depuis les options de la source
    opts      = source.options or {}
    file_path = opts.get("file_path") or source.base_url or ""
    if not file_path:
        raise HTTPException(400, "Chemin de fichier non configuré pour cette source")
    if not file_path.lower().endswith((".xlsx", ".xlsm", ".xls")):
        raise HTTPException(400, "Cette source n'est pas un fichier Excel")
    from .data_profiler import extract_excel_metadata
    return extract_excel_metadata(file_path)


# ══════════════════════════════════════════════════════════════════════
# PATCH CROSS-SOURCES — Détection + Path-finding multi-sources
# ══════════════════════════════════════════════════════════════════════

@app.post("/cross-source/detect", tags=["Cross-Source"])
async def detect_cross_source(body: dict):
    """
    Phase 3 — Détecte les mappings entre colonnes de deux sources différentes.
    Body: { source_id_a: UUID, source_id_b: UUID, min_overlap: float = 0.30 }
    """
    try:
        from .cross_source_mapper import (
            detect_cross_source_mappings, save_cross_source_mappings
        )
    except ImportError:
        raise HTTPException(503, "cross_source_mapper.py non disponible")

    sid_a = UUID(body.get("source_id_a", ""))
    sid_b = UUID(body.get("source_id_b", ""))
    min_overlap = float(body.get("min_overlap", 0.30))

    src_a = await get_source(sid_a)
    src_b = await get_source(sid_b)
    if not src_a:
        raise HTTPException(404, f"Source A {sid_a} introuvable")
    if not src_b:
        raise HTTPException(404, f"Source B {sid_b} introuvable")

    mappings = await detect_cross_source_mappings(sid_a, sid_b, min_overlap)
    saved    = await save_cross_source_mappings(mappings)
    return {
        "source_id_a": str(sid_a),
        "source_id_b": str(sid_b),
        "mappings_found": len(mappings),
        "mappings_saved": saved,
        "top_mappings":   mappings[:10],
    }


@app.post("/cross-source/path", tags=["Cross-Source"])
async def find_cross_source_path(body: dict):
    """
    Phase 3 — Trouve les chemins de jointure entre entités de sources différentes.
    Body: {
      source_ids: [UUID, ...],
      from: { source_id: UUID, entity: str },
      to:   { source_id: UUID, entity: str },
      max_depth: int = 4
    }
    """
    try:
        from .cross_source_mapper import build_multi_source_graph
    except ImportError:
        raise HTTPException(503, "cross_source_mapper.py non disponible")

    source_ids = [UUID(s) for s in body.get("source_ids", [])]
    from_info  = body.get("from", {})
    to_info    = body.get("to",   {})
    max_depth  = int(body.get("max_depth", 4))

    if not source_ids or not from_info or not to_info:
        raise HTTPException(400, "source_ids, from et to requis")

    graph = await build_multi_source_graph(source_ids)
    paths = graph.find_paths(
        str(from_info.get("source_id", "")), from_info.get("entity", ""),
        str(to_info.get("source_id",   "")), to_info.get("entity",   ""),
        max_depth,
    )
    return {
        "from":       from_info,
        "to":         to_info,
        "paths_found": len(paths),
        "paths":       paths,
        "graph_stats": {"nodes": graph.node_count, "edges": graph.edge_count},
    }


@app.get("/cross-source/mappings", tags=["Cross-Source"])
async def list_cross_source_mappings(
    source_id_a:    UUID = Query(...),
    source_id_b:    Optional[UUID] = Query(None),
    min_confidence: float = Query(0.30, ge=0.0, le=1.0),
):
    """Phase 3 — Liste les mappings cross-sources existants."""
    try:
        from .cross_source_mapper import get_cross_source_mappings
    except ImportError:
        raise HTTPException(503, "cross_source_mapper.py non disponible")
    return await get_cross_source_mappings(source_id_a, source_id_b, min_confidence)


@app.put("/cross-source/mappings/{mapping_id}/validate", tags=["Cross-Source"])
async def validate_cross_source_mapping(mapping_id: UUID, body: dict):
    """
    Valide ou rejette un mapping cross-source.
    Body: { validated: bool, validated_by: str }
    """
    validated    = bool(body.get("validated", True))
    validated_by = body.get("validated_by", "user")

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE cross_source_mappings
            SET is_validated = $1,
                validated_by = $2
            WHERE id = $3
        """, validated, validated_by, mapping_id)

    if result == "UPDATE 0":
        raise HTTPException(404, f"Mapping {mapping_id} introuvable")

    return {
        "success":      True,
        "mapping_id":   str(mapping_id),
        "validated":    validated,
        "validated_by": validated_by,
        "message":      f"Mapping {'validé' if validated else 'rejeté'} par {validated_by}",
    }


# ══════════════════════════════════════════════════════════════
# CHAT IA ENDPOINT §2.4
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    source_id: str
    question: str
    history: Optional[List[dict]] = []

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """Endpoint Chat IA — fallback local en attendant Ollama (Phase 2)."""
    try:
        answer = await _build_chat_answer(req.source_id, req.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"answer": f"❌ Erreur: {str(e)}"}


# ══════════════════════════════════════════════════════════════
# CONVERSATIONS CHAT IA
# ══════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str  # "user" ou "bot"
    content: str

class ConversationIn(BaseModel):
    id: str
    source_id: str
    title: str
    messages: List[Message]
    created_at: str

@app.post("/conversations")
async def save_conversation(conv: ConversationIn):
    """Sauvegarder ou mettre à jour une conversation (PAS d'ID UUID requis)"""
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            # Vérifier si la conversation existe déjà (par son mongo-like ID string)
            existing_id = await conn.fetchval(
                "SELECT id FROM conversations WHERE id::text = $1",
                conv.id
            )
            
            messages_json = json_module.dumps([m.model_dump() for m in conv.messages])
            
            if existing_id:
                # Mise à jour
                await conn.execute(
                    """UPDATE conversations 
                       SET title=$2, messages=$3, updated_at=NOW()
                       WHERE id::text = $1""",
                    conv.id, conv.title, messages_json
                )
                return {"status": "ok", "id": str(existing_id)}
            else:
                # Insertion - générer UUID si nécessaire
                try:
                    conv_uuid = UUID(conv.id)
                except ValueError:
                    conv_uuid = uuid4()
                
                await conn.execute(
                    """INSERT INTO conversations (id, source_id, title, messages)
                       VALUES ($1::uuid, $2::uuid, $3, $4)
                       ON CONFLICT (id) DO UPDATE
                       SET title=$3, messages=$4, updated_at=NOW()""",
                    conv_uuid, 
                    conv.source_id, 
                    conv.title, 
                    messages_json
                )
                
                return {"status": "ok", "id": str(conv_uuid)}
    except Exception as e:
        logger.error(f"Conversation save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations")
async def get_conversations():
    """Récupérer toutes les conversations"""
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, source_id, title, created_at FROM conversations ORDER BY created_at DESC LIMIT 50"
            )
            
            conversations = []
            for row in rows:
                conversations.append({
                    "id": str(row['id']),
                    "source_id": str(row['source_id']),
                    "title": row['title'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None
                })
            
            return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        return {"conversations": []}

@app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Récupérer une conversation spécifique"""
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            # Essayer de parser en UUID, sinon utiliser comme string
            try:
                uuid_val = UUID(conv_id)
            except ValueError:
                uuid_val = None
            
            if uuid_val:
                row = await conn.fetchrow(
                    "SELECT id, source_id, title, messages FROM conversations WHERE id = $1 OR id::text = $2",
                    uuid_val, conv_id
                )
            else:
                row = await conn.fetchrow(
                    "SELECT id, source_id, title, messages FROM conversations WHERE id::text = $1",
                    conv_id
                )
            
            if not row:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            messages = json_module.loads(row['messages']) if row['messages'] else []
            
            return {
                "id": str(row['id']),
                "source_id": str(row['source_id']),
                "title": row['title'],
                "messages": messages
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# CHAT & CHAT STREAMING ENDPOINTS
# ══════════════════════════════════════════════════════════════

async def _build_chat_answer(source_id_str: str, question: str) -> str:
    """Génère une réponse contextuelle basée sur les métadonnées de la source."""
    try:
        source_id = UUID(source_id_str)
    except (ValueError, AttributeError):
        return "❌ ID source invalide."

    source = await get_source(source_id)
    if not source:
        return "❌ Source introuvable."

    q = question.lower()
    name = source.name
    entity_count = source.entity_count or 0

    try:
        detail = await get_source_with_entities(source_id, page=1, page_size=20)
        entities = detail.entities if detail else []
    except Exception:
        entities = []

    try:
        relations = await get_relations_for_source(source_id)
        relations = relations[:5] if relations else []
    except Exception:
        relations = []

    if any(k in q for k in ["combien", "nombre", "count", "total"]) and any(k in q for k in ["table", "entit", "entity"]):
        sample = ", ".join(f"`{e.name}`" for e in entities[:6])
        extra = f"\n\nExemples : {sample}{'…' if entity_count > 6 else ''}" if sample else ""
        return f"📊 La source **{name}** contient **{entity_count} entités** indexées.{extra}"

    elif any(k in q for k in ["liste", "list", "table", "entit", "top"]):
        if not entities:
            return f"⚠️ Aucune entité synchronisée pour **{name}**.\n\nCliquez sur **↻ Sync** depuis la page d'accueil."
        rows = "\n".join(f"{i+1}. `{e.name}` — {e.field_count or 0} champs" for i, e in enumerate(entities[:10]))
        more = f"\n\n*… et {entity_count - 10} autres entités.*" if entity_count > 10 else ""
        return f"📋 Entités de **{name}** :\n\n{rows}{more}"

    elif any(k in q for k in ["relation", "fk", "jointure", "join", "lien"]):
        if not relations:
            return f"🔗 Aucune relation détectée pour **{name}**.\n\n💡 Lancez la découverte depuis **Relations → ↻ Relancer**."
        rows = "\n".join(
            f"- `{r.get('source_entity','?')}.{r.get('source_field','?')}` → `{r.get('target_entity','?')}.{r.get('target_field','?')}` *({r.get('detection_method','?')}, {round((r.get('confidence') or 1)*100)}%)*"
            for r in relations
        )
        return f"🔗 Relations détectées pour **{name}** :\n\n{rows}\n\n💡 Explorez toutes les relations dans l'onglet **Relations** de l'accueil."

    elif any(k in q for k in ["sql", "requête", "query", "genere", "génère", "select"]):
        if relations:
            r = relations[0]
            return f"🔧 Exemple SQL pour **{name}** :\n\n```sql\nSELECT a.*, b.*\nFROM {r.get('source_entity','TableA')} a\nJOIN {r.get('target_entity','TableB')} b\n  ON a.{r.get('source_field','id')} = b.{r.get('target_field','id')}\nLIMIT 100;\n```"
        elif entities:
            return f"🔧 Exemple SQL :\n\n```sql\nSELECT *\nFROM {entities[0].name}\nLIMIT 100;\n```"
        return "⚠️ Synchronisez la source d'abord."

    elif any(k in q for k in ["structure", "info", "résumé", "resume", "détail", "detail"]):
        host_info = f"`{source.host}:{source.port}`" if source.host else "—"
        return (
            f"📊 **{name}**\n\n"
            f"- **Type** : {source.connector_type}\n"
            f"- **Hôte** : {host_info}\n"
            f"- **Base** : `{source.database_name or source.base_url or '—'}`\n"
            f"- **Entités** : {entity_count}\n"
            f"- **Relations** : {len(relations)}\n"
            f"- **Statut** : {source.status}"
        )

    elif any(k in q for k in ["champ", "field", "colonne", "column"]):
        if entities:
            e = entities[0]
            fields = e.fields[:8] if e.fields else []
            rows = "\n".join(f"- `{f.name}` ({f.data_type}{'  🔑' if f.is_primary_key else ''}{'  🔗' if f.is_foreign_key else ''})" for f in fields)
            return f"📋 Champs de `{e.name}` :\n\n{rows}"
        return "⚠️ Aucune entité chargée. Lancez une synchronisation d'abord."

    else:
        return (
            f"🤖 **OnePilot** — Mode local *(LLM non configuré)*\n\n"
            f"Source **{name}** : **{entity_count} entités**.\n\n"
            f"**Je peux répondre sur :**\n"
            f"- Combien de tables ?\n- Liste des entités\n- Relations détectées\n"
            f"- Générer du SQL\n- Résumé de la structure"
        )




@app.delete("/conversations/{conv_id}", status_code=204)
async def delete_conversation(conv_id: str):
    """Supprimer une conversation"""
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            try:
                uuid_val = UUID(conv_id)
            except ValueError:
                uuid_val = None
            if uuid_val:
                result = await conn.execute(
                    "DELETE FROM conversations WHERE id = $1 OR id::text = $2",
                    uuid_val, conv_id
                )
            else:
                result = await conn.execute(
                    "DELETE FROM conversations WHERE id::text = $1",
                    conv_id
                )
            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Conversation not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream SSE mot par mot avec la vraie réponse contextuelle."""
    from fastapi.responses import StreamingResponse
    try:
        source_id = req.source_id
        question = req.question
        if not source_id or not question:
            raise HTTPException(status_code=400, detail="Missing source_id or question")

        answer = await _build_chat_answer(source_id, question)

        async def generate():
            words = answer.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield "data: " + json_module.dumps({"token": token}) + "\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════════
# TEXT-TO-SPEECH & STT ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.post("/tts")
async def text_to_speech(payload: dict):
    """Convert text to speech"""
    try:
        text = payload.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Missing text")
        
        # Placeholder TTS — retourne 204 No Content si pas de moteur configuré
        from fastapi.responses import Response
        return Response(
            content=b"",
            media_type="audio/mpeg",
            status_code=204
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """Convert speech to text"""
    try:
        # For now, return a placeholder transcription
        return {"text": "Transcribed text placeholder", "confidence": 0.95}
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════════
# SEMANTIC ENRICHMENT — §2.2.3
# ══════════════════════════════════════════════════════════════

try:
    from .semantic_enricher import enrich_source, semantic_search as _semantic_search
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False
    logger.warning("semantic_enricher non disponible")


@app.post("/sources/{source_id}/enrich", tags=["Semantic"])
async def enrich_source_endpoint(
    source_id: UUID,
    background_tasks: BackgroundTasks,
):
    """
    Lance l'enrichissement sémantique d'une source :
    - Classification domaine métier (Finance, RH, Ventes...)
    - Extraction concept métier (Customer, Order, Invoice...)
    - Détection dimensions analytiques (temps, géo, produit)
    - Indexation MeiliSearch (full-text search)
    - Calcul embeddings TF-IDF + pgvector (recherche sémantique)
    """
    if not _SEMANTIC_AVAILABLE:
        raise HTTPException(503, "Module semantic_enricher non disponible")

    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")

    async def _run_enrich():
        try:
            result = await enrich_source(
                source_id=source_id,
                source_name=source.name,
                source_type=source.connector_type.value,
            )
            logger.info(f"[Enrich] ✅ {source.name}: {result.get('enriched', 0)} entités enrichies")
        except Exception as e:
            logger.error(f"[Enrich] ❌ {source.name}: {e}", exc_info=True)

    background_tasks.add_task(_run_enrich)
    return {
        "source_id": str(source_id),
        "source_name": source.name,
        "message": "Enrichissement sémantique lancé en arrière-plan",
        "status": "running",
    }


@app.get("/sources/{source_id}/enrich/status", tags=["Semantic"])
async def get_enrich_status(source_id: UUID):
    """Retourne le statut de l'enrichissement sémantique d'une source."""
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT
                COUNT(*)                                            AS total,
                COUNT(*) FILTER (WHERE business_domain IS NOT NULL) AS enriched,
                COUNT(DISTINCT business_domain)                     AS domains_count,
                COUNT(DISTINCT business_concept)                    AS concepts_count,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL)       AS with_embedding
            FROM source_entities
            WHERE source_id = $1
              AND entity_type IN ('table','view','materialized_view',
                                  'odata_entity','rest_resource','excel_sheet')
        """, source_id)

        domain_stats = await conn.fetch("""
            SELECT business_domain, COUNT(*) AS cnt
            FROM source_entities
            WHERE source_id = $1
              AND business_domain IS NOT NULL
            GROUP BY business_domain
            ORDER BY cnt DESC
        """, source_id)

        concept_stats = await conn.fetch("""
            SELECT business_concept, COUNT(*) AS cnt
            FROM source_entities
            WHERE source_id = $1
              AND business_concept IS NOT NULL
            GROUP BY business_concept
            ORDER BY cnt DESC
        """, source_id)

    enriched = stats["enriched"] or 0
    total    = stats["total"] or 1
    return {
        "source_id":       str(source_id),
        "source_name":     source.name,
        "total_entities":  stats["total"],
        "enriched":        enriched,
        "progress_pct":    round(enriched / total * 100, 1),
        "with_embedding":  stats["with_embedding"],
        "domains_count":   stats["domains_count"],
        "concepts_count":  stats["concepts_count"],
        "domain_stats":    {r["business_domain"]: r["cnt"] for r in domain_stats},
        "concept_stats":   {r["business_concept"]: r["cnt"] for r in concept_stats},
    }


@app.get("/search", tags=["Semantic"])
async def search_entities(
    q:          str   = Query(..., description="Requête en langage naturel"),
    source_ids: str   = Query(None, description="IDs sources séparés par virgule"),
    limit:      int   = Query(10, ge=1, le=50),
    use_vector: bool  = Query(True, description="Activer la recherche vectorielle"),
):
    """
    Recherche sémantique hybride sur toutes les entités indexées.

    Exemples :
    - /search?q=chiffre affaires client
    - /search?q=customer invoice total amount
    - /search?q=commande fournisseur&source_ids=uuid1,uuid2
    """
    if not _SEMANTIC_AVAILABLE:
        raise HTTPException(503, "Module semantic_enricher non disponible")

    sids = [s.strip() for s in source_ids.split(",")] if source_ids else None

    try:
        results = await _semantic_search(
            query=q,
            source_ids=sids,
            limit=limit,
            use_vector=use_vector,
        )
        return {
            "query":   q,
            "total":   len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"[Search] {q}: {e}", exc_info=True)
        raise HTTPException(500, f"Erreur recherche: {str(e)}")


@app.get("/sources/{source_id}/entities/by-domain/{domain}", tags=["Semantic"])
async def get_entities_by_domain(source_id: UUID, domain: str):
    """Retourne les entités d'un domaine métier spécifique."""
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT name, entity_type, business_concept, entity_class,
                   semantic_tags, row_count, description
            FROM source_entities
            WHERE source_id = $1
              AND LOWER(business_domain) = LOWER($2)
            ORDER BY entity_class, name
        """, source_id, domain)

    return {
        "source_id": str(source_id),
        "domain":    domain,
        "total":     len(rows),
        "entities":  [
            {
                "name":        r["name"],
                "type":        r["entity_type"],
                "concept":     r["business_concept"],
                "class":       r["entity_class"],
                "tags":        json_module.loads(r["semantic_tags"] or "[]"),
                "row_count":   r["row_count"],
                "description": r["description"],
            }
            for r in rows
        ],
    }


@app.get("/sources/{source_id}/dimensions", tags=["Semantic"])
async def get_source_dimensions(source_id: UUID):
    """
    Retourne toutes les dimensions analytiques détectées pour une source,
    avec les hiérarchies SQL (Year→Quarter→Month→Week→Day, etc.)
    Utile pour construire des dashboards et des analyses OLAP.
    """
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT se.name AS entity_name, ef.name AS field_name,
                   ef.dimension_type, ef.data_type,
                   ef.dimension_hierarchy
            FROM entity_fields ef
            JOIN source_entities se ON se.id = ef.entity_id
            WHERE se.source_id = $1
              AND ef.dimension_type IS NOT NULL
            ORDER BY ef.dimension_type, se.name, ef.name
        """, source_id)

    # Grouper par type de dimension
    dimensions: dict = {}
    for r in rows:
        dim = r["dimension_type"]
        if dim not in dimensions:
            dimensions[dim] = []

        # Parser la hiérarchie JSON
        hierarchy = None
        if r["dimension_hierarchy"]:
            try:
                import json as json_module
                raw = r["dimension_hierarchy"]
                hierarchy = json_module.loads(raw) if isinstance(raw, str) else dict(raw)
            except Exception:
                hierarchy = None

        dimensions[dim].append({
            "entity":    r["entity_name"],
            "field":     r["field_name"],
            "type":      r["data_type"],
            "hierarchy": hierarchy,
        })

    return {
        "source_id":  str(source_id),
        "dimensions": dimensions,
        "summary":    {dim: len(cols) for dim, cols in dimensions.items()},
    }


# ══════════════════════════════════════════════════════════════
# SYNONYMES CLIENT — §2.2.3.A
# ══════════════════════════════════════════════════════════════

@app.post("/sources/{source_id}/synonyms", tags=["Semantic"])
async def add_source_synonym(source_id: UUID, body: dict):
    """
    Ajoute ou met à jour un synonyme métier spécifique à une source.
    Ex: {"term": "BALI", "synonyms": ["balance", "solde"], "description": "Balance SXA"}
    """
    term        = body.get("term", "").strip().upper()
    synonyms    = body.get("synonyms", [])
    description = body.get("description", "")
    created_by  = body.get("created_by", "user")

    if not term or not synonyms:
        raise HTTPException(400, "term et synonyms sont requis")

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO source_synonyms (source_id, term, synonyms, description, created_by)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (source_id, term)
            DO UPDATE SET synonyms=$3, description=$4, updated_at=NOW()
            RETURNING *
        """, source_id, term, synonyms, description, created_by)

    # Synchroniser dans MeiliSearch
    try:
        from .semantic_enricher import _get_meili_client, MEILI_INDEX
        meili = _get_meili_client()
        if meili:
            # Charger tous les synonymes de la source pour MeiliSearch
            async with pool.acquire() as conn:
                all_rows = await conn.fetch(
                    "SELECT term, synonyms FROM source_synonyms WHERE source_id=$1",
                    source_id
                )
            meili_synonyms = {}
            for r in all_rows:
                t = r["term"].lower()
                syns = [s.lower() for s in r["synonyms"]]
                meili_synonyms[t] = syns
                for s in syns:
                    meili_synonyms[s] = meili_synonyms.get(s, []) + [t]

            meili.index(MEILI_INDEX).update_synonyms(meili_synonyms)
            logger.info(f"[Synonyms] MeiliSearch mis à jour: {len(meili_synonyms)} synonymes")
    except Exception as e:
        logger.warning(f"[Synonyms] MeiliSearch sync failed: {e}")

    return {
        "success":     True,
        "term":        term,
        "synonyms":    synonyms,
        "description": description,
        "message":     f"Synonyme '{term}' ajouté et synchronisé dans MeiliSearch",
    }


@app.get("/sources/{source_id}/synonyms", tags=["Semantic"])
async def list_source_synonyms(source_id: UUID):
    """Liste tous les synonymes métier d'une source."""
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT term, synonyms, description, created_by, created_at, updated_at
            FROM source_synonyms
            WHERE source_id = $1
            ORDER BY term
        """, source_id)

    return {
        "source_id": str(source_id),
        "count":     len(rows),
        "synonyms":  [
            {
                "term":        r["term"],
                "synonyms":    list(r["synonyms"]),
                "description": r["description"],
                "created_by":  r["created_by"],
                "updated_at":  r["updated_at"].isoformat() if r["updated_at"] else None,
            }
            for r in rows
        ],
    }


@app.delete("/sources/{source_id}/synonyms/{term}", tags=["Semantic"])
async def delete_source_synonym(source_id: UUID, term: str):
    """Supprime un synonyme métier et met à jour MeiliSearch."""
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM source_synonyms WHERE source_id=$1 AND term=$2",
            source_id, term.upper()
        )

    if result == "DELETE 0":
        raise HTTPException(404, f"Synonyme '{term}' introuvable")

    # Re-synchroniser MeiliSearch sans ce synonyme
    try:
        from .semantic_enricher import _get_meili_client, MEILI_INDEX
        meili = _get_meili_client()
        if meili:
            async with pool.acquire() as conn:
                all_rows = await conn.fetch(
                    "SELECT term, synonyms FROM source_synonyms WHERE source_id=$1",
                    source_id
                )
            meili_synonyms = {}
            for r in all_rows:
                t = r["term"].lower()
                syns = [s.lower() for s in r["synonyms"]]
                meili_synonyms[t] = syns
            meili.index(MEILI_INDEX).update_synonyms(meili_synonyms)
    except Exception as e:
        logger.warning(f"[Synonyms] MeiliSearch sync failed: {e}")

    return {"success": True, "message": f"Synonyme '{term}' supprimé"}


# ══════════════════════════════════════════════════════════════
# CALENDRIERS FISCAUX — §2.2.3.B
# ══════════════════════════════════════════════════════════════

@app.post("/sources/{source_id}/fiscal-calendar", tags=["Semantic"])
async def set_fiscal_calendar(source_id: UUID, body: dict):
    """
    Configure l'année fiscale d'une source.
    Ex: {"fiscal_year_start": 7, "description": "Exercice juillet-juin"}
    fiscal_year_start : 1=Janvier (défaut), 4=Avril, 7=Juillet, 10=Octobre
    """
    fiscal_year_start = int(body.get("fiscal_year_start", 1))
    description       = body.get("description", "")
    fiscal_year_label = body.get("fiscal_year_label", "FY")

    if not 1 <= fiscal_year_start <= 12:
        raise HTTPException(400, "fiscal_year_start doit être entre 1 et 12")

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO source_fiscal_calendars
                (source_id, fiscal_year_start, fiscal_year_label, description)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (source_id)
            DO UPDATE SET fiscal_year_start=$2, fiscal_year_label=$3,
                          description=$4, updated_at=NOW()
        """, source_id, fiscal_year_start, fiscal_year_label, description)

    month_names = {1:"Janvier",2:"Février",3:"Mars",4:"Avril",5:"Mai",6:"Juin",
                   7:"Juillet",8:"Août",9:"Septembre",10:"Octobre",11:"Novembre",12:"Décembre"}

    return {
        "success":           True,
        "fiscal_year_start": fiscal_year_start,
        "fiscal_year_label": fiscal_year_label,
        "description":       description,
        "message":           f"Année fiscale configurée: début en {month_names.get(fiscal_year_start, str(fiscal_year_start))}. Lancez un sync pour recalculer les hiérarchies.",
    }


@app.get("/sources/{source_id}/fiscal-calendar", tags=["Semantic"])
async def get_fiscal_calendar(source_id: UUID):
    """Retourne la configuration du calendrier fiscal d'une source."""
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM source_fiscal_calendars WHERE source_id=$1",
            source_id
        )

    if not row:
        return {
            "source_id":         str(source_id),
            "fiscal_year_start": 1,
            "fiscal_year_label": "FY",
            "description":       "Calendrier standard (janvier)",
            "configured":        False,
        }

    return {
        "source_id":         str(source_id),
        "fiscal_year_start": row["fiscal_year_start"],
        "fiscal_year_label": row["fiscal_year_label"],
        "description":       row["description"],
        "configured":        True,
        "updated_at":        row["updated_at"].isoformat() if row["updated_at"] else None,
    }


# ══════════════════════════════════════════════════════════════
# TAXONOMIE PERSONNALISABLE — §2.2.3.A
# ══════════════════════════════════════════════════════════════

@app.post("/sources/{source_id}/domains", tags=["Semantic"])
async def add_custom_domain(source_id: UUID, body: dict):
    """
    Ajoute un domaine métier personnalisé pour une source.
    Ex: {"domain_name": "Trésorerie", "patterns": ["tresor","cash","bnk"], "color": "#f59e0b"}
    """
    domain_name = body.get("domain_name", "").strip()
    patterns    = body.get("patterns", [])
    color       = body.get("color", "#6366f1")
    icon        = body.get("icon", "database")
    description = body.get("description", "")
    priority    = int(body.get("priority", 10))

    if not domain_name or not patterns:
        raise HTTPException(400, "domain_name et patterns sont requis")

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO source_domains
                (source_id, domain_name, patterns, color, icon, description, priority)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (source_id, domain_name)
            DO UPDATE SET patterns=$3, color=$4, icon=$5,
                          description=$6, priority=$7, updated_at=NOW()
        """, source_id, domain_name, patterns, color, icon, description, priority)

    return {
        "success":     True,
        "domain_name": domain_name,
        "patterns":    patterns,
        "message":     f"Domaine '{domain_name}' ajouté. Lancez un sync pour reclassifier les entités.",
    }


@app.get("/sources/{source_id}/domains", tags=["Semantic"])
async def list_custom_domains(source_id: UUID):
    """Liste tous les domaines personnalisés d'une source."""
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT domain_name, patterns, color, icon, description, priority, updated_at
            FROM source_domains
            WHERE source_id = $1
            ORDER BY priority DESC, domain_name
        """, source_id)

    return {
        "source_id": str(source_id),
        "count":     len(rows),
        "domains":   [
            {
                "domain_name": r["domain_name"],
                "patterns":    list(r["patterns"]),
                "color":       r["color"],
                "icon":        r["icon"],
                "description": r["description"],
                "priority":    r["priority"],
                "updated_at":  r["updated_at"].isoformat() if r["updated_at"] else None,
            }
            for r in rows
        ],
    }


@app.delete("/sources/{source_id}/domains/{domain_name}", tags=["Semantic"])
async def delete_custom_domain(source_id: UUID, domain_name: str):
    """Supprime un domaine personnalisé."""
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM source_domains WHERE source_id=$1 AND domain_name=$2",
            source_id, domain_name
        )
    if result == "DELETE 0":
        raise HTTPException(404, f"Domaine '{domain_name}' introuvable")
    return {"success": True, "message": f"Domaine '{domain_name}' supprimé. Relancez un sync."}


# ══════════════════════════════════════════════════════════════
# HISTORIQUE RECHERCHES — §2.2.3.C
# ══════════════════════════════════════════════════════════════

@app.post("/sources/{source_id}/search/click", tags=["Semantic"])
async def record_search_click(source_id: UUID, body: dict):
    """
    Enregistre le clic d'un utilisateur sur un résultat de recherche.
    Ex: {"query": "facture client", "entity_id": "uuid", "entity_name": "GS_GLACC"}
    Utilisé pour boosting des résultats futurs.
    """
    query       = body.get("query", "").strip()
    entity_id   = body.get("entity_id")
    entity_name = body.get("entity_name", "")
    user_id     = body.get("user_id")

    if not query or not entity_id:
        raise HTTPException(400, "query et entity_id sont requis")

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE search_history
            SET clicked_id=$1, clicked_name=$2
            WHERE source_id=$3
              AND query_norm=$4
              AND clicked_id IS NULL
              AND search_at = (
                SELECT MAX(search_at) FROM search_history
                WHERE source_id=$3 AND query_norm=$4
              )
        """, UUID(entity_id), entity_name, source_id, query.lower().strip())

    return {"success": True, "message": f"Clic enregistré pour '{entity_name}'"}


@app.get("/sources/{source_id}/search/history", tags=["Semantic"])
async def get_search_history(source_id: UUID, limit: int = 20):
    """Retourne les statistiques des recherches passées pour une source."""
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT query_norm, search_count, click_count,
                   click_rate_pct, last_searched_at, most_clicked
            FROM search_history_stats
            WHERE source_id = $1
            LIMIT $2
        """, source_id, limit)

    return {
        "source_id": str(source_id),
        "count":     len(rows),
        "history":   [
            {
                "query":           r["query_norm"],
                "search_count":    r["search_count"],
                "click_count":     r["click_count"],
                "click_rate_pct":  float(r["click_rate_pct"] or 0),
                "last_searched":   r["last_searched_at"].isoformat() if r["last_searched_at"] else None,
                "most_clicked":    r["most_clicked"],
            }
            for r in rows
        ],
    }


# ══════════════════════════════════════════════════════════════
# CDC / VERSIONING — §2.2.5
# ══════════════════════════════════════════════════════════════

def _get_cdc() -> CDCEngine:
    """Fabrique un CDCEngine (pool + redis déjà initialisés au lifespan)."""
    from .database import _pg_pool, _redis  # singletons module-level
    return CDCEngine(_pg_pool, _redis)


def _get_versioner() -> SchemaVersioner:
    from .database import _pg_pool, _redis
    return SchemaVersioner(_pg_pool, _redis)


# ── 1. Détecter les changements (CDC) ────────────────────────

@app.post("/sources/{source_id}/cdc/detect", tags=["CDC"])
async def cdc_detect(source_id: UUID):
    """
    Compare le schéma courant avec la dernière version enregistrée.
    - Crée une nouvelle version si des changements sont détectés.
    - Publie les breaking changes dans Redis.
    - Invalide le cache de la source.
    """
    cdc = _get_cdc()
    try:
        result = await cdc.detect_and_record(source_id)
        return result
    except Exception as e:
        logger.error(f"[CDC] detect error source {source_id}: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ── 2. Historique des versions (git log) ─────────────────────

@app.get("/sources/{source_id}/versions", tags=["CDC"])
async def schema_log(
    source_id: UUID,
    limit: int = Query(50, ge=1, le=200),
):
    """
    Retourne l'historique des versions de schéma pour une source.
    Enrichi avec les tags et le résumé des changements.
    """
    versioner = _get_versioner()
    return {
        "source_id": str(source_id),
        "versions":  await versioner.get_log(source_id, limit),
    }


# ── 3. Détail d'une version (git show) ───────────────────────

@app.get("/sources/{source_id}/versions/{version}", tags=["CDC"])
async def schema_version_detail(source_id: UUID, version: int):
    """
    Retourne le détail complet d'une version :
    snapshot complet du schéma + liste des deltas + tags.
    """
    versioner = _get_versioner()
    v = await versioner.get_version(source_id, version)
    if not v:
        raise HTTPException(404, f"Version {version} introuvable pour source {source_id}")
    return v


# ── 4. Diff entre deux versions (git diff) ───────────────────

@app.get("/sources/{source_id}/versions/diff", tags=["CDC"])
async def schema_diff(
    source_id: UUID,
    v1: int = Query(..., description="Version de base"),
    v2: int = Query(..., description="Version cible"),
):
    """
    Diff entre deux versions arbitraires.
    Retourne la liste complète des changements avec flag breaking.
    """
    versioner = _get_versioner()
    try:
        return await versioner.diff_versions(source_id, v1, v2)
    except ValueError as e:
        raise HTTPException(404, str(e))


# ── 5. Rollback vers une version passée ──────────────────────

@app.post("/sources/{source_id}/versions/{version}/rollback", tags=["CDC"])
async def schema_rollback(source_id: UUID, version: int):
    """
    Rollback logique : crée une nouvelle version HEAD
    qui restaure le snapshot d'une version passée.
    Ne modifie PAS la source de données réelle —
    sert à tracer le retour arrière dans l'historique OnePilot.
    """
    cdc = _get_cdc()
    try:
        return await cdc.rollback_to_version(source_id, version)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"[CDC] rollback error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ── 6. Tagger une version (git tag) ──────────────────────────

@app.post("/sources/{source_id}/versions/{version}/tag", tags=["CDC"])
async def schema_tag(source_id: UUID, version: int, body: dict):
    """
    Pose un tag nommé sur une version.
    Body: { "tag": "v1.2-stable", "note": "Avant migration SAGE" }
    """
    tag  = body.get("tag", "").strip()
    note = body.get("note", "")
    if not tag:
        raise HTTPException(400, "Le champ 'tag' est requis")

    versioner = _get_versioner()
    try:
        return await versioner.tag_version(source_id, version, tag, note)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.delete("/sources/{source_id}/tags/{tag}", tags=["CDC"])
async def schema_delete_tag(source_id: UUID, tag: str):
    """Supprime un tag."""
    versioner = _get_versioner()
    try:
        return await versioner.delete_tag(source_id, tag)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/sources/{source_id}/tags", tags=["CDC"])
async def schema_list_tags(source_id: UUID):
    """Liste tous les tags d'une source."""
    versioner = _get_versioner()
    tags = await versioner.list_tags(source_id)
    return {"source_id": str(source_id), "tags": tags}


# ── 7. Notifications breaking changes ────────────────────────

@app.get("/sources/{source_id}/cdc/notifications", tags=["CDC"])
async def cdc_notifications(
    source_id: UUID,
    limit: int = Query(50, ge=1, le=100),
):
    """
    Retourne les dernières notifications de breaking changes.
    Lecture depuis Redis (TTL 7j), fallback PostgreSQL.
    """
    versioner = _get_versioner()
    notifs = await versioner.get_notifications(source_id, limit)
    return {
        "source_id": str(source_id),
        "count":     len(notifs),
        "notifications": notifs,
    }


# ── 8. Impact analysis ────────────────────────────────────────

@app.get("/sources/{source_id}/versions/{version}/impact", tags=["CDC"])
async def schema_impact(source_id: UUID, version: int):
    """
    Analyse l'impact des breaking changes d'une version :
    - Tables et colonnes affectées
    - Jointures / relations potentiellement cassées
    """
    versioner = _get_versioner()
    try:
        return await versioner.impact_analysis(source_id, version)
    except ValueError as e:
        raise HTTPException(404, str(e))


# ── 9. CDC intégré au pipeline de sync ───────────────────────
# Le sync background (_run_sync_background) appelle detect_and_record
# automatiquement après chaque sync complet.
# Ce endpoint permet un CDC on-demand sans sync complet.

@app.post("/sources/{source_id}/cdc/watch", tags=["CDC"])
async def cdc_watch(
    source_id: UUID,
    background_tasks: BackgroundTasks,
    interval_seconds: int = Query(300, ge=30, le=3600),
):
    """
    Lance une détection CDC différée (une seule fois, non-récurrente).
    Pour un polling continu, utilise un cron externe ou Celery Beat.
    """
    async def _delayed_detect():
        import asyncio
        await asyncio.sleep(interval_seconds)
        cdc = _get_cdc()
        try:
            result = await cdc.detect_and_record(source_id)
            logger.info(f"[CDC Watch] source {source_id}: {result.get('status')}")
        except Exception as e:
            logger.error(f"[CDC Watch] error: {e}")

    background_tasks.add_task(_delayed_detect)
    return {
        "status":           "scheduled",
        "source_id":        str(source_id),
        "detect_in_seconds": interval_seconds,
        "message":          f"Détection CDC programmée dans {interval_seconds}s",
    }


# ══════════════════════════════════════════════════════════════
# SERVEUR FICHIERS STATIQUES UI
# ══════════════════════════════════════════════════════════════

import pathlib
UI_DIR = pathlib.Path(__file__).parent.parent / "ui"
app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


# ══════════════════════════════════════════════════════════════
# ENTRYPOINT (dev uniquement — en prod: uvicorn api.main:app)
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)