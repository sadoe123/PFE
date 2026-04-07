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
            try:
                discovery = await discover_relationships(source_id)
                logger.info(f"[Sync BG] Relations découvertes: {discovery.get('relations_found', 0)}")
            except Exception as e:
                logger.warning(f"[Sync BG] Relationship discovery failed (non-bloquant): {e}")
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
        from .api.cross_source_mapper import (  # type: ignore
            detect_cross_source_mappings, save_cross_source_mappings
        )
    except ImportError:
        raise HTTPException(503, "cross_source_mapper.py non disponible dans /api/")

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
        from .api.cross_source_mapper import build_multi_source_graph  # type: ignore
    except ImportError:
        raise HTTPException(503, "cross_source_mapper.py non disponible dans /api/")

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
        from .api.cross_source_mapper import get_cross_source_mappings  # type: ignore
    except ImportError:
        raise HTTPException(503, "cross_source_mapper.py non disponible dans /api/")
    return await get_cross_source_mappings(source_id_a, source_id_b, min_confidence)


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