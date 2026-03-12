"""
OnePilot – API FastAPI
Universal Data Access Layer – Phase 3
"""
from __future__ import annotations

import logging
import os
import io
import csv as csv_module
import json as json_module
from contextlib import asynccontextmanager
from typing import Optional, List
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .database import init_schema, close_connections, get_pg_pool, get_redis
from .schemas  import (
    DataSourceCreate, DataSourceUpdate, DataSourceOut,
    DataSourceDetail, DataSourceList, ConnectionTestResult,
    MetadataSyncResult, ConnectorType, SourceCategory
)
from .relationship_discovery import (
    discover_relationships, get_relations_for_source,
    validate_relation, get_validation_stats,
    get_join_paths_for_source,
)
from .repository import (
    create_source, list_sources, get_source,
    get_source_with_entities, update_source, delete_source
)
from .connection_service import test_connection, sync_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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


app = FastAPI(
    title="OnePilot API",
    description="Universal Data Access Layer – Phase 3",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# SCHÉMAS PYDANTIC (import-fk + validate)
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


# ── Health ────────────────────────────────────────────────────

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
        content={"status": "healthy" if all_ok else "degraded", "checks": checks}
    )


# ── Connector Types ───────────────────────────────────────────

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
            ]
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
            ]
        },
        "file": {
            "label": "Fichiers",
            "description": "Import de fichiers locaux ou upload",
            "icon": "file",
            "types": [
                {"id": "file_csv",   "label": "CSV",   "icon": "📄", "default_port": None},
                {"id": "file_excel", "label": "Excel", "icon": "📊", "default_port": None},
                {"id": "file_json",  "label": "JSON",  "icon": "📋", "default_port": None},
            ]
        }
    }


# ── Sources CRUD ──────────────────────────────────────────────

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


# ── Test + Sync ───────────────────────────────────────────────

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
        tested_at=result.get("tested_at") or __import__("datetime").datetime.utcnow(),
    )


@app.post("/sources/{source_id}/sync", response_model=MetadataSyncResult, tags=["Connections"])
async def sync_source_metadata(source_id: UUID):
    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")

    result = await sync_metadata(source_id)

    if result.get("success") and result.get("entity_count", 0) > 0:
        try:
            discovery = await discover_relationships(source_id)
            result["relation_count"] = discovery.get("relations_found", 0)
            logger.info(f"[Sync] Relations découvertes: {discovery.get('relations_found', 0)}")
        except Exception as e:
            logger.warning(f"[Sync] Relationship discovery failed: {e}")

    return MetadataSyncResult(source_id=source_id, **result)


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


# ── Relations ────────────────────────────────────────────────

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
            "relations": relations
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
    relation_id est un bigint.
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
    source_id: UUID,
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


# ── Import FK depuis SSMS ────────────────────────────────────

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
        imported = 0
        updated  = 0
        skipped  = 0

        for rel in req.relations:
            # Vérifier que les tables existent dans notre source
            src_exists = await conn.fetchval(
                "SELECT 1 FROM source_entities WHERE source_id=$1 AND name=$2",
                source_id, rel.source_entity)
            tgt_exists = await conn.fetchval(
                "SELECT 1 FROM source_entities WHERE source_id=$1 AND name=$2",
                source_id, rel.target_entity)

            if not src_exists or not tgt_exists:
                skipped += 1
                continue

            result = await conn.execute("""
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
            """, source_id,
                rel.source_entity, rel.source_field,
                rel.target_entity, rel.target_field)

            if "INSERT 0 1" in result:
                imported += 1
            else:
                updated += 1

        return {
            "success":  True,
            "imported": imported,
            "updated":  updated,
            "skipped":  skipped,
            "message":  (f"{imported} FK importées, {updated} mises à jour, "
                         f"{skipped} ignorées (tables inconnues)"),
        }


# ── File Upload ───────────────────────────────────────────────

@app.post("/upload", tags=["Files"])
async def upload_file(file: UploadFile = File(...)):
    """Upload un fichier CSV ou JSON et retourne un aperçu des données."""
    allowed = {".csv", ".json", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Type de fichier non supporté : {ext}. Acceptés : {allowed}")

    content = await file.read()
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(content)

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    if ext in (".csv", ".txt"):
        reader = csv_module.DictReader(io.StringIO(text))
        rows = []
        for i, row in enumerate(reader):
            if i >= 5:
                break
            rows.append(dict(row))
        columns = list(rows[0].keys()) if rows else []
        return {
            "filename":      file.filename,
            "uploaded_path": dest,
            "size":          len(content),
            "format":        "csv",
            "columns":       columns,
            "column_count":  len(columns),
            "preview":       rows,
            "message":       f"Fichier uploadé — {len(columns)} colonnes détectées"
        }
    else:  # json
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
            "size":          len(content),
            "format":        "json",
            "columns":       columns,
            "column_count":  len(columns),
            "preview":       preview,
            "message":       f"Fichier uploadé — {len(columns)} colonnes détectées"
        }


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



# ── Import Vues SQL ──────────────────────────────────────────

class ImportViewsRequest(BaseModel):
    views: Optional[List[dict]] = None   # [{view_name, sql_definition}] pour mode manuel


@app.post("/sources/{source_id}/import-views", tags=["Relations"])
async def import_views_joins(source_id: UUID, req: ImportViewsRequest = ImportViewsRequest()):
    """
    Extrait les jointures depuis les vues SQL Server (sys.views) et les importe.
    Mode auto: se connecte à la source et lit sys.views directement.
    Mode manuel: accepte une liste de {view_name, sql_definition}.
    """
    from .view_parser import parse_views

    source = await get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source {source_id} introuvable")

    # ── Récupérer les définitions de vues ──
    views_list = req.views if req.views else None

    if not views_list:
        # Mode AUTO: connexion SQL Server pour lire sys.views
        try:
            views_list = await _fetch_views_from_db(source_id)
        except Exception as e:
            raise HTTPException(500, f"Impossible de lire les vues: {e}")

    if not views_list:
        return {"success": False, "message": "Aucune vue trouvée", "imported": 0, "updated": 0}

    # ── Parser les vues ──
    joins, views_analyzed, raw_count = parse_views(views_list)
    logger.info(f"[ImportViews] {views_analyzed} vues → {len(joins)} jointures")

    # ── Insérer en DB ──
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        imported = 0
        updated  = 0
        skipped  = 0

        for j in joins:
            src = j["source_entity"]
            tgt = j["target_entity"]

            # Vérifier si les tables existent dans cette source
            src_exists = await conn.fetchval(
                "SELECT 1 FROM source_entities WHERE source_id=$1 AND LOWER(name)=LOWER($2)",
                source_id, src)
            tgt_exists = await conn.fetchval(
                "SELECT 1 FROM source_entities WHERE source_id=$1 AND LOWER(name)=LOWER($2)",
                source_id, tgt)

            if not src_exists or not tgt_exists:
                skipped += 1
                continue

            # Ne pas écraser les FK réelles (explicit_fk)
            existing_method = await conn.fetchval("""
                SELECT detection_method FROM entity_relations
                WHERE source_id=$1 AND LOWER(source_entity)=LOWER($2)
                  AND LOWER(source_field)=LOWER($3) AND LOWER(target_entity)=LOWER($4)
            """, source_id, src, j["source_field"], tgt)

            if existing_method == "explicit_fk":
                skipped += 1
                continue

            result = await conn.execute("""
                INSERT INTO entity_relations
                    (source_id, source_entity, source_field,
                     target_entity, target_field,
                     relation_type, confidence, detection_method, is_confirmed)
                VALUES ($1,$2,$3,$4,$5,'many_to_one',1.0,'view_join',TRUE)
                ON CONFLICT (source_id, source_entity, source_field, target_entity)
                DO UPDATE SET
                    target_field=EXCLUDED.target_field,
                    confidence=1.0,
                    detection_method='view_join',
                    is_confirmed=TRUE
            """, source_id, src, j["source_field"], tgt, j["target_field"])

            if "INSERT 0 1" in result:
                imported += 1
            else:
                updated += 1

    return {
        "success":        True,
        "views_analyzed": views_analyzed,
        "joins_extracted": len(joins),
        "imported":       imported,
        "updated":        updated,
        "skipped":        skipped,
        "message": (f"{views_analyzed} vues analysées · "
                    f"{len(joins)} jointures extraites · "
                    f"{imported} nouvelles · {updated} mises à jour · "
                    f"{skipped} ignorées (FK réelle ou table inconnue)"),
    }


async def _fetch_views_from_db(source_id: UUID) -> List[dict]:
    """Lit sys.views depuis SQL Server via pyodbc."""
    import asyncio
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        source_row = await conn.fetchrow(
            "SELECT connector_type, host, port, database_name, username, options "
            "FROM data_sources WHERE id=$1", source_id)
        if not source_row:
            return []
        source_dict = dict(source_row)
        secret_rows = await conn.fetch(
            "SELECT secret_key, secret_value FROM connection_secrets WHERE source_id=$1", source_id)
        for sr in secret_rows:
            if sr["secret_key"] == "password":
                source_dict["password"] = sr["secret_value"]

    ct = (source_dict.get("connector_type") or "").lower()
    if not any(x in ct for x in ["mssql", "sql_server", "sqlserver"]):
        return []

    def _sync_fetch(sd):
        import pyodbc, json as _json
        opts = sd.get("options") or {}
        if isinstance(opts, str):
            try: opts = _json.loads(opts)
            except: opts = {}
        host = sd.get("host") or opts.get("host", "localhost")
        port = sd.get("port") or opts.get("port", 1433)
        db   = sd.get("database_name") or ""
        user = sd.get("username") or ""
        pwd  = sd.get("password", "")

        available = [d for d in pyodbc.drivers() if "SQL Server" in d]
        driver = next((d for d in ["ODBC Driver 18 for SQL Server",
                                    "ODBC Driver 17 for SQL Server"] if d in available),
                       available[0] if available else "ODBC Driver 18 for SQL Server")

        conn_str = (f"DRIVER={{{driver}}};SERVER={host},{port};DATABASE={db};"
                    f"UID={user};PWD={pwd};TrustServerCertificate=yes;Encrypt=no")
        conn = pyodbc.connect(conn_str, timeout=30)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT v.name AS view_name, m.definition AS sql_definition
            FROM sys.views v
            JOIN sys.sql_modules m ON v.object_id = m.object_id
            WHERE v.type = 'V'
            ORDER BY v.name
        """)
        rows = cursor.fetchall()
        conn.close()
        return [{"view_name": r[0], "sql_definition": r[1]} for r in rows]

    return await asyncio.to_thread(_sync_fetch, source_dict)


# ── Profiling ────────────────────────────────────────────────

@app.get("/sources/{source_id}/profile", tags=["Profiling"])
async def get_source_profile(source_id: UUID):
    """Résumé de profiling pour toutes les tables déjà profilees."""
    try:
        from .data_profiler import profile_source_summary
        return await profile_source_summary(source_id)
    except Exception as e:
        logger.error(f"[Profile] {source_id}: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/sources/{source_id}/profile/{table_name}", tags=["Profiling"])
async def get_table_profile(source_id: UUID, table_name: str, refresh: bool = False):
    """Profile une table spécifique (avec cache)."""
    try:
        from .data_profiler import profile_entity, get_cached_profile
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
    """Profile plusieurs tables en parallèle (max 10)."""
    import asyncio
    from .data_profiler import profile_entity
    tables = (body.get("tables") or [])[:10]
    if not tables:
        raise HTTPException(400, "tables[] requis")
    results = await asyncio.gather(
        *[profile_entity(source_id, t) for t in tables],
        return_exceptions=True
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
    sample_size: int  = Query(1000, ge=100, le=10000, description="Lignes à échantillonner par table"),
    batch_size:  int  = Query(5,    ge=1,   le=20,    description="Tables en parallèle par micro-batch"),
):
    """
    Profile TOUTES les entités de la source en séquence par micro-batches.
    - Reprend automatiquement là où il s'est arrêté (skip entités déjà OK)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)



# ======================================================================
# FIX P2 — PROFILING COMPLET : POST /sources/{id}/profile/all
# Lance le profiling de TOUTES les entités en séquence (reprend si interrompu)
# ======================================================================
