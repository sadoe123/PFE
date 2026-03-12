"""
OnePilot – view_parser.py  §2.2.2A
Extraction des jointures depuis les définitions SQL des vues.
Gère : alias courts (A, AB, UI, F), noms avec espaces ([cours marchés]),
       schéma dbo., triple notation dbo.Table.Col
"""
from __future__ import annotations
import re, logging
from typing import Dict, List, Optional, Tuple, Set
from uuid import UUID

logger = logging.getLogger(__name__)

# ── Mots-clés SQL à ignorer comme alias ───────────────────────────────
_KW = {
    'inner','outer','left','right','full','cross','join','on','where',
    'group','order','having','select','from','as','with','union','all',
    'and','or','not','in','is','null','by','asc','desc','top','distinct',
    'case','when','then','else','end','set','insert','update','delete',
    'begin','go','use','exec','execute','declare','if','else','return',
}

# ══════════════════════════════════════════════════════════════════════
# EXTRACTION DES ALIAS (3 passes)
# ══════════════════════════════════════════════════════════════════════

def _extract_aliases(sql: str) -> Dict[str, str]:
    """
    Construit alias_map (ALIAS_UPPER → vrai_nom_table) en 3 passes :
    1. [schema].[table avec espaces] alias   ← noms entre crochets
    2. schema.table alias                    ← notation pointée sans crochets
    3. table alias                           ← table seule sans schéma
    """
    alias_map: Dict[str, str] = {}

    def _add(tbl: str, alias: str):
        t = tbl.strip().strip('[]')
        a = alias.strip() if alias else ''
        if not t or t.lower() in _KW:
            return
        alias_map[t.upper()] = t
        if a and a.lower() not in _KW and len(a) >= 1:
            alias_map[a.upper()] = t

    # Passe 1 — [schema].[Table Name]
    for m in re.finditer(
        r'(?:FROM|JOIN)\s+\[[\w]+\]\s*\.\s*\[([\w\s]+)\]'
        r'(?:\s+(?:AS\s+)?(\w+))?',
        sql, re.IGNORECASE
    ):
        _add(m.group(1), m.group(2) or '')

    # Passe 2 — schema.Table
    for m in re.finditer(
        r'(?:FROM|JOIN)\s+\[?[\w]+\]?\s*\.\s*\[?([\w]+)\]?'
        r'(?:\s+(?:AS\s+)?(\w+))?'
        r'(?=\s+(?:JOIN|ON|WHERE|INNER|LEFT|RIGHT|FULL|CROSS|$)|\s+\w+\s+(?:JOIN|ON))',
        sql, re.IGNORECASE
    ):
        _add(m.group(1), m.group(2) or '')

    # Passe 3 — Table seule (sans schéma)
    for m in re.finditer(
        r'(?:FROM|JOIN)\s+(?!\[?[\w]+\]?\s*\.)'
        r'\[?([\w\s]+?)\]?'
        r'(?:\s+(?:AS\s+)?(\w+))?'
        r'(?=\s+(?:JOIN|ON|WHERE|INNER|LEFT|RIGHT|FULL|CROSS)|$)',
        sql, re.IGNORECASE
    ):
        tbl   = m.group(1).strip().strip('[]')
        alias = m.group(2) or ''
        if tbl and tbl.lower() not in _KW and len(tbl) > 1:
            if tbl.upper() not in alias_map:
                _add(tbl, alias)
            elif alias:
                a = alias.strip()
                if a and a.lower() not in _KW:
                    alias_map[a.upper()] = alias_map[tbl.upper()]

    return alias_map


def _parse_colref(ref: str) -> Optional[Tuple[str, str]]:
    """
    Parse une référence colonne :
    - alias.col            → (alias, col)
    - schema.table.col     → (table, col)
    - [schema].[table].col → (table, col)
    """
    ref = ref.strip().replace('[', '').replace(']', '')
    parts = [p.strip() for p in ref.split('.') if p.strip()]
    if len(parts) == 2:
        return parts[0], parts[1]
    elif len(parts) >= 3:
        return parts[-2], parts[-1]
    return None


def _resolve_tbl(raw: str, alias_map: Dict[str, str]) -> Optional[str]:
    """Résout un alias ou nom de table."""
    k = raw.strip().upper()
    if k in alias_map:
        return alias_map[k]
    if raw.lower() not in _KW and raw.strip():
        return raw.strip()
    return None


# ══════════════════════════════════════════════════════════════════════
# PARSER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

_RE_JOIN = re.compile(
    r'(?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?'
    r'JOIN\s+'
    r'(?:\[?[\w]+\]?\s*\.\s*)?'      # schema optionnel
    r'\[?([\w\s]+?)\]?'              # nom table
    r'(?:\s+(?:AS\s+)?\w+)?'        # alias optionnel
    r'\s+ON\s+'
    r'([\w.\[\]]+)\s*=\s*([\w.\[\]]+)',
    re.IGNORECASE,
)


def parse_view_sql(view_name: str, sql: str) -> List[Dict]:
    if not sql:
        return []

    sql = re.sub(r'\s+', ' ', sql).strip()
    alias_map = _extract_aliases(sql)
    logger.debug(f"[VP] {view_name}: alias_map={alias_map}")

    relations: List[Dict] = []
    seen: Set[Tuple] = set()

    for m in _RE_JOIN.finditer(sql):
        lref = _parse_colref(m.group(2) or '')
        rref = _parse_colref(m.group(3) or '')
        if not lref or not rref:
            continue

        lt = _resolve_tbl(lref[0], alias_map)
        rt = _resolve_tbl(rref[0], alias_map)
        lc, rc = lref[1], rref[1]

        if not lt or not rt:
            continue
        # Filtrer artefacts
        if lt.lower() in ('dbo', 'schema') or rt.lower() in ('dbo', 'schema'):
            continue
        if lt.upper() == rt.upper():
            continue

        key = tuple(sorted([(lt.upper(), lc.upper()), (rt.upper(), rc.upper())]))
        if key in seen:
            continue
        seen.add(key)

        relations.append({
            "view_name":     view_name,
            "source_entity": lt,
            "source_field":  lc,
            "target_entity": rt,
            "target_field":  rc,
        })

    logger.debug(f"[VP] {view_name}: {len(relations)} jointures")
    return relations


def parse_multiple_views(views: List[Dict]) -> List[Dict]:
    all_rels: List[Dict] = []
    seen: Set[Tuple] = set()
    for v in views:
        for r in parse_view_sql(v.get("name",""), v.get("definition","")):
            key = tuple(sorted([
                (r["source_entity"].upper(), r["source_field"].upper()),
                (r["target_entity"].upper(), r["target_field"].upper()),
            ]))
            if key not in seen:
                seen.add(key)
                all_rels.append(r)
    logger.info(f"[VP] {len(views)} vues → {len(all_rels)} jointures uniques")
    return all_rels


# ══════════════════════════════════════════════════════════════════════
# FETCH SQL SERVER
# ══════════════════════════════════════════════════════════════════════

FETCH_VIEWS_SQL = {
    "mssql": """
        SELECT v.name AS view_name, m.definition AS sql_definition
        FROM sys.views v JOIN sys.sql_modules m ON v.object_id = m.object_id
        WHERE v.type = 'V' ORDER BY v.name
    """,
    "postgresql": """
        SELECT viewname AS view_name, definition AS sql_definition
        FROM pg_views WHERE schemaname NOT IN ('information_schema','pg_catalog')
        ORDER BY viewname
    """,
    "mysql": """
        SELECT TABLE_NAME AS view_name, VIEW_DEFINITION AS sql_definition
        FROM information_schema.VIEWS WHERE TABLE_SCHEMA = DATABASE()
        ORDER BY TABLE_NAME
    """,
}


def _fetch_views_sync(source_dict: Dict) -> List[Dict]:
    import pyodbc
    from sqlalchemy import create_engine, text

    ct = (source_dict.get("connector_type") or "").lower()
    dialect = "mssql"
    if "postgres" in ct or ct == "pg":
        dialect = "postgresql"
    elif "mysql" in ct:
        dialect = "mysql"

    drivers = pyodbc.drivers()
    driver  = next(
        (d for d in ["ODBC Driver 18 for SQL Server",
                     "ODBC Driver 17 for SQL Server",
                     "ODBC Driver 13 for SQL Server",
                     "SQL Server"]
         if d in drivers),
        drivers[0] if drivers else "ODBC Driver 17 for SQL Server"
    )

    host = source_dict.get("host","")
    port = source_dict.get("port") or 1433
    db   = source_dict.get("database_name","")
    user = source_dict.get("username","")
    pwd  = source_dict.get("password","")
    trust = "yes" if "18" in driver else "no"

    url = (
        f"mssql+pyodbc://{user}:{pwd}@{host}:{port}/{db}"
        f"?driver={driver.replace(' ','+')}"
        f"&TrustServerCertificate={trust}&Encrypt=no"
    )
    engine = create_engine(url, pool_pre_ping=True, pool_size=1, max_overflow=0)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(FETCH_VIEWS_SQL[dialect]))
            rows   = result.fetchall()
            keys   = list(result.keys())
        return [dict(zip(keys, r)) for r in rows]
    finally:
        engine.dispose()


# ══════════════════════════════════════════════════════════════════════
# SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════

async def _save_view_relations(pool, source_id: UUID,
                                relations: List[Dict]) -> Dict:
    imported = updated = skipped = 0
    async with pool.acquire() as conn:
        for r in relations:
            try:
                ex = await conn.fetchrow(
                    """SELECT id, detection_method FROM entity_relations
                       WHERE source_id=$1 AND source_entity=$2
                         AND source_field=$3 AND target_entity=$4""",
                    source_id, r["source_entity"],
                    r["source_field"], r["target_entity"],
                )
                if ex:
                    if ex["detection_method"] == "explicit_fk":
                        skipped += 1; continue
                    await conn.execute(
                        """UPDATE entity_relations
                           SET target_field=$1, confidence=1.0,
                               detection_method='view_join',
                               is_confirmed=TRUE, view_name=$2
                           WHERE id=$3""",
                        r["target_field"], r.get("view_name"), ex["id"],
                    )
                    updated += 1
                else:
                    await conn.execute(
                        """INSERT INTO entity_relations
                           (source_id,source_entity,source_field,
                            target_entity,target_field,
                            relation_type,confidence,
                            detection_method,is_confirmed,view_name)
                           VALUES($1,$2,$3,$4,$5,
                                  'many_to_one',1.0,
                                  'view_join',TRUE,$6)""",
                        source_id,
                        r["source_entity"], r["source_field"],
                        r["target_entity"], r["target_field"],
                        r.get("view_name"),
                    )
                    imported += 1
            except Exception as e:
                logger.warning(f"[VP] save {r}: {e}")
    return {"imported": imported, "updated": updated, "skipped": skipped}


# ══════════════════════════════════════════════════════════════════════
# POINTS D'ENTRÉE ASYNC
# ══════════════════════════════════════════════════════════════════════

async def _get_source_dict(pool, source_id: UUID) -> Optional[Dict]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT connector_type,host,port,database_name,username
               FROM data_sources WHERE id=$1""", source_id)
        if not row:
            return None
        d = dict(row)
        for s in await conn.fetch(
            "SELECT secret_key,secret_value FROM connection_secrets WHERE source_id=$1",
            source_id
        ):
            if s["secret_key"] == "password":
                d["password"] = s["secret_value"]
        if "password" not in d:
            d["password"] = ""
        return d


async def import_views_from_db(source_id: UUID) -> Dict:
    import asyncio
    from .database import get_pg_pool
    pool = await get_pg_pool()

    src = await _get_source_dict(pool, source_id)
    if not src:
        return {"success": False, "error": "Source introuvable"}

    try:
        rows = await asyncio.to_thread(_fetch_views_sync, src)
    except Exception as e:
        logger.error(f"[VP] fetch: {e}")
        return {"success": False, "error": str(e)}

    if not rows:
        return {"success": True, "views_found": 0,
                "message": "Aucune vue trouvée"}

    views = [{"name": r.get("view_name",""),
              "definition": r.get("sql_definition","")}
             for r in rows if r.get("view_name") and r.get("sql_definition")]

    rels  = parse_multiple_views(views)
    stats = await _save_view_relations(pool, source_id, rels)

    return {
        "success": True, "views_found": len(views),
        "relations_extracted": len(rels),
        "imported": stats["imported"],
        "updated":  stats["updated"],
        "skipped":  stats["skipped"],
        "message": (f"{len(views)} vues → "
                    f"{stats['imported']} nouvelles, "
                    f"{stats['updated']} mises à jour, "
                    f"{stats['skipped']} ignorées"),
    }


async def import_views_from_paste(source_id: UUID,
                                   views: List[Dict]) -> Dict:
    from .database import get_pg_pool
    pool  = await get_pg_pool()
    rels  = parse_multiple_views(views)
    stats = await _save_view_relations(pool, source_id, rels)
    return {
        "success": True, "views_parsed": len(views),
        "relations_extracted": len(rels),
        "imported": stats["imported"],
        "updated":  stats["updated"],
        "skipped":  stats["skipped"],
        "message": f"{len(views)} vues → {stats['imported']} nouvelles",
    }


async def get_view_relations_stats(source_id: UUID) -> Dict:
    from .database import get_pg_pool
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM entity_relations "
            "WHERE source_id=$1 AND detection_method='view_join'",
            source_id)
        by_view = await conn.fetch(
            """SELECT view_name, COUNT(*) AS cnt FROM entity_relations
               WHERE source_id=$1 AND detection_method='view_join'
               GROUP BY view_name ORDER BY cnt DESC""", source_id)
        top = await conn.fetch(
            """SELECT source_entity, COUNT(*) AS cnt FROM entity_relations
               WHERE source_id=$1 AND detection_method='view_join'
               GROUP BY source_entity ORDER BY cnt DESC LIMIT 10""",
            source_id)
    return {
        "total":      total or 0,
        "by_view":    [{"view": r["view_name"], "count": r["cnt"]}
                       for r in by_view],
        "top_tables": [{"table": r["source_entity"], "count": r["cnt"]}
                       for r in top],
    }