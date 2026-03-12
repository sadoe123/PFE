"""
OnePilot - Phase 3 : Intelligent Relationship Discovery
========================================================
Schema entity_relations :
  id, source_id, source_entity, source_field,
  target_entity, target_field, relation_type,
  confidence, detection_method, is_confirmed,
  validated_by, validated_at, reject_reason,
  value_overlap, created_at

4 algorithmes actifs :
  1. explicit_fk  — FKs déclarées (MSSQL/MySQL/PostgreSQL)     → confidence=1.0
  2. name_based   — Pattern matching pascal/snake/m2m           → 0.60-0.93
  3. value_based  — Analyse statistique valeurs (MSSQL)         → 0.55-0.95
  4. fuzzy_match  — Distance Levenshtein sur noms de colonnes   → 0.60-0.74

FIX P3 : ODBC Driver auto-détecté (plus de Driver 17 hardcodé)
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from dataclasses import dataclass, field as dc_field

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class RelationCandidate:
    source_entity:    str
    source_field:     str
    target_entity:    str
    target_field:     str
    confidence:       float
    detection_method: str
    relation_type:    str = "many_to_one"
    features:         Dict = dc_field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES COMMUNS
# ══════════════════════════════════════════════════════════════════════

def _to_words(name: str) -> List[str]:
    s = name.replace("_", " ").replace("-", " ")
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [w.lower() for w in s.split() if w]


def _normalize(name: str) -> str:
    return "_".join(_to_words(name))


FK_SUFFIXES = {
    "id", "code", "no", "num", "number", "key", "ref",
    "uuid", "guid", "fk", "pk", "idfk", "cd", "nr",
}
FK_PREFIXES = {"id", "fk", "ref"}

AUDIT_PATTERNS = re.compile(r"_A$|_LOG$|_HIST$|_ARC$|_BAK$|_TEMP$|_TMP$", re.IGNORECASE)
MTM_PATTERNS   = re.compile(r"_2_|_TO_|_X_|_LINK|_MAP|_REL|_ASSOC",        re.IGNORECASE)


def _is_fk_field(words: List[str]) -> Tuple[bool, str]:
    if not words:
        return False, ""
    if len(words) >= 2 and words[-1] in FK_SUFFIXES:
        return True, "_".join(words[:-1])
    if len(words) >= 2 and words[0] in FK_PREFIXES:
        return True, "_".join(words[1:])
    return False, ""


def _match_entity(base: str, entity_index: Dict[str, str], min_len: int = 3) -> Optional[str]:
    b = base.lower()
    if len(b) < min_len:
        return None
    if b in entity_index:
        return entity_index[b]
    for variant in [
        b + "s",
        b + "es",
        b[:-1] if b.endswith("s") and len(b) > 4 else None,
        b[:-2] if b.endswith("es") and len(b) > 5 else None,
    ]:
        if variant and variant in entity_index:
            return entity_index[variant]
    for key, real_name in entity_index.items():
        if key.endswith("_" + b) or key == b:
            return real_name
        if b.endswith("_" + key) or b == key:
            return real_name
    return None


def _get_pk_field(entity_name: str, fields_by_entity: Dict[str, List[Dict]]) -> Optional[str]:
    fields = fields_by_entity.get(entity_name, [])
    pk = [f for f in fields if f.get("is_primary_key")]
    if len(pk) == 1:
        return pk[0]["name"]
    for f in fields:
        w = _to_words(f["name"])
        if len(w) <= 2 and w[-1] in ("id", "code"):
            return f["name"]
    return pk[0]["name"] if pk else None


def _levenshtein(a: str, b: str) -> int:
    """Distance de Levenshtein pour fuzzy_match."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = curr
    return prev[-1]


# ══════════════════════════════════════════════════════════════════════
# FIX P3 — ODBC DRIVER AUTO-DÉTECTÉ
# Préfère Driver 18, fallback 17, puis tout autre disponible
# Harmonisé avec data_profiler.py._detect_odbc_driver()
# ══════════════════════════════════════════════════════════════════════

def _detect_odbc_driver() -> str:
    """Auto-détection du driver ODBC SQL Server installé sur le système."""
    try:
        import pyodbc
        available = [d for d in pyodbc.drivers()
                     if "SQL Server" in d or "ODBC" in d.upper()]
        for preferred in [
            "ODBC Driver 18 for SQL Server",
            "ODBC Driver 17 for SQL Server",
            "ODBC Driver 13 for SQL Server",
            "SQL Server",
        ]:
            if preferred in available:
                logger.debug(f"[discovery] Driver ODBC sélectionné: {preferred}")
                return preferred
        if available:
            logger.debug(f"[discovery] Driver ODBC fallback: {available[0]}")
            return available[0]
    except Exception as e:
        logger.warning(f"[discovery] pyodbc.drivers() erreur: {e}")
    return "ODBC Driver 18 for SQL Server"


# ══════════════════════════════════════════════════════════════════════
# DSN — MULTI-DB
# ══════════════════════════════════════════════════════════════════════

def _get_db_type(conn_info: Optional[Dict]) -> str:
    if not conn_info:
        return ""
    t = conn_info.get("source_type", "") or conn_info.get("connector_type", "")
    if t in ("mssql", "sqlserver", "sql_server", "sage_100"):
        return "mssql"
    if t == "mysql":
        return "mysql"
    if t in ("postgresql", "postgres"):
        return "postgresql"
    return t


def _build_dsn_mssql(conn_info: Dict) -> str:
    """DSN MSSQL avec driver auto-détecté (FIX P3)."""
    h      = conn_info.get("host", "localhost")
    p      = conn_info.get("port", 1433)
    d      = conn_info.get("database_name") or conn_info.get("database", "")
    u      = conn_info.get("username", "")
    pw     = conn_info.get("password", "")
    driver = _detect_odbc_driver()
    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={h},{p};DATABASE={d};UID={u};PWD={pw};"
        f"TrustServerCertificate=yes;"
    )


def _build_dsn_mysql(conn_info: Dict) -> str:
    h  = conn_info.get("host", "localhost")
    p  = conn_info.get("port", 3306)
    d  = conn_info.get("database_name") or conn_info.get("database", "")
    u  = conn_info.get("username", "")
    pw = conn_info.get("password", "")
    return (
        f"DRIVER={{MySQL ODBC 8.0 Unicode Driver}};"
        f"SERVER={h};PORT={p};DATABASE={d};UID={u};PWD={pw};"
    )


# ══════════════════════════════════════════════════════════════════════
# ALGO 1 — FK EXPLICITES (MSSQL / MySQL / PostgreSQL)
# ══════════════════════════════════════════════════════════════════════

async def _algo_explicit_fk(
    entity_names_set: set,
    conn_info:        Optional[Dict],
) -> List[RelationCandidate]:
    results: List[RelationCandidate] = []
    if not conn_info:
        return results

    db_type = _get_db_type(conn_info)

    # ── MSSQL ─────────────────────────────────────────────────
    if db_type == "mssql":
        try:
            import pyodbc
            import asyncio as _asyncio
            dsn = _build_dsn_mssql(conn_info)
            QUERY = """
                SELECT
                    OBJECT_NAME(fkc.parent_object_id)                             AS src_table,
                    COL_NAME(fkc.parent_object_id,   fkc.parent_column_id)        AS src_col,
                    OBJECT_NAME(fkc.referenced_object_id)                          AS tgt_table,
                    COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id)  AS tgt_col
                FROM sys.foreign_keys fk
                JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                ORDER BY src_table, src_col
            """
            def _sync():
                with pyodbc.connect(dsn, timeout=15) as mc:
                    with mc.cursor() as cur:
                        cur.execute(QUERY)
                        return cur.fetchall()

            loop = _asyncio.get_event_loop()
            rows = await loop.run_in_executor(None, _sync)
            for row in rows:
                src_t, src_c, tgt_t, tgt_c = row[0], row[1], row[2], row[3]
                if src_t not in entity_names_set or tgt_t not in entity_names_set:
                    continue
                if src_t == tgt_t:
                    continue
                results.append(RelationCandidate(
                    source_entity=src_t, source_field=src_c,
                    target_entity=tgt_t, target_field=tgt_c,
                    confidence=1.0, detection_method="explicit_fk"))
            logger.info(f"[explicit_fk/mssql] {len(results)} FKs")
        except Exception as e:
            logger.warning(f"[explicit_fk/mssql] {e}")

    # ── MySQL ─────────────────────────────────────────────────
    elif db_type == "mysql":
        try:
            import pyodbc
            import asyncio as _asyncio
            dsn = _build_dsn_mysql(conn_info)
            db  = conn_info.get("database_name") or conn_info.get("database", "")
            QUERY = f"""
                SELECT TABLE_NAME, COLUMN_NAME,
                       REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = '{db}'
                  AND REFERENCED_TABLE_NAME IS NOT NULL
            """
            def _sync():
                with pyodbc.connect(dsn, timeout=15) as mc:
                    with mc.cursor() as cur:
                        cur.execute(QUERY)
                        return cur.fetchall()

            loop = _asyncio.get_event_loop()
            rows = await loop.run_in_executor(None, _sync)
            for row in rows:
                src_t, src_c, tgt_t, tgt_c = row[0], row[1], row[2], row[3]
                if src_t not in entity_names_set or tgt_t not in entity_names_set:
                    continue
                results.append(RelationCandidate(
                    source_entity=src_t, source_field=src_c,
                    target_entity=tgt_t, target_field=tgt_c,
                    confidence=1.0, detection_method="explicit_fk"))
            logger.info(f"[explicit_fk/mysql] {len(results)} FKs")
        except Exception as e:
            logger.warning(f"[explicit_fk/mysql] {e}")

    # ── PostgreSQL — asyncpg (pool déjà disponible) ───────────
    elif db_type == "postgresql":
        try:
            from .database import get_pg_pool
            pool = await get_pg_pool()
            QUERY = """
                SELECT
                    tc.table_name       AS src_table,
                    kcu.column_name     AS src_col,
                    ccu.table_name      AS tgt_table,
                    ccu.column_name     AS tgt_col
                FROM information_schema.table_constraints       tc
                JOIN information_schema.key_column_usage        kcu
                     ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema    = kcu.table_schema
                JOIN information_schema.referential_constraints rc
                     ON tc.constraint_name = rc.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                     ON rc.unique_constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema    = 'public'
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(QUERY)
            for row in rows:
                src_t = row["src_table"]; src_c = row["src_col"]
                tgt_t = row["tgt_table"]; tgt_c = row["tgt_col"]
                if src_t not in entity_names_set or tgt_t not in entity_names_set:
                    continue
                results.append(RelationCandidate(
                    source_entity=src_t, source_field=src_c,
                    target_entity=tgt_t, target_field=tgt_c,
                    confidence=1.0, detection_method="explicit_fk"))
            logger.info(f"[explicit_fk/postgresql] {len(results)} FKs")
        except Exception as e:
            logger.warning(f"[explicit_fk/postgresql] {e}")

    return results


# ══════════════════════════════════════════════════════════════════════
# ALGO 2 — NAME-BASED (pascal + snake + m2m)
# ══════════════════════════════════════════════════════════════════════

def _algo_name_based(
    entity_index:     Dict[str, str],
    fields_by_entity: Dict[str, List[Dict]],
) -> List[RelationCandidate]:
    results: List[RelationCandidate] = []
    seen = set()

    for src_name, src_fields in fields_by_entity.items():
        src_words = _to_words(src_name)
        is_audit  = bool(AUDIT_PATTERNS.search(src_name))
        is_m2m    = bool(MTM_PATTERNS.search(src_name))

        for f in src_fields:
            if f.get("is_primary_key") and not f.get("is_foreign_key"):
                continue
            is_meta_fk  = bool(f.get("is_foreign_key"))
            field_words = _to_words(f["name"])
            is_fk, base = _is_fk_field(field_words)
            if not is_fk or not base or len(base) < 3:
                continue

            src_norm = _normalize(src_name)
            if base == src_norm or "_".join(src_words).endswith(base):
                continue

            tgt_name = _match_entity(base, entity_index)
            if not tgt_name or tgt_name == src_name:
                continue

            tgt_pk = _get_pk_field(tgt_name, fields_by_entity)
            if not tgt_pk:
                continue

            key = f"{src_name}.{f['name']}>{tgt_name}"
            if key in seen:
                continue
            seen.add(key)

            tgt_words  = _to_words(tgt_name)
            base_words = base.split("_")

            if base_words == tgt_words:
                score = 0.93
            elif "_".join(base_words) == "_".join(tgt_words):
                score = 0.91
            elif len(tgt_words) >= len(base_words) and tgt_words[-len(base_words):] == base_words:
                score = 0.87
            else:
                bs = set(base_words); ts = set(tgt_words)
                j  = len(bs & ts) / max(len(bs | ts), 1)
                score = 0.60 + 0.30 * j

            if is_meta_fk:
                score = min(score + 0.05, 0.99)
            if is_audit:
                score = max(score - 0.08, 0.50)

            method = (
                "name_m2m"    if is_m2m
                else "name_pascal" if any(c.isupper() for c in f["name"][1:])
                else "name_snake"
            )

            results.append(RelationCandidate(
                source_entity=src_name,     source_field=f["name"],
                target_entity=tgt_name,     target_field=tgt_pk,
                confidence=round(score, 3), detection_method=method,
                features={
                    "base":        base,
                    "is_audit":    is_audit,
                    "is_m2m":      is_m2m,
                    "metadata_fk": is_meta_fk,
                }))

    logger.info(f"[name_based] {len(results)} candidats")
    return results


# ══════════════════════════════════════════════════════════════════════
# ALGO 3 — VALUE-BASED (analyse statistique, MSSQL uniquement)
# ══════════════════════════════════════════════════════════════════════

async def _algo_value_based(
    conn_info:       Optional[Dict],
    name_candidates: List[RelationCandidate],
    sample_size:     int = 500,
) -> List[RelationCandidate]:
    results: List[RelationCandidate] = []
    if not conn_info or _get_db_type(conn_info) != "mssql":
        return results

    candidates = [c for c in name_candidates if c.confidence >= 0.65]
    if not candidates:
        return results

    try:
        import pyodbc
        import asyncio as _asyncio
        dsn = _build_dsn_mssql(conn_info)

        def _fetch(queries_and_keys):
            out = {}
            with pyodbc.connect(dsn, timeout=15) as mc:
                cur = mc.cursor()
                for key, q1, q2, q3 in queries_and_keys:
                    try:
                        cur.execute(q1)
                        row = cur.fetchone()
                        if not row or row[0] == 0:
                            continue
                        total, non_null, card_fk = row
                        cur.execute(q2)
                        r2      = cur.fetchone()
                        card_pk = r2[0] if r2 else 0
                        cur.execute(q3)
                        r3      = cur.fetchone()
                        covered = r3[0] if r3 else 0
                        out[key] = (total, non_null, card_fk, card_pk, covered)
                    except Exception as e:
                        logger.debug(f"[value_based] {key}: {e}")
            return out

        queries = []
        for cand in candidates:
            st, sc = cand.source_entity, cand.source_field
            tt, tc = cand.target_entity,  cand.target_field
            key = f"{st}.{sc}>{tt}"
            q1 = (f"SELECT COUNT(*), COUNT([{sc}]), COUNT(DISTINCT [{sc}]) "
                  f"FROM (SELECT TOP {sample_size} [{sc}] FROM [{st}]) s")
            q2 = (f"SELECT COUNT(DISTINCT [{tc}]) "
                  f"FROM (SELECT TOP {sample_size} [{tc}] FROM [{tt}]) t")
            q3 = (f"SELECT COUNT(DISTINCT s.[{sc}]) "
                  f"FROM (SELECT TOP {sample_size} [{sc}] FROM [{st}] "
                  f"WHERE [{sc}] IS NOT NULL) s "
                  f"WHERE EXISTS (SELECT 1 FROM [{tt}] t WHERE t.[{tc}] = s.[{sc}])")
            queries.append((key, q1, q2, q3))

        loop      = _asyncio.get_event_loop()
        stats_map = await loop.run_in_executor(None, _fetch, queries)

        for cand in candidates:
            key = f"{cand.source_entity}.{cand.source_field}>{cand.target_entity}"
            if key not in stats_map:
                continue
            total, non_null, card_fk, card_pk, covered = stats_map[key]
            null_ratio = 1.0 - (non_null / max(total, 1))
            coverage   = covered / max(card_fk, 1)
            v_score    = coverage * 0.7 + (1 - min(null_ratio, 0.5) * 2) * 0.15
            if 0.01 <= card_fk / max(card_pk, 1) <= 1.5:
                v_score += 0.15
            v_score   = max(0.0, min(v_score, 1.0))
            final_scr = round(cand.confidence * 0.45 + v_score * 0.55, 3)
            results.append(RelationCandidate(
                source_entity=cand.source_entity, source_field=cand.source_field,
                target_entity=cand.target_entity, target_field=cand.target_field,
                confidence=final_scr, detection_method="value_based",
                features={
                    **cand.features,
                    "coverage":   round(coverage, 3),
                    "null_ratio": round(null_ratio, 3),
                    "v_score":    round(v_score, 3),
                }))
    except Exception as e:
        logger.warning(f"[value_based] {e}")

    logger.info(f"[value_based] {len(results)} candidats valides")
    return results


# ══════════════════════════════════════════════════════════════════════
# ALGO 4 — FUZZY MATCH (Levenshtein)
# ══════════════════════════════════════════════════════════════════════

def _algo_fuzzy_match(
    entity_index:     Dict[str, str],
    fields_by_entity: Dict[str, List[Dict]],
    max_dist:         int = 2,
    min_len:          int = 4,
) -> List[RelationCandidate]:
    """
    Pour chaque champ potentiel-FK, compare son 'base' à tous les noms d'entités
    via distance Levenshtein. Cible les cas où name_based échoue (typos, abréviations).
    Confidence : dist=1 → 0.74 | dist=2 → 0.62
    """
    results: List[RelationCandidate] = []
    seen         = set()
    entity_names = list(entity_index.values())

    for src_name, src_fields in fields_by_entity.items():
        for f in src_fields:
            if f.get("is_primary_key") and not f.get("is_foreign_key"):
                continue
            field_words = _to_words(f["name"])
            is_fk, base = _is_fk_field(field_words)
            if not is_fk or len(base) < min_len:
                continue

            base_norm = _normalize(base)

            for tgt_name in entity_names:
                if tgt_name == src_name:
                    continue
                tgt_norm = _normalize(tgt_name)
                dist     = _levenshtein(base_norm, tgt_norm)
                if dist == 0 or dist > max_dist:
                    continue
                if len(base_norm) < min_len or len(tgt_norm) < min_len:
                    continue

                tgt_pk = _get_pk_field(tgt_name, fields_by_entity)
                if not tgt_pk:
                    continue

                key = f"{src_name}.{f['name']}>{tgt_name}"
                if key in seen:
                    continue
                seen.add(key)

                conf = round(0.74 - (dist - 1) * 0.12, 3)

                results.append(RelationCandidate(
                    source_entity=src_name,  source_field=f["name"],
                    target_entity=tgt_name,  target_field=tgt_pk,
                    confidence=conf,         detection_method="fuzzy_match",
                    features={"base": base, "lev_dist": dist}))

    logger.info(f"[fuzzy_match] {len(results)} candidats")
    return results


# ══════════════════════════════════════════════════════════════════════
# MOTEUR DE FUSION
# ══════════════════════════════════════════════════════════════════════

ALGO_PRIORITY: Dict[str, int] = {
    "explicit_fk": 0,
    "value_based": 1,
    "name_m2m":    2,
    "name_pascal": 3,
    "name_snake":  4,
    "fuzzy_match": 5,
}


def _merge_candidates(
    *candidate_lists,
    min_confidence: float = 0.55,
) -> List[RelationCandidate]:
    best: Dict[str, RelationCandidate] = {}
    value_coverage: Dict[str, float]   = {}

    # Passe 1 — mémoriser le coverage value_based par clé
    for cands in candidate_lists:
        for c in cands:
            if c.detection_method == "value_based" and "coverage" in c.features:
                key = f"{c.source_entity}.{c.source_field}>{c.target_entity}"
                value_coverage[key] = c.features["coverage"]

    # Passe 2 — fusion avec priorité algo
    for cands in candidate_lists:
        for c in cands:
            if c.confidence < min_confidence:
                continue
            key = f"{c.source_entity}.{c.source_field}>{c.target_entity}"
            if key not in best:
                best[key] = c
            else:
                ex = best[key]
                pn = ALGO_PRIORITY.get(c.detection_method, 99)
                po = ALGO_PRIORITY.get(ex.detection_method, 99)
                if pn < po or (pn == po and c.confidence > ex.confidence):
                    if "coverage" not in c.features and key in value_coverage:
                        c.features["coverage"] = value_coverage[key]
                    best[key] = c

    merged = sorted(
        best.values(),
        key=lambda x: (-x.confidence, x.source_entity, x.source_field)
    )
    logger.info(f"[merge] {len(merged)} relations uniques après fusion")
    return merged


# ══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

async def discover_relationships(source_id: UUID) -> Dict[str, Any]:
    """Alias public utilisé par main.py."""
    return await discover_relations(source_id)


async def discover_relations(source_id: UUID) -> Dict[str, Any]:
    from .database import get_pg_pool
    pool = await get_pg_pool()

    async with pool.acquire() as conn:
        entity_rows = await conn.fetch(
            "SELECT id, name FROM source_entities WHERE source_id=$1 ORDER BY name",
            source_id)
        if not entity_rows:
            return {
                "success": True, "relations_found": 0,
                "message": "Aucune entité. Lancez d'abord une synchronisation.",
            }

        entity_ids = [r["id"] for r in entity_rows]
        field_rows = await conn.fetch(
            """SELECT ef.entity_id, ef.name, ef.is_primary_key, ef.is_foreign_key
               FROM entity_fields ef WHERE ef.entity_id = ANY($1)
               ORDER BY ef.entity_id, ef.position""",
            entity_ids)

        source_row = await conn.fetchrow(
            """SELECT connector_type, host, port, database_name,
                      username, password, options
               FROM data_sources WHERE id=$1""",
            source_id)

        # ── Index entités ──────────────────────────────────────
        entity_index:     Dict[str, str]        = {}
        entity_names_set: set                   = set()
        fields_by_entity: Dict[str, List[Dict]] = {}
        id_to_name:       Dict[str, str]        = {}

        for er in entity_rows:
            eid  = str(er["id"])
            name = er["name"]
            id_to_name[eid]                = name
            entity_names_set.add(name)
            entity_index[name.lower()]     = name
            entity_index[_normalize(name)] = name
            fields_by_entity[name]         = []

        for fr in field_rows:
            eid  = str(fr["entity_id"])
            name = id_to_name.get(eid)
            if name:
                fields_by_entity[name].append({
                    "name":           fr["name"],
                    "is_primary_key": fr["is_primary_key"],
                    "is_foreign_key": fr["is_foreign_key"],
                })

        # ── conn_info (colonnes directes + fallback options) ───
        conn_info = None
        if source_row:
            import json as _json
            opts = source_row["options"] or {}
            if isinstance(opts, str):
                try:
                    opts = _json.loads(opts)
                except Exception:
                    opts = {}
            conn_info = {
                "connector_type": source_row["connector_type"] or "",
                "source_type":    source_row["connector_type"] or "",
                "host":           source_row["host"]          or opts.get("host", ""),
                "port":           source_row["port"]          or opts.get("port", 1433),
                "database_name":  source_row["database_name"] or opts.get("database_name", ""),
                "username":       source_row["username"]      or opts.get("username", ""),
                "password":       source_row["password"]      or opts.get("password", ""),
            }

    # ── Lancement des 4 algos ──────────────────────────────────
    name_cands     = _algo_name_based(entity_index, fields_by_entity)
    fuzzy_cands    = _algo_fuzzy_match(entity_index, fields_by_entity)
    explicit_cands = await _algo_explicit_fk(entity_names_set, conn_info)
    value_cands    = await _algo_value_based(conn_info, name_cands)

    final = _merge_candidates(
        explicit_cands, value_cands, name_cands, fuzzy_cands,
        min_confidence=0.55
    )

    stats: Dict[str, int] = {}
    for c in final:
        stats[c.detection_method] = stats.get(c.detection_method, 0) + 1

    # ── Persistance ───────────────────────────────────────────
    async with pool.acquire() as conn:
        # Supprimer les non-confirmées avant re-insert
        await conn.execute(
            """DELETE FROM entity_relations
               WHERE source_id=$1
               AND (is_confirmed = FALSE OR is_confirmed IS NULL)""",
            source_id)

        saved = 0
        for rel in final:
            try:
                coverage = rel.features.get("coverage")
                await conn.execute(
                    """INSERT INTO entity_relations
                           (source_id, source_entity, source_field,
                            target_entity, target_field,
                            relation_type, confidence, detection_method,
                            is_confirmed, value_overlap)
                       VALUES ($1,$2,$3,$4,$5,'many_to_one',$6,$7,FALSE,$8)
                       ON CONFLICT (source_id, source_entity, source_field, target_entity)
                       DO UPDATE SET
                           confidence       = $6,
                           detection_method = $7,
                           value_overlap    = COALESCE($8, entity_relations.value_overlap)""",
                    source_id,
                    rel.source_entity, rel.source_field,
                    rel.target_entity, rel.target_field,
                    rel.confidence,    rel.detection_method,
                    coverage,
                )
                saved += 1
            except Exception as e:
                logger.warning(f"[discover] INSERT error: {e}")

    logger.info(f"[discover] {saved} relations | {stats}")
    return {
        "success":         True,
        "relations_found": saved,
        "stats_by_algo":   stats,
        "message": (
            f"{saved} relation(s) — "
            + " | ".join(f"{m}: {n}" for m, n in sorted(stats.items()))
        ),
    }


# ══════════════════════════════════════════════════════════════════════
# VALIDATION MANUELLE
# ══════════════════════════════════════════════════════════════════════

async def validate_relation(
    relation_id:   int,
    confirmed:     bool,
    validated_by:  str           = "expert",
    reject_reason: Optional[str] = None,
) -> bool:
    from .database import get_pg_pool
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """UPDATE entity_relations
               SET is_confirmed  = $1,
                   validated_by  = $2,
                   reject_reason = $3,
                   validated_at  = NOW()
               WHERE id = $4
               RETURNING id""",
            confirmed, validated_by, reject_reason, int(relation_id))
        return row is not None


# ══════════════════════════════════════════════════════════════════════
# LECTURE
# ══════════════════════════════════════════════════════════════════════

async def get_relations_for_source(source_id: UUID) -> List[Dict]:
    from .database import get_pg_pool
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, source_entity, source_field,
                      target_entity, target_field,
                      relation_type, confidence,
                      detection_method, is_confirmed,
                      value_overlap, reject_reason
               FROM entity_relations
               WHERE source_id=$1
               ORDER BY confidence DESC, source_entity, source_field""",
            source_id)
    return [
        {
            "id":               r["id"],
            "source_entity":    r["source_entity"],
            "source_field":     r["source_field"],
            "target_entity":    r["target_entity"] or "",
            "target_field":     r["target_field"]  or "",
            "relation_type":    r["relation_type"],
            "confidence":       round(float(r["confidence"] or 0), 3),
            "detection_method": r["detection_method"],
            "is_confirmed":     r["is_confirmed"],
            "reject_reason":    r["reject_reason"],
            "value_overlap":    round(float(r["value_overlap"]), 3) if r["value_overlap"] is not None else None,
        }
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════════
# STATISTIQUES DE VALIDATION
# ══════════════════════════════════════════════════════════════════════

async def get_validation_stats(source_id: UUID) -> Dict[str, Any]:
    from .database import get_pg_pool
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                COUNT(*)                                              AS total,
                COUNT(*) FILTER (WHERE is_confirmed=TRUE)            AS confirmed,
                COUNT(*) FILTER (WHERE is_confirmed=FALSE
                                   AND reject_reason IS NOT NULL)    AS rejected,
                COUNT(*) FILTER (WHERE is_confirmed IS NULL
                                    OR (is_confirmed=FALSE
                                        AND reject_reason IS NULL))  AS pending,
                ROUND(AVG(confidence)::numeric, 3)                   AS avg_conf,
                ROUND(AVG(value_overlap)::numeric, 3)                AS avg_overlap,
                COUNT(*) FILTER (WHERE detection_method='explicit_fk')       AS n_explicit,
                COUNT(*) FILTER (WHERE detection_method LIKE 'name%')        AS n_name,
                COUNT(*) FILTER (WHERE detection_method='value_based')       AS n_value,
                COUNT(*) FILTER (WHERE detection_method='fuzzy_match')       AS n_fuzzy,
                COUNT(*) FILTER (WHERE detection_method='ml_predicted')      AS n_ml,
                COUNT(*) FILTER (WHERE detection_method='view_join')         AS n_view
            FROM entity_relations WHERE source_id=$1
        """, source_id)

        if not row or not row["total"]:
            return {
                "total": 0, "confirmed": 0, "rejected": 0, "pending": 0,
                "accuracy_est": 0.0, "avg_confidence": 0.0, "avg_overlap": 0.0,
                "by_method": {},
            }

        confirmed = row["confirmed"] or 0
        rejected  = row["rejected"]  or 0
        return {
            "total":          row["total"],
            "confirmed":      confirmed,
            "rejected":       rejected,
            "pending":        row["pending"] or 0,
            "accuracy_est":   round(confirmed / max(confirmed + rejected, 1) * 100, 1),
            "avg_confidence": float(row["avg_conf"]    or 0),
            "avg_overlap":    float(row["avg_overlap"] or 0),
            "by_method": {
                "explicit_fk":  row["n_explicit"] or 0,
                "name_based":   row["n_name"]     or 0,
                "value_based":  row["n_value"]    or 0,
                "fuzzy_match":  row["n_fuzzy"]    or 0,
                "ml_predicted": row["n_ml"]       or 0,
                "view_join":    row["n_view"]      or 0,
            },
        }


# ══════════════════════════════════════════════════════════════════════
# DIALECTES SQL — génération multi-connecteur
# ══════════════════════════════════════════════════════════════════════

def _get_sql_dialect(connector_type: str) -> str:
    ct = (connector_type or "").lower()
    if any(x in ct for x in ["mssql", "sqlserver", "sql_server", "mssql_odbc"]):
        return "mssql"
    if any(x in ct for x in ["postgres", "postgresql", "pg"]):
        return "postgresql"
    if any(x in ct for x in ["mysql", "mariadb"]):
        return "mysql"
    if any(x in ct for x in ["oracle"]):
        return "oracle"
    if any(x in ct for x in ["sqlite"]):
        return "sqlite"
    if any(x in ct for x in ["odata", "rest", "api", "http"]):
        return "odata"
    if any(x in ct for x in ["csv", "excel", "file", "json"]):
        return "pandas"
    return "mssql"


def _quote_table(name: str, dialect: str) -> str:
    if dialect == "mssql":
        return f"[{name}]"
    if dialect in ("postgresql", "mysql", "sqlite"):
        return f'"{name}"'
    if dialect == "oracle":
        return f'"{name.upper()}"'
    return name


def _build_sql(path: List[str], edges: List[Dict], dialect: str) -> str:
    q = lambda n: _quote_table(n, dialect)

    if dialect == "mssql":
        lines = ["SELECT TOP 100 *", f"FROM {q(path[0])}"]
        for e in edges:
            lines.append(
                f"JOIN {q(e['to'])} ON {q(e['from'])}.{q(e['via'])} = {q(e['to'])}.{q(e['tgt_field'])}")
        return "\n".join(lines)

    elif dialect == "postgresql":
        lines = ["SELECT *", f"FROM {q(path[0])}"]
        for e in edges:
            lines.append(
                f"JOIN {q(e['to'])} ON {q(e['from'])}.{q(e['via'])} = {q(e['to'])}.{q(e['tgt_field'])}")
        lines.append("LIMIT 100;")
        return "\n".join(lines)

    elif dialect == "mysql":
        lines = ["SELECT *", f"FROM {q(path[0])}"]
        for e in edges:
            lines.append(
                f"JOIN {q(e['to'])} ON {q(e['from'])}.{q(e['via'])} = {q(e['to'])}.{q(e['tgt_field'])}")
        lines.append("LIMIT 100;")
        return "\n".join(lines)

    elif dialect == "oracle":
        lines = ["SELECT *", f"FROM {q(path[0])}"]
        for e in edges:
            lines.append(
                f"JOIN {q(e['to'])} ON {q(e['from'])}.{q(e['via'])} = {q(e['to'])}.{q(e['tgt_field'])}")
        lines.append("FETCH FIRST 100 ROWS ONLY;")
        return "\n".join(lines)

    elif dialect == "sqlite":
        lines = ["SELECT *", f"FROM {q(path[0])}"]
        for e in edges:
            lines.append(
                f"JOIN {q(e['to'])} ON {q(e['from'])}.{q(e['via'])} = {q(e['to'])}.{q(e['tgt_field'])}")
        lines.append("LIMIT 100;")
        return "\n".join(lines)

    elif dialect == "odata":
        expands = [e["to"] for e in edges]
        return f"GET /{path[0]}?$expand={','.join(expands)}&$top=100"

    elif dialect == "pandas":
        lines = ["import pandas as pd", "# Charger les DataFrames"]
        for t in path:
            lines.append(f"df_{t.lower()} = pd.read_csv('{t}.csv')")
        lines.append("")
        lines.append("# Jointures")
        lines.append(f"result = df_{path[0].lower()}")
        for e in edges:
            lines.append(
                f"result = result.merge(df_{e['to'].lower()}, "
                f"left_on='{e['via']}', right_on='{e['tgt_field']}', how='inner')"
            )
        lines.append("result = result.head(100)")
        return "\n".join(lines)

    else:
        lines = ["SELECT *", f"FROM {path[0]}"]
        for e in edges:
            lines.append(f"JOIN {e['to']} ON {e['from']}.{e['via']} = {e['to']}.{e['tgt_field']}")
        lines.append("-- LIMIT 100")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# GRAPHE BFS BIDIRECTIONNEL + JOIN PATHS
# ══════════════════════════════════════════════════════════════════════

def _build_join_graph(relations: List[Dict]) -> Dict[str, List[Tuple]]:
    """Graphe BIDIRECTIONNEL : chaque FK navigable dans les 2 sens."""
    graph: Dict[str, List] = defaultdict(list)
    seen = set()
    for r in relations:
        src  = r.get("source_entity", "")
        tgt  = r.get("target_entity", "")
        sf   = r.get("source_field",  "")
        tf   = r.get("target_field",  "")
        conf = r.get("confidence", 0.5)
        if not src or not tgt:
            continue
        # Sens normal : src → tgt
        key_fwd = (src, sf, tgt, tf)
        if key_fwd not in seen:
            seen.add(key_fwd)
            graph[src].append((tgt, sf, tf, conf))
        # Sens inverse : tgt → src
        key_rev = (tgt, tf, src, sf)
        if key_rev not in seen:
            seen.add(key_rev)
            graph[tgt].append((src, tf, sf, conf))
    return dict(graph)


def get_join_paths(
    graph:     Dict,
    start:     str,
    end:       str,
    max_depth: int = 3,
    dialect:   str = "mssql",
) -> List[Dict]:
    """BFS bidirectionnel — trouve tous les chemins entre start et end."""
    start_up = start.upper()
    end_up   = end.upper()

    # Rebuild graph insensible à la casse
    graph_up: Dict[str, List] = defaultdict(list)
    for k, vlist in graph.items():
        for (nb, sf, tf, c) in vlist:
            graph_up[k.upper()].append((nb.upper(), sf, tf, c))

    if start_up == end_up or not start_up or not end_up:
        return []
    if start_up not in graph_up and end_up not in graph_up:
        return []

    queue       = deque([(start_up, [start_up], [], 1.0)])
    paths       = []
    seen_paths  = set()
    MAX_RESULTS = 20
    MAX_VISITED = 50_000
    visited     = 0

    while queue and len(paths) < MAX_RESULTS and visited < MAX_VISITED:
        node, path, edges, path_conf = queue.popleft()
        visited += 1
        if len(path) - 1 >= max_depth:
            continue
        for (neighbor, via_fk, via_pk, conf) in graph_up.get(node, []):
            if neighbor in path:
                continue
            new_conf  = round(path_conf * conf, 3)
            new_path  = path + [neighbor]
            new_edges = edges + [{
                "from":      node,
                "via":       via_fk,
                "to":        neighbor,
                "tgt_field": via_pk,
                "conf":      conf,
            }]
            if neighbor == end_up:
                path_key = "->".join(new_path)
                if path_key not in seen_paths:
                    seen_paths.add(path_key)
                    paths.append({
                        "path":       new_path,
                        "edges":      new_edges,
                        "length":     len(new_path) - 1,
                        "confidence": new_conf,
                        "joins_sql":  _build_sql(new_path, new_edges, dialect),
                        "dialect":    dialect,
                    })
                    if len(paths) >= MAX_RESULTS:
                        break
            else:
                queue.append((neighbor, new_path, new_edges, new_conf))

    paths.sort(key=lambda p: (p["length"], -p["confidence"]))
    return paths[:MAX_RESULTS]


async def get_join_paths_for_source(
    source_id:  UUID,
    from_table: str,
    to_table:   str,
    max_depth:  int = 3,
) -> List[Dict]:
    from .database import get_pg_pool
    pool = await get_pg_pool()

    dialect = "mssql"
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT connector_type FROM data_sources WHERE id=$1", source_id)
            if row and row["connector_type"]:
                dialect = _get_sql_dialect(row["connector_type"])
    except Exception:
        pass

    relations = await get_relations_for_source(source_id)
    graph     = _build_join_graph(relations)
    return get_join_paths(graph, from_table, to_table, max_depth, dialect)