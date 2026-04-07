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
    """
    Décompose un nom en tokens normalisés.
    PATCH P1 : Stemming Snowball optionnel (nltk) — graceful fallback si absent.
    Ex: 'customers' et 'customer' → même stem, améliore le matching name-based.
    """
    s = name.replace("_", " ").replace("-", " ")
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    words = [w.lower() for w in s.split() if w]
    try:
        from nltk.stem import SnowballStemmer  # type: ignore
        stemmer = SnowballStemmer("english")
        return [stemmer.stem(w) for w in words]
    except ImportError:
        return words


def _to_words_raw(name: str) -> List[str]:
    """Version sans stemming — pour l'affichage et les logs."""
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
        import pyodbc  # type: ignore[import]
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
            import pyodbc  # type: ignore[import]
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
            import pyodbc  # type: ignore[import]
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
    if not conn_info:
        return results
    db_type = _get_db_type(conn_info)
    if db_type not in ("mssql", "mysql", "postgresql"):
        return results

    candidates = [c for c in name_candidates if c.confidence >= 0.65]
    if not candidates:
        return results

    try:
        import pyodbc  # type: ignore[import]
        import asyncio as _asyncio
        # ── Construction des requêtes selon le dialecte SQL ──────────────
        def _quote(identifier: str, dialect: str) -> str:
            """Quote un identifiant selon le dialecte."""
            if dialect == "mssql":
                return f"[{identifier}]"
            else:  # postgresql, mysql
                return f'"{identifier}"'

        def _limit_clause(n: int, dialect: str) -> str:
            """Clause de limitation selon le dialecte."""
            if dialect == "mssql":
                return f"TOP {n}"
            return ""  # LIMIT en fin de requête pour PG/MySQL

        def _limit_suffix(n: int, dialect: str) -> str:
            if dialect in ("postgresql", "mysql"):
                return f" LIMIT {n}"
            return ""

        def _build_queries(cands, dialect, n):
            queries = []
            for cand in cands:
                st, sc = cand.source_entity, cand.source_field
                tt, tc = cand.target_entity,  cand.target_field
                key    = f"{st}.{sc}>{tt}"
                sc_q   = _quote(sc, dialect)
                tc_q   = _quote(tc, dialect)
                st_q   = _quote(st, dialect)
                tt_q   = _quote(tt, dialect)
                top    = _limit_clause(n, dialect)
                lim    = _limit_suffix(n, dialect)

                if dialect == "mssql":
                    q1 = (f"SELECT COUNT(*), COUNT({sc_q}), COUNT(DISTINCT {sc_q}) "
                          f"FROM (SELECT {top} {sc_q} FROM {st_q}) s")
                    q2 = (f"SELECT COUNT(DISTINCT {tc_q}) "
                          f"FROM (SELECT {top} {tc_q} FROM {tt_q}) t")
                    q3 = (f"SELECT COUNT(DISTINCT s.{sc_q}) "
                          f"FROM (SELECT {top} {sc_q} FROM {st_q} "
                          f"WHERE {sc_q} IS NOT NULL) s "
                          f"WHERE EXISTS (SELECT 1 FROM {tt_q} t WHERE t.{tc_q} = s.{sc_q})")
                else:  # postgresql / mysql
                    q1 = (f"SELECT COUNT(*), COUNT({sc_q}), COUNT(DISTINCT {sc_q}) "
                          f"FROM (SELECT {sc_q} FROM {st_q}{lim}) s")
                    q2 = (f"SELECT COUNT(DISTINCT {tc_q}) "
                          f"FROM (SELECT {tc_q} FROM {tt_q}{lim}) t")
                    q3 = (f"SELECT COUNT(DISTINCT s.{sc_q}) "
                          f"FROM (SELECT {sc_q} FROM {st_q} "
                          f"WHERE {sc_q} IS NOT NULL{lim}) s "
                          f"WHERE EXISTS (SELECT 1 FROM {tt_q} t WHERE t.{tc_q} = s.{sc_q})")
                queries.append((key, q1, q2, q3))
            return queries

        # ── Connexion et exécution selon le dialecte ──────────────────────
        if db_type == "mssql":
            dsn = _build_dsn_mssql(conn_info)
            def _fetch_mssql(queries_and_keys):
                out = {}
                with pyodbc.connect(dsn, timeout=15) as mc:
                    cur = mc.cursor()
                    for key, q1, q2, q3 in queries_and_keys:
                        try:
                            cur.execute(q1); row = cur.fetchone()
                            if not row or row[0] == 0: continue
                            total, non_null, card_fk = row
                            cur.execute(q2); r2 = cur.fetchone()
                            card_pk = r2[0] if r2 else 0
                            cur.execute(q3); r3 = cur.fetchone()
                            covered = r3[0] if r3 else 0
                            out[key] = (total, non_null, card_fk, card_pk, covered)
                        except Exception as e:
                            logger.debug(f"[value_based] {key}: {e}")
                return out

            queries    = _build_queries(candidates, "mssql", sample_size)
            loop       = _asyncio.get_event_loop()
            stats_map  = await loop.run_in_executor(None, _fetch_mssql, queries)

        elif db_type == "mysql":
            dsn = _build_dsn_mysql(conn_info)
            def _fetch_mysql(queries_and_keys):
                out = {}
                try:
                    with pyodbc.connect(dsn, timeout=15) as mc:
                        cur = mc.cursor()
                        for key, q1, q2, q3 in queries_and_keys:
                            try:
                                cur.execute(q1); row = cur.fetchone()
                                if not row or row[0] == 0: continue
                                total, non_null, card_fk = row
                                cur.execute(q2); r2 = cur.fetchone()
                                card_pk = r2[0] if r2 else 0
                                cur.execute(q3); r3 = cur.fetchone()
                                covered = r3[0] if r3 else 0
                                out[key] = (total, non_null, card_fk, card_pk, covered)
                            except Exception as e:
                                logger.debug(f"[value_based mysql] {key}: {e}")
                except Exception as e:
                    logger.warning(f"[value_based mysql] connexion: {e}")
                return out

            queries   = _build_queries(candidates, "mysql", sample_size)
            loop      = _asyncio.get_event_loop()
            stats_map = await loop.run_in_executor(None, _fetch_mysql, queries)

        elif db_type == "postgresql":
            # PostgreSQL — asyncpg natif (pas de pyodbc nécessaire)
            from .database import get_pg_pool as _get_pg_pool
            h   = conn_info.get("host", "localhost")
            p   = conn_info.get("port", 5432)
            d   = conn_info.get("database_name") or conn_info.get("database", "")
            u   = conn_info.get("username", "")
            pw  = conn_info.get("password", "")
            schema = conn_info.get("schema_name", "public") or "public"

            async def _fetch_pg(queries_and_keys):
                out = {}
                try:
                    import asyncpg  # type: ignore
                    pg_conn = await asyncpg.connect(
                        host=h, port=int(p), database=d,
                        user=u, password=pw, timeout=15
                    )
                    try:
                        for key, q1, q2, q3 in queries_and_keys:
                            try:
                                row = await pg_conn.fetchrow(q1)
                                if not row or row[0] == 0: continue
                                total, non_null, card_fk = row[0], row[1], row[2]
                                r2 = await pg_conn.fetchrow(q2)
                                card_pk = r2[0] if r2 else 0
                                r3 = await pg_conn.fetchrow(q3)
                                covered = r3[0] if r3 else 0
                                out[key] = (total, non_null, card_fk, card_pk, covered)
                            except Exception as e:
                                logger.debug(f"[value_based pg] {key}: {e}")
                    finally:
                        await pg_conn.close()
                except Exception as e:
                    logger.warning(f"[value_based pg] connexion: {e}")
                return out

            queries   = _build_queries(candidates, "postgresql", sample_size)
            stats_map = await _fetch_pg(queries)

        else:
            stats_map = {}

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
            # PATCH P2 — bonus statistique (Chi²/KS + pattern matching)
            stat_bonus = 0.0
            stat_features: dict = {}
            try:
                from scipy import stats as _scipy_stats  # type: ignore
                import re as _re2
                sc2 = cand.source_field; st2 = cand.source_entity
                tc2 = cand.target_field; tt2 = cand.target_entity
                _vals_a, _vals_b = [], []
                if db_type == "mssql":
                    import pyodbc as _pyodbc  # type: ignore
                    with _pyodbc.connect(dsn, timeout=10) as _mc:
                        with _mc.cursor() as _cur:
                            _cur.execute(f"SELECT TOP 200 [{sc2}] FROM [{st2}] WHERE [{sc2}] IS NOT NULL")
                            _vals_a = [r[0] for r in _cur.fetchall()]
                            _cur.execute(f"SELECT TOP 200 [{tc2}] FROM [{tt2}] WHERE [{tc2}] IS NOT NULL")
                            _vals_b = [r[0] for r in _cur.fetchall()]
                elif db_type == "mysql":
                    import pyodbc as _pyodbc  # type: ignore
                    with _pyodbc.connect(_build_dsn_mysql(conn_info), timeout=10) as _mc:
                        with _mc.cursor() as _cur:
                            _cur.execute(f'SELECT `{sc2}` FROM `{st2}` WHERE `{sc2}` IS NOT NULL LIMIT 200')
                            _vals_a = [r[0] for r in _cur.fetchall()]
                            _cur.execute(f'SELECT `{tc2}` FROM `{tt2}` WHERE `{tc2}` IS NOT NULL LIMIT 200')
                            _vals_b = [r[0] for r in _cur.fetchall()]
                elif db_type == "postgresql":
                    import asyncpg as _asyncpg  # type: ignore
                    import asyncio as _asyncio2
                    async def _pg_stat():
                        _c = await _asyncpg.connect(
                            host=conn_info.get("host"), port=conn_info.get("port",5432),
                            database=conn_info.get("database_name",""), 
                            user=conn_info.get("username",""), password=conn_info.get("password",""),
                            timeout=10)
                        try:
                            _ra = await _c.fetch(f'SELECT "{sc2}" FROM "{st2}" WHERE "{sc2}" IS NOT NULL LIMIT 200')
                            _rb = await _c.fetch(f'SELECT "{tc2}" FROM "{tt2}" WHERE "{tc2}" IS NOT NULL LIMIT 200')
                            return [r[0] for r in _ra], [r[0] for r in _rb]
                        finally:
                            await _c.close()
                    _vals_a, _vals_b = await _pg_stat()
                UUID_RE2 = _re2.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", _re2.I)
                NUM_RE2  = _re2.compile(r"^\d+$")
                def _pat(vals):
                    s = [str(v) for v in vals[:30]]
                    if not s: return "mixed"
                    if sum(1 for v in s if UUID_RE2.match(v)) / len(s) > 0.7: return "uuid"
                    if sum(1 for v in s if NUM_RE2.match(v)) / len(s) > 0.7: return "numeric"
                    return "mixed"
                pat_a, pat_b = _pat(_vals_a), _pat(_vals_b)
                stat_features["pattern_match"] = float(pat_a == pat_b and pat_a != "mixed")
                if stat_features["pattern_match"] == 1.0:
                    stat_bonus += 0.03
                try:
                    fa_num = [float(v) for v in _vals_a if v is not None]
                    fb_num = [float(v) for v in _vals_b if v is not None]
                    if len(fa_num) >= 10 and len(fb_num) >= 10:
                        ks_stat, ks_p = _scipy_stats.ks_2samp(fa_num, fb_num)
                        stat_features["ks_statistic"] = round(float(ks_stat), 4)
                        stat_features["ks_pvalue"]    = round(float(ks_p), 4)
                        if ks_stat < 0.3:
                            stat_bonus += 0.02
                except (ValueError, TypeError):
                    pass
            except Exception:
                pass
            final_scr = min(round(final_scr + stat_bonus, 3), 0.99)
            results.append(RelationCandidate(
                source_entity=cand.source_entity, source_field=cand.source_field,
                target_entity=cand.target_entity, target_field=cand.target_field,
                confidence=final_scr, detection_method="value_based",
                features={
                    **cand.features,
                    "coverage":   round(coverage, 3),
                    "null_ratio": round(null_ratio, 3),
                    "v_score":    round(v_score, 3),
                    **stat_features,
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
# PATCH P3 — Heuristiques ERP spécifiques (SAP, Dynamics, Odoo, génériques)
# ══════════════════════════════════════════════════════════════════════

ERP_PATTERNS: List[Dict] = [
    # ── Génériques ERP ────────────────────────────────────────────────
    {"name": "Order→Customer",
     "src_tables": ["orders","sales","invoices","factures","commandes","ventes"],
     "tgt_tables": ["customers","clients","partners","tiers","partenaires"],
     "src_cols":   ["customer_id","client_id","customer_code","id_client","tiers_id"],
     "confidence": 0.88},
    {"name": "OrderLine→Order",
     "src_tables": ["order_lines","order_details","line_items","lignes_commande","sales_lines","invoice_lines"],
     "tgt_tables": ["orders","sales_orders","commandes","invoices"],
     "src_cols":   ["order_id","order_number","order_ref","commande_id","id_commande"],
     "confidence": 0.90},
    {"name": "Product→Category",
     "src_tables": ["products","items","articles","produits"],
     "tgt_tables": ["categories","product_families","familles","product_groups"],
     "src_cols":   ["category_id","family_id","type_id","famille_id","categorie_id"],
     "confidence": 0.85},
    {"name": "StockMove→Product",
     "src_tables": ["stock_moves","mouvements_stock","inventory_moves","stock_lines"],
     "tgt_tables": ["products","articles","items"],
     "src_cols":   ["product_id","article_id","item_id"],
     "confidence": 0.87},
    {"name": "Employee→Department",
     "src_tables": ["employees","personnel","staff","users","hr_employees"],
     "tgt_tables": ["departments","services","cost_centers","hr_departments"],
     "src_cols":   ["department_id","dept_id","service_id","cost_center_id"],
     "confidence": 0.85},
    # ── SAP ───────────────────────────────────────────────────────────
    {"name": "SAP: MARA→MAKT",
     "src_tables": ["mara"], "tgt_tables": ["makt"],
     "src_cols": ["matnr"], "confidence": 0.97},
    {"name": "SAP: VBAK→VBAP",
     "src_tables": ["vbak"], "tgt_tables": ["vbap"],
     "src_cols": ["vbeln"], "confidence": 0.97},
    {"name": "SAP: VBAP→MARA",
     "src_tables": ["vbap"], "tgt_tables": ["mara"],
     "src_cols": ["matnr"], "confidence": 0.95},
    {"name": "SAP: BKPF→BSEG",
     "src_tables": ["bkpf"], "tgt_tables": ["bseg"],
     "src_cols": ["belnr","bukrs","gjahr"], "confidence": 0.97},
    {"name": "SAP: LFA1→LFB1",
     "src_tables": ["lfa1"], "tgt_tables": ["lfb1"],
     "src_cols": ["lifnr"], "confidence": 0.95},
    {"name": "SAP: KNA1→KNB1",
     "src_tables": ["kna1"], "tgt_tables": ["knb1"],
     "src_cols": ["kunnr"], "confidence": 0.95},
    # ── Microsoft Dynamics 365 ────────────────────────────────────────
    {"name": "Dynamics: SalesOrder→Account",
     "src_tables": ["salesorder","salesorderdetail","quote","invoice"],
     "tgt_tables": ["account","contact","lead"],
     "src_cols":   ["customerid","accountid","regardingobjectid"],
     "confidence": 0.90},
    {"name": "Dynamics: SystemUser",
     "src_tables": [], "tgt_tables": ["systemuser","team"],
     "src_cols":   ["owninguser","owningteam","createdby","modifiedby","ownerid"],
     "confidence": 0.85},
    # ── Odoo ──────────────────────────────────────────────────────────
    {"name": "Odoo: res.partner",
     "src_tables": ["res_partner","sale_order","purchase_order","account_move",
                    "stock_picking","hr_employee","crm_lead"],
     "tgt_tables": ["res_partner"],
     "src_cols":   ["partner_id","customer_id","supplier_id","company_id"],
     "confidence": 0.92},
    {"name": "Odoo: product",
     "src_tables": ["sale_order_line","purchase_order_line","stock_move",
                    "account_move_line","mrp_production"],
     "tgt_tables": ["product_product","product_template"],
     "src_cols":   ["product_id","product_tmpl_id"],
     "confidence": 0.93},
    {"name": "Odoo: res.company",
     "src_tables": [], "tgt_tables": ["res_company"],
     "src_cols":   ["company_id"], "confidence": 0.88},
    {"name": "Odoo: account.account",
     "src_tables": ["account_move_line","account_journal"],
     "tgt_tables": ["account_account"],
     "src_cols":   ["account_id","debit_account_id","credit_account_id"],
     "confidence": 0.90},
    {"name": "Odoo: hr.employee",
     "src_tables": ["hr_leave","hr_payslip","hr_attendance","hr_expense"],
     "tgt_tables": ["hr_employee"],
     "src_cols":   ["employee_id"], "confidence": 0.92},    # ── Oracle E-Business Suite (EBS) ───────────────────────────────
    # Clients et commandes de vente
    {"name": "OracleEBS: RA_CUSTOMERS→HZ_PARTIES",
     "src_tables": ["ra_customers","hz_cust_accounts"],
     "tgt_tables": ["hz_parties"],
     "src_cols":   ["party_id","cust_account_id"],
     "confidence": 0.95},
    {"name": "OracleEBS: OE_ORDER_HEADERS→RA_CUSTOMERS",
     "src_tables": ["oe_order_headers_all","oe_order_lines_all"],
     "tgt_tables": ["ra_customers","hz_cust_accounts"],
     "src_cols":   ["sold_to_org_id","ship_to_org_id","invoice_to_org_id"],
     "confidence": 0.93},
    {"name": "OracleEBS: OE_ORDER_LINES→MTL_SYSTEM_ITEMS",
     "src_tables": ["oe_order_lines_all"],
     "tgt_tables": ["mtl_system_items_b","mtl_system_items"],
     "src_cols":   ["inventory_item_id","ordered_item_id"],
     "confidence": 0.94},
    {"name": "OracleEBS: AP_INVOICES→PO_HEADERS",
     "src_tables": ["ap_invoices_all","ap_invoice_lines_all"],
     "tgt_tables": ["po_headers_all","po_lines_all"],
     "src_cols":   ["po_header_id","po_line_id"],
     "confidence": 0.93},
    {"name": "OracleEBS: AP_INVOICES→AP_SUPPLIERS",
     "src_tables": ["ap_invoices_all","ap_invoice_distributions_all"],
     "tgt_tables": ["ap_suppliers","po_vendors"],
     "src_cols":   ["vendor_id","vendor_site_id"],
     "confidence": 0.95},
    {"name": "OracleEBS: GL_JE_LINES→GL_CODE_COMBINATIONS",
     "src_tables": ["gl_je_lines","gl_balances"],
     "tgt_tables": ["gl_code_combinations"],
     "src_cols":   ["code_combination_id","ccid"],
     "confidence": 0.97},
    {"name": "OracleEBS: HR_EMPLOYEES→PER_ALL_PEOPLE",
     "src_tables": ["per_all_assignments_f","pay_payroll_actions"],
     "tgt_tables": ["per_all_people_f","per_people_f"],
     "src_cols":   ["person_id","employee_id"],
     "confidence": 0.94},
    {"name": "OracleEBS: MTL_TRANSACTIONS→MTL_SYSTEM_ITEMS",
     "src_tables": ["mtl_material_transactions","mtl_onhand_quantities_detail"],
     "tgt_tables": ["mtl_system_items_b"],
     "src_cols":   ["inventory_item_id"],
     "confidence": 0.96},
    {"name": "OracleEBS: ORG_ID multi-org",
     "src_tables": [],
     "tgt_tables": ["hr_operating_units","hr_all_organization_units"],
     "src_cols":   ["org_id","operating_unit_id","business_group_id"],
     "confidence": 0.88},

    # ── NetSuite (Oracle NetSuite) ────────────────────────────────────
    {"name": "NetSuite: TRANSACTION→CUSTOMER",
     "src_tables": ["transaction","transactionline"],
     "tgt_tables": ["customer","entity"],
     "src_cols":   ["entity","customer","customerid"],
     "confidence": 0.91},
    {"name": "NetSuite: TRANSACTIONLINE→ITEM",
     "src_tables": ["transactionline"],
     "tgt_tables": ["item"],
     "src_cols":   ["item","itemid"],
     "confidence": 0.93},
    {"name": "NetSuite: TRANSACTION→SUBSIDIARY",
     "src_tables": ["transaction","employee","vendor"],
     "tgt_tables": ["subsidiary"],
     "src_cols":   ["subsidiary","subsidiaryid"],
     "confidence": 0.90},
    {"name": "NetSuite: VENDOR→VENDORBILL",
     "src_tables": ["vendorbill","vendorpayment","purchaseorder"],
     "tgt_tables": ["vendor","entity"],
     "src_cols":   ["entity","vendor","vendorid"],
     "confidence": 0.92},
    {"name": "NetSuite: EMPLOYEE→DEPARTMENT",
     "src_tables": ["employee","transaction"],
     "tgt_tables": ["department","classification"],
     "src_cols":   ["department","departmentid"],
     "confidence": 0.89},
    {"name": "NetSuite: ACCOUNT (GL)",
     "src_tables": ["transactionline","accountingline"],
     "tgt_tables": ["account"],
     "src_cols":   ["account","accountid","expenseaccount"],
     "confidence": 0.94},

    # ── SAGE X3 ───────────────────────────────────────────────────────
    {"name": "SAGE X3: SORDER→BPCUSTOMER",
     "src_tables": ["sorder","sinvoice","squote"],
     "tgt_tables": ["bpcustomer","bpartner"],
     "src_cols":   ["bpcord","bpcnam","bpcustomer"],
     "confidence": 0.92},
    {"name": "SAGE X3: SORDERQ→ITMMASTER",
     "src_tables": ["sorderq","sinvoiced","sdeliveryd"],
     "tgt_tables": ["itmmaster","itmfacilit"],
     "src_cols":   ["itmref","itemref"],
     "confidence": 0.94},
    {"name": "SAGE X3: PORDER→BPSUPPLIER",
     "src_tables": ["porder","pinvoice"],
     "tgt_tables": ["bpsupplier","bpartner"],
     "src_cols":   ["bpsord","bpsupplier"],
     "confidence": 0.92},
    {"name": "SAGE X3: GACCENTRY→GACCOUNT",
     "src_tables": ["gaccentry","gaccentryd"],
     "tgt_tables": ["gaccount","caccount"],
     "src_cols":   ["acc","account","gacc"],
     "confidence": 0.95},
    {"name": "SAGE X3: FACILITY (multi-site)",
     "src_tables": [],
     "tgt_tables": ["facility","fcymaster"],
     "src_cols":   ["fcy","facility","fcyref"],
     "confidence": 0.87},
    {"name": "SAGE X3: ITMMASTER→ITMFACILIT",
     "src_tables": ["itmmaster"],
     "tgt_tables": ["itmfacilit"],
     "src_cols":   ["itmref"],
     "confidence": 0.96},
    {"name": "SAGE X3: SDELIVERY→SORDER",
     "src_tables": ["sdelivery","sdeliveryd"],
     "tgt_tables": ["sorder"],
     "src_cols":   ["sohnum","sorder"],
     "confidence": 0.93},

    # ── Infor (LN, M3, CloudSuite) ───────────────────────────────────
    # Infor LN
    {"name": "Infor LN: baan_salesorder→baan_customer",
     "src_tables": ["oohed001","ooline001","ooshp001"],
     "tgt_tables": ["tccom100","bpcustomer"],
     "src_cols":   ["sold_to_bp","ship_to_bp","invoice_to_bp","cust_order"],
     "confidence": 0.91},
    {"name": "Infor LN: baan_item (tiitm)",
     "src_tables": ["oohed001","ooline001","whwmd200","pcsfc200"],
     "tgt_tables": ["tiitm001"],
     "src_cols":   ["item","item_code","item_no"],
     "confidence": 0.93},
    {"name": "Infor LN: baan_supplier→purchaseorder",
     "src_tables": ["purord","pdline","purrcv"],
     "tgt_tables": ["tccom100"],
     "src_cols":   ["supplier","buy_from_bp"],
     "confidence": 0.91},
    {"name": "Infor LN: baan_ledger (tfgld)",
     "src_tables": ["tfgld010","tfgld106"],
     "tgt_tables": ["tfgld001","tfgld008"],
     "src_cols":   ["ledger_account","account","dim_1","dim_2"],
     "confidence": 0.94},
    {"name": "Infor LN: company (tcemm)",
     "src_tables": [],
     "tgt_tables": ["tcemm100","tcemm124"],
     "src_cols":   ["company","enterprise_unit"],
     "confidence": 0.87},
    # Infor M3
    {"name": "Infor M3: OOHEAD→OCUSMA",
     "src_tables": ["oohead","ooline","oodetl"],
     "tgt_tables": ["ocusma","cidmas"],
     "src_cols":   ["oacuno","cuno","customer_number"],
     "confidence": 0.92},
    {"name": "Infor M3: MITMAS→MITMAH",
     "src_tables": ["ooline","mplind","pplpri"],
     "tgt_tables": ["mitmas","mitmah"],
     "src_cols":   ["itno","item_number","mmitno"],
     "confidence": 0.94},
    {"name": "Infor M3: FGLEDG→FCHACC",
     "src_tables": ["fgledg","fginsl"],
     "tgt_tables": ["fchacc","fchacp"],
     "src_cols":   ["ait1","ait2","aait1","account"],
     "confidence": 0.93},
    {"name": "Infor M3: MNDIVS (division/warehouse)",
     "src_tables": [],
     "tgt_tables": ["mndivs","mitwhl"],
     "src_cols":   ["divi","division","whlo","warehouse"],
     "confidence": 0.88},
]



def _algo_erp_heuristics(
    entity_index:     Dict[str, str],
    fields_by_entity: Dict[str, List[Dict]],
) -> List[RelationCandidate]:
    """
    PATCH P3 — Détection via patterns ERP spécifiques (SAP, Dynamics, Odoo, génériques).
    """
    results: List[RelationCandidate] = []
    seen: set = set()

    for _src_key, src_entity_name in entity_index.items():
        src_lower  = src_entity_name.lower()
        src_fields = fields_by_entity.get(src_entity_name, [])

        for pattern in ERP_PATTERNS:
            src_match = (
                not pattern["src_tables"]
                or any(s in src_lower for s in pattern["src_tables"])
                or any(src_lower == s for s in pattern["src_tables"])
            )
            if not src_match:
                continue

            for field in src_fields:
                fname_lower = field["name"].lower()
                col_match = any(
                    c == fname_lower or fname_lower.endswith(c) or fname_lower.startswith(c)
                    for c in pattern["src_cols"]
                )
                if not col_match:
                    continue

                for tgt_table_pattern in (pattern["tgt_tables"] or []):
                    tgt_name = None
                    for ek, ev in entity_index.items():
                        if ev.lower() == tgt_table_pattern or tgt_table_pattern in ev.lower():
                            tgt_name = ev
                            break
                    if not tgt_name or tgt_name == src_entity_name:
                        continue
                    tgt_pk = _get_pk_field(tgt_name, fields_by_entity)
                    if not tgt_pk:
                        tgt_f_list = fields_by_entity.get(tgt_name, [])
                        tgt_pk = next(
                            (f["name"] for f in tgt_f_list
                             if f["name"].lower() in ("id", tgt_name.lower() + "_id")),
                            None
                        )
                    if not tgt_pk:
                        continue
                    key = f"{src_entity_name}.{field['name']}>{tgt_name}"
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(RelationCandidate(
                        source_entity=src_entity_name,
                        source_field=field["name"],
                        target_entity=tgt_name,
                        target_field=tgt_pk,
                        confidence=pattern["confidence"],
                        detection_method=f"heuristic_erp",
                        features={"erp_pattern": pattern["name"]},
                    ))

    logger.info(f"[erp_heuristics] {len(results)} candidats")
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
    "heuristic_erp": 6,
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
# ML — ALGO IMPLICITE (notebook v6 intégré)
# ══════════════════════════════════════════════════════════════════════

import os, pickle, re as _re
from typing import Tuple

_ML_FEATURE_COLS = [
    'name_sim', 'norm_sim', 'entity_in_field', 'type_compat',
    'fk_pattern_a', 'fk_pattern_b', 'pk_fk_pair', 'common_parts',
    'name_contains_entity_a', 'name_contains_entity_b',
    'len_diff', 'prefix_match', 'suffix_match',
    'value_overlap', 'cardinality_ratio', 'null_rate_compat',
]

# v8 : 5 features supplémentaires
_ML_FEATURE_COLS_V2 = _ML_FEATURE_COLS + [
    'embed_sim', 'embed_entity_sim',
    'topo_distance', 'chi2_compatible', 'pattern_match',
]

# Valeurs neutres pour les features v8 (utilisées quand embeddings non dispo)
_ML_V8_DEFAULTS = {
    'embed_sim':        0.0,
    'embed_entity_sim': 0.0,
    'topo_distance':    0.0,
    'chi2_compatible':  1.0,
    'pattern_match':    0.0,
}

_ML_FK_PATTERNS = ['_id','_fk','id_','fk_','_code','_num','_no','_key','_ref']
_ML_TYPE_GROUPS = {
    'int':  ['int','integer','bigint','smallint','tinyint','numeric','decimal','number'],
    'str':  ['varchar','nvarchar','char','nchar','text','ntext','string'],
    'date': ['date','datetime','datetime2','timestamp'],
}


def _ml_normalize(name: str) -> str:
    n = name.lower()
    for p in ['fk_','pk_','id_','num_','cod_','f_','c_']:
        if n.startswith(p): n = n[len(p):]; break
    for s in ['_id','_fk','_pk','_key','_code','_num','_no','_ref']:
        if n.endswith(s): n = n[:-len(s)]; break
    return n


def _ml_type_group(dtype: str) -> str:
    d = dtype.lower()
    return next((g for g, ts in _ML_TYPE_GROUPS.items() if any(t in d for t in ts)), 'other')


def _ml_fk_pat(name: str) -> float:
    n = name.lower()
    return float(any(n.startswith(p) or n.endswith(p) for p in _ML_FK_PATTERNS))


def _ml_sim(a: str, b: str) -> float:
    """Similarité Jaccard bigrams — identique au notebook v6."""
    a, b = a.lower(), b.lower()
    if a == b: return 1.0
    if not a or not b: return 0.0
    sa = set(a[i:i+2] for i in range(len(a)-1))
    sb = set(b[i:i+2] for i in range(len(b)-1))
    if not sa and not sb: return 0.0
    return len(sa & sb) / len(sa | sb)


def _ml_camel_parts(name: str) -> set:
    return set(p.lower() for p in _re.sub(r'([A-Z])', r' \1', name).split() if len(p) > 1)


def _ml_compute_features(
    ea: str, fa: str, dta: str, is_pk_a: bool, is_fk_a: bool,
    eb: str, fb: str, dtb: str, is_pk_b: bool, is_fk_b: bool,
    profile_index: Dict,
) -> Dict:
    na, nb     = _ml_normalize(fa), _ml_normalize(fb)
    ea_n, eb_n = _ml_normalize(ea), _ml_normalize(eb)
    parts_a    = _ml_camel_parts(fa)
    parts_b    = _ml_camel_parts(fb)
    common     = len(parts_a & parts_b) / max(len(parts_a | parts_b), 1)

    # Profiling features (neutres si absent)
    ka = (ea.upper(), fa.upper())
    kb = (eb.upper(), fb.upper())
    pa = profile_index.get(ka, {})
    pb = profile_index.get(kb, {})

    top_a = pa.get('top_values', set())
    top_b = pb.get('top_values', set())
    if top_a and top_b:
        inter = len(top_a & top_b)
        union = len(top_a | top_b)
        value_overlap = inter / union if union > 0 else 0.0
    else:
        value_overlap = 0.0

    uc_a = pa.get('unique_count', 0)
    uc_b = pb.get('unique_count', 0)
    cardinality_ratio = min(uc_a, uc_b) / max(uc_a, uc_b) if uc_a > 0 and uc_b > 0 else 0.5

    nr_a = pa.get('null_rate', 0.5)
    nr_b = pb.get('null_rate', 0.5)
    null_rate_compat = 1.0 - abs(nr_a - nr_b)

    return {
        'name_sim':               _ml_sim(fa, fb),
        'norm_sim':               _ml_sim(na, nb),
        'entity_in_field':        float(ea_n in nb or eb_n in na
                                        or _ml_sim(ea_n, nb) > 0.7
                                        or _ml_sim(eb_n, na) > 0.7),
        'type_compat':            float(_ml_type_group(dta) == _ml_type_group(dtb)),
        'fk_pattern_a':           _ml_fk_pat(fa),
        'fk_pattern_b':           _ml_fk_pat(fb),
        'pk_fk_pair':             float((is_pk_a and is_fk_b) or (is_pk_b and is_fk_a)),
        'common_parts':           common,
        'name_contains_entity_a': float(ea_n in fa.lower()),
        'name_contains_entity_b': float(eb_n in fb.lower()),
        'len_diff':               abs(len(fa) - len(fb)) / max(len(fa), len(fb), 1),
        'prefix_match':           float(fa[:3].lower() == fb[:3].lower()),
        'suffix_match':           float(fa[-3:].lower() == fb[-3:].lower()),
        'value_overlap':          value_overlap,
        'cardinality_ratio':      cardinality_ratio,
        'null_rate_compat':       null_rate_compat,
    }


def _ml_load_model(source_id: UUID) -> Tuple[Optional[object], float]:
    """
    Cherche le meilleur modèle pkl dans l'ordre :
      1. best_model_xgboost_{source_id}.pkl
      2. best_model_randomforest_{source_id}.pkl
      3. best_model_xgboost.pkl  (générique)
      4. best_model_randomforest.pkl
    Retourne (model, threshold) ou (None, 0.5).
    """
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        f"best_model_xgboost_{source_id}.pkl",
        f"best_model_randomforest_{source_id}.pkl",
        "best_model_xgboost.pkl",
        "best_model_randomforest.pkl",
    ]
    for name in candidates:
        path = os.path.join(base, name)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    bundle = pickle.load(f)
                model     = bundle.get('model')
                threshold = float(bundle.get('threshold', 0.5))
                logger.info(f"[ML] Modèle chargé : {name} (seuil={threshold:.3f})")
                return model, threshold, bundle
            except Exception as e:
                logger.warning(f"[ML] Erreur chargement {name}: {e}")
    logger.info("[ML] Aucun modèle pkl trouvé — algo ML ignoré")
    return None, 0.5, {}


async def _algo_ml_predict(
    source_id: UUID,
    fields_by_entity: Dict[str, List[Dict]],
    existing_pairs: set,
) -> List[RelationCandidate]:
    """
    Pipeline ML complet (notebook v6) :
    1. Charge le modèle pkl si disponible
    2. Charge les profils entity_profiles depuis DB (features profiling)
    3. Génère toutes les paires FK-candidate × PK-candidate
    4. Calcule les 16 features
    5. Prédit avec le modèle, filtre par seuil auto
    6. Retourne des RelationCandidate avec detection_method='ml_predicted'
    """
    model, threshold, _ml_bundle = _ml_load_model(source_id)
    if model is None:
        return []

    try:
        import numpy as np
    except ImportError:
        logger.warning("[ML] numpy non disponible — ML ignoré")
        return []

    # ── Charger entity_profiles depuis DB ──────────────────────
    profile_index: Dict = {}
    try:
        from .database import get_pg_pool
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            prof_rows = await conn.fetch(
                """SELECT entity_name, profile_data
                   FROM entity_profiles
                   WHERE source_id = $1""",
                source_id
            )
        import json as _json
        for pr in prof_rows:
            try:
                pd_data = pr["profile_data"]
                if isinstance(pd_data, str):
                    pd_data = _json.loads(pd_data)
                entity_name = pr["entity_name"].upper()
                # columns est une LISTE [{name, top_values,...}] dans data_profiler.py
                cols = pd_data.get("columns", [])
                if isinstance(cols, dict):
                    # ancien format dict — compatibilité
                    items = cols.items()
                else:
                    # format liste (data_profiler.py actuel)
                    items = [(c["name"], c) for c in cols if "name" in c]
                for col_name, col_data in items:
                    # top_values est une liste de {value, count} ou de valeurs directes
                    raw_tv = col_data.get("top_values") or []
                    top_vals = set()
                    for tv in raw_tv:
                        if isinstance(tv, dict):
                            v = tv.get("value")
                        else:
                            v = tv
                        if v is not None:
                            top_vals.add(str(v))
                    profile_index[(entity_name, col_name.upper())] = {
                        "top_values":   top_vals,
                        "unique_count": col_data.get("unique_count", 0),
                        "null_rate":    col_data.get("null_rate", 0.5),
                    }
            except Exception as _pe:
                logger.debug(f"[ML] Profil ignoré: {_pe}")
        logger.info(f"[ML] {len(profile_index)} profils chargés")
    except Exception as e:
        logger.warning(f"[ML] Profils non disponibles (features neutres): {e}")

    # ── Construire listes FK/PK candidates ──────────────────────
    fk_cands: List[Dict] = []
    pk_cands: List[Dict] = []

    for entity_name, fields in fields_by_entity.items():
        for f in fields:
            fname = f["name"]
            is_pk = f.get("is_primary_key", False)
            is_fk = f.get("is_foreign_key", False)
            dtype = f.get("data_type", "string")
            rec = {
                "entity_name": entity_name,
                "field_name":  fname,
                "data_type":   dtype,
                "is_primary_key": is_pk,
                "is_foreign_key": is_fk,
            }
            if is_pk:
                pk_cands.append(rec)
            if is_fk or _ml_fk_pat(fname) > 0:
                fk_cands.append(rec)

    if not fk_cands or not pk_cands:
        logger.info("[ML] Pas assez de candidats FK/PK pour ML")
        return []

    logger.info(f"[ML] {len(fk_cands)} FK-candidates × {len(pk_cands)} PK-candidates")

    # ── Stratification FK par table (max 20 par table) ─────────
    # Sans cap → 4051×2070 = 8.4M paires → freeze API
    FK_PER_TABLE = 20
    PK_CAP       = 600
    PAIRS_MAX    = 50_000

    from collections import defaultdict as _dd
    import random as _random
    fk_by_table = _dd(list)
    for f in fk_cands:
        fk_by_table[f["entity_name"]].append(f)

    fk_cands_strat = []
    for table, cols in fk_by_table.items():
        expl = [c for c in cols if c["is_foreign_key"]]
        impl = [c for c in cols if not c["is_foreign_key"]]
        selected = expl[:FK_PER_TABLE]
        rem = FK_PER_TABLE - len(selected)
        if rem > 0:
            selected += impl[:rem]
        fk_cands_strat += selected

    pk_cands = pk_cands[:PK_CAP]

    logger.info(f"[ML] Après stratification : {len(fk_cands_strat)} FK × {len(pk_cands)} PK")

    # ── Calcul topo_distance depuis les relations existantes ─────
    # Feature la plus importante du modèle v8 — calculée depuis le graphe en base
    _topo_dist: Dict = {}
    try:
        from .database import get_pg_pool as _get_pg
        _pool = await _get_pg()
        async with _pool.acquire() as _conn:
            _rel_rows = await _conn.fetch(
                "SELECT source_entity, target_entity FROM entity_relations "
                "WHERE source_id=$1 AND source_entity IS NOT NULL AND target_entity IS NOT NULL",
                source_id
            )
        from collections import deque as _deque
        _graph_topo: Dict = {}
        for _r in _rel_rows:
            _s = _r["source_entity"].upper()
            _t = _r["target_entity"].upper()
            _graph_topo.setdefault(_s, set()).add(_t)
            _graph_topo.setdefault(_t, set()).add(_s)
        for _start in list(_graph_topo.keys()):
            _visited = {_start: 0}
            _q = _deque([_start])
            while _q:
                _node = _q.popleft()
                for _nb in _graph_topo.get(_node, set()):
                    if _nb not in _visited:
                        _visited[_nb] = _visited[_node] + 1
                        _q.append(_nb)
            for _end, _dist in _visited.items():
                if _end != _start:
                    _score = round(1.0 / (1.0 + _dist), 4)
                    _topo_dist[(_start, _end)] = _score
        logger.info(f"[ML] topo_distance calculée : {len(_topo_dist)//2} paires")
    except Exception as _te:
        logger.warning(f"[ML] topo_distance non calculée (neutre=0.0): {_te}")

    # ── Générer paires et features ───────────────────────────────
    all_pairs_meta = []
    for fa in fk_cands_strat:
        for fb in pk_cands:
            if fa["entity_name"] == fb["entity_name"]:
                continue
            pair_key = (fa["entity_name"], fa["field_name"],
                        fb["entity_name"], fb["field_name"])
            if pair_key in existing_pairs:
                continue
            all_pairs_meta.append((fa, fb))

    # Garde-fou mémoire
    if len(all_pairs_meta) > PAIRS_MAX:
        _random.seed(42)
        all_pairs_meta = _random.sample(all_pairs_meta, PAIRS_MAX)
        logger.info(f"[ML] Sous-échantillonnage → {PAIRS_MAX:,} paires (seed=42)")

    pairs = []
    for fa, fb in all_pairs_meta:
        feat = _ml_compute_features(
            fa["entity_name"], fa["field_name"], fa["data_type"],
            fa["is_primary_key"], fa["is_foreign_key"],
            fb["entity_name"], fb["field_name"], fb["data_type"],
            fb["is_primary_key"], fb["is_foreign_key"],
            profile_index,
        )
        # Injecter topo_distance calculée depuis le graphe réel
        ea_up = fa["entity_name"].upper()
        eb_up = fb["entity_name"].upper()
        feat["topo_distance"] = _topo_dist.get((ea_up, eb_up), 0.0)
        feat.update({
            "source_entity": fa["entity_name"],
            "source_field":  fa["field_name"],
            "target_entity": fb["entity_name"],
            "target_field":  fb["field_name"],
        })
        pairs.append(feat)

    if not pairs:
        return []

    # ── Prédiction ML ────────────────────────────────────────────
    # Utiliser les features du modèle sauvegardé (16 v7 ou 21 v8)
    active_cols = _ml_bundle.get('active_cols',
                  _ml_bundle.get('feature_cols', _ML_FEATURE_COLS))
    # Ajouter les features v8 manquantes avec valeurs neutres
    for p in pairs:
        for col, default_val in _ML_V8_DEFAULTS.items():
            if col not in p:
                p[col] = default_val
    X = np.array([[p.get(c, 0.0) for c in active_cols] for p in pairs], dtype=float)
    logger.info(f"[ML] Prédiction sur {len(pairs)} paires avec {len(active_cols)} features")
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception as e:
        logger.error(f"[ML] predict_proba failed: {e}")
        return []

    # ── Filtrer par seuil + dedup par (source_entity, source_field) ──
    results: List[RelationCandidate] = []
    best_per_field: Dict[Tuple, float] = {}  # (src_entity, src_field) → best_conf

    indexed = sorted(
        [(proba[i], pairs[i]) for i in range(len(pairs)) if proba[i] >= threshold],
        key=lambda x: -x[0]
    )  # pas de cap — le seuil filtre naturellement

    seen_field: Dict[Tuple, bool] = {}
    for conf, p in indexed:
        key_field = (p["source_entity"], p["source_field"])
        if key_field in seen_field:
            continue
        seen_field[key_field] = True
        # Stocker les 9 features clés pour explication du score (Item 51)
        results.append(RelationCandidate(
            source_entity    = p["source_entity"],
            source_field     = p["source_field"],
            target_entity    = p["target_entity"],
            target_field     = p["target_field"],
            confidence       = round(float(conf), 4),
            detection_method = "ml_predicted",
            features         = {
                "value_overlap":     round(float(p.get("value_overlap")    or 0), 3),
                "name_sim":          round(float(p.get("name_sim")         or 0), 3),
                "norm_sim":          round(float(p.get("norm_sim")         or 0), 3),
                "topo_distance":     round(float(p.get("topo_distance")    or 0), 3),
                "pk_fk_pair":        round(float(p.get("pk_fk_pair")       or 0), 3),
                "fk_pattern_b":      round(float(p.get("fk_pattern_b")     or 0), 3),
                "type_compat":       round(float(p.get("type_compat")      or 0), 3),
                "embed_sim":         round(float(p.get("embed_sim")        or 0), 3),
                "cardinality_ratio": round(float(p.get("cardinality_ratio")or 0.5), 3),
            },
        ))

    logger.info(f"[ML] {len(results)} relations prédites (seuil={threshold:.3f})")
    return results


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
                      username, options
               FROM data_sources WHERE id=$1""",
            source_id)
        # Mot de passe dans connection_secrets (pas dans data_sources)
        secret_row = await conn.fetchrow(
            "SELECT secret_value FROM connection_secrets WHERE source_id=$1 AND secret_key='password' LIMIT 1",
            source_id)
        _db_password = secret_row["secret_value"] if secret_row else ""

        # ── Index entités ──────────────────────────────────────
        entity_index:     Dict[str, str]        = {}
        entity_names_set: set                   = set()
        fields_by_entity: Dict[str, List[Dict]] = {}
        id_to_name:       Dict[str, str]        = {}

        # ── Charger les faux positifs récurrents (≥3 rejets) ─────────────
        blacklist_rows = await conn.fetch(
            """SELECT source_entity, source_field, target_entity
               FROM relation_feedback
               WHERE source_id=$1 AND feedback='rejected'
               GROUP BY source_entity, source_field, target_entity
               HAVING COUNT(*) >= 3""",
            source_id)
        blacklist: set = {
            (r["source_entity"], r["source_field"], r["target_entity"])
            for r in blacklist_rows
        }
        if blacklist:
            logger.info(f"[discover] {len(blacklist)} faux positifs récurrents blacklistés")

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
                "password":       _db_password                or opts.get("password", ""),
            }

    # ══════════════════════════════════════════════════════════
    # PHASE 1 — Relations EXPLICITES (CDC §2.2.2.A)
    # FK déclarées, OData NavigationProperties → certitude 100%
    # ══════════════════════════════════════════════════════════
    explicit_cands = await _algo_explicit_fk(entity_names_set, conn_info)

    # Paires déjà connues via explicites → exclure du ML
    known_explicit: set = set()
    for c in explicit_cands:
        known_explicit.add((c.source_entity, c.source_field,
                            c.target_entity, c.target_field))

    # ══════════════════════════════════════════════════════════
    # PHASE 2 — Relations IMPLICITES ML-powered (CDC §2.2.2.B)
    #
    # Conforme CDC : name, fuzzy, value NE sont PAS des algos
    # indépendants — ce sont des FEATURES du pipeline ML.
    # Le modèle XGBoost/RandomForest prédit is_foreign_key
    # à partir des 16 features (dont name_sim, value_overlap...).
    #
    # Si .pkl disponible → ML prédit (mode normal)
    # Si .pkl absent    → fallback heuristiques + warning
    # ══════════════════════════════════════════════════════════
    model, _threshold, _ml_bundle = _ml_load_model(source_id)
    ml_available = model is not None

    if ml_available:
        # ── Mode normal : ML prédit sur les 16 features ──────
        implicit_cands = await _algo_ml_predict(
            source_id, fields_by_entity, known_explicit
        )
        logger.info(f"[discover] ML-powered: {len(implicit_cands)} relations implicites prédites")

    else:
        # ── Fallback : heuristiques en attendant l'entraînement ──
        # ⚠ NON CONFORME CDC — lancer ml_relation_detector_v6.ipynb
        # pour entraîner le modèle et activer le mode ML-powered
        logger.warning(
            "[discover] ⚠ FALLBACK heuristiques — modèle ML non disponible. "
            "Lancer ml_relation_detector_v6.ipynb pour activer ML-powered (CDC §2.2.2.B)"
        )
        name_cands  = _algo_name_based(entity_index, fields_by_entity)
        erp_cands   = _algo_erp_heuristics(entity_index, fields_by_entity)
        fuzzy_cands = _algo_fuzzy_match(entity_index, fields_by_entity)
        value_cands = await _algo_value_based(conn_info, name_cands)
        implicit_cands = _merge_candidates(
            [], value_cands, name_cands, fuzzy_cands, erp_cands,
            min_confidence=0.55
        )
        # Marquer clairement comme heuristique (pas ML)
        for c in implicit_cands:
            if c.detection_method not in ("explicit_fk", "view_join"):
                # Préfixer pour distinguer du ML-powered
                c.detection_method = f"heuristic_{c.detection_method}"

    # ══════════════════════════════════════════════════════════
    # MERGE FINAL — explicit + implicites ML-powered
    # ══════════════════════════════════════════════════════════
    # Filtrer les faux positifs récurrents (sauf FK explicites)
    all_candidates_raw = explicit_cands + implicit_cands
    all_candidates = []
    skipped_fp = 0
    for c in all_candidates_raw:
        key = (c.source_entity, c.source_field, c.target_entity)
        if c.detection_method != "explicit_fk" and key in blacklist:
            skipped_fp += 1
            continue
        all_candidates.append(c)
    if skipped_fp:
        logger.info(f"[discover] {skipped_fp} candidat(s) ignoré(s) — faux positifs récurrents")

    stats: Dict[str, int] = {}
    for c in all_candidates:
        stats[c.detection_method] = stats.get(c.detection_method, 0) + 1

    logger.info(
        f"[discover] explicit={len(explicit_cands)} | "
        f"implicit={'ml_powered' if ml_available else 'heuristic_fallback'}={len(implicit_cands)} | "
        f"ml_available={ml_available}"
    )

    # ══════════════════════════════════════════════════════════
    # PERSISTANCE
    # ══════════════════════════════════════════════════════════
    async with pool.acquire() as conn:
        # Supprimer uniquement les non-évaluées (NULL) — préserver confirmées/rejetées
        await conn.execute(
            """DELETE FROM entity_relations
               WHERE source_id=$1
               AND is_confirmed IS NULL""",
            source_id)

        saved = 0
        for rel in all_candidates:
            try:
                coverage = rel.features.get("value_overlap") or rel.features.get("coverage")
                # explicit_fk → TRUE (confirmé auto), implicites → NULL (à valider)
                auto_confirm = True if rel.detection_method == "explicit_fk" else None
                import json as _json
                feat_json = _json.dumps(rel.features) if rel.features else '{}'
                await conn.execute(
                    """INSERT INTO entity_relations
                           (source_id, source_entity, source_field,
                            target_entity, target_field,
                            relation_type, confidence, detection_method,
                            is_confirmed, value_overlap, features)
                       VALUES ($1,$2,$3,$4,$5,'many_to_one',$6,$7,$8,$9,$10::jsonb)
                       ON CONFLICT (source_id, source_entity, source_field, target_entity)
                       DO UPDATE SET
                           confidence       = EXCLUDED.confidence,
                           detection_method = EXCLUDED.detection_method,
                           value_overlap    = COALESCE(EXCLUDED.value_overlap,
                                                       entity_relations.value_overlap),
                           features         = EXCLUDED.features,
                           is_confirmed     = COALESCE(entity_relations.is_confirmed,
                                                       EXCLUDED.is_confirmed)""",
                    source_id,
                    rel.source_entity, rel.source_field,
                    rel.target_entity, rel.target_field,
                    rel.confidence,    rel.detection_method,
                    auto_confirm,      coverage,   feat_json,
                )
                saved += 1
            except Exception as e:
                logger.warning(f"[discover] INSERT error: {e}")

    logger.info(f"[discover] {saved} relations sauvegardées | {stats}")

    # PATCH P4 — Persister le graphe dans Redis + détecter les cycles
    try:
        _relations_for_graph = await get_relations_for_source(source_id)
        _graph = _build_join_graph(_relations_for_graph)
        _cycles = detect_cycles(_graph)
        await persist_graph_to_redis(source_id, _graph, _cycles)
        if _cycles:
            logger.warning(f"[discover] {len(_cycles)} cycle(s) détecté(s) dans le graphe de relations")
    except Exception as _graph_err:
        logger.warning(f"[discover] Graph persistence failed (non-bloquant): {_graph_err}")

    return {
        "success":           True,
        "relations_found":   saved,
        "blacklisted_pairs": skipped_fp,
        "stats_by_algo":   stats,
        "ml_powered":      ml_available,
        "ml_warning":      None if ml_available else (
            "Modèle ML non entraîné — relations implicites détectées par heuristiques. "
            "Lancer ml_relation_detector_v6.ipynb pour activer ML-powered (CDC §2.2.2.B)."
        ),
        "message": (
            f"{saved} relation(s) [{'ML-powered' if ml_available else '⚠ heuristique fallback'}] — "
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
                      value_overlap, reject_reason,
                      features
               FROM entity_relations
               WHERE source_id=$1
               ORDER BY confidence DESC, source_entity, source_field""",
            source_id)

    import json as _json, math as _math
    def _safe_v(v):
        if v is None: return None
        try:
            f = float(v)
            return None if (_math.isnan(f) or _math.isinf(f)) else round(f, 3)
        except: return None

    def _parse_features(raw):
        if not raw: return {}
        try:
            d = _json.loads(raw) if isinstance(raw, str) else dict(raw)
            return {k: (None if v is None else (round(float(v),3) if isinstance(v,(int,float)) else v))
                    for k, v in d.items()}
        except: return {}

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
            "value_overlap":    _safe_v(r["value_overlap"]),
            "features":         _parse_features(r["features"]),
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
                ROUND(LEAST(AVG(value_overlap), 1.0)::numeric, 3)   AS avg_overlap,
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

        def _safe_float(v, default=0.0):
            """Convertit en float JSON-safe (élimine NaN, Infinity)."""
            try:
                f = float(v or default)
                import math
                return default if (math.isnan(f) or math.isinf(f)) else round(f, 3)
            except (TypeError, ValueError):
                return default

        return {
            "total":          row["total"],
            "confirmed":      confirmed,
            "rejected":       rejected,
            "pending":        row["pending"] or 0,
            "accuracy_est":   round(confirmed / max(confirmed + rejected, 1) * 100, 1),
            "avg_confidence": _safe_float(row["avg_conf"]),
            "avg_overlap":    _safe_float(row["avg_overlap"]),
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
# PATCH P4 — Détection de cycles (Tarjan) + persistance graphe Redis
# ══════════════════════════════════════════════════════════════════════

import json as _json_rd

def detect_cycles(graph: Dict) -> List[List[str]]:
    """
    Détection de cycles via DFS (Tarjan simplifié).
    graph : {node -> [(neighbor, via, tgt, conf), ...]}
    Retourne liste de cycles (chaque cycle = liste de nœuds).
    """
    from typing import Set as _Set
    visited:   _Set[str] = set()
    rec_stack: _Set[str] = set()
    cycles:    List[List[str]] = []
    path:      List[str] = []

    def _dfs(node: str):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        for (neighbor, _, _, _) in graph.get(node, []):
            if neighbor not in visited:
                _dfs(neighbor)
            elif neighbor in rec_stack:
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycle_key = frozenset(cycle)
                if not any(frozenset(c) == cycle_key for c in cycles):
                    cycles.append(cycle)
        path.pop()
        rec_stack.discard(node)

    for node in list(graph.keys()):
        if node not in visited:
            _dfs(node)
    return cycles


async def persist_graph_to_redis(source_id: UUID, graph: Dict, cycles: List[List[str]]) -> None:
    """
    PATCH P4 — Persiste le graphe dans Redis (TTL 1h) + metadata en PostgreSQL.
    """
    try:
        from .database import cache_set, get_pg_pool
    except ImportError:
        from database import cache_set, get_pg_pool  # type: ignore

    serializable: Dict[str, List] = {}
    for node, edges in graph.items():
        serializable[node] = [
            {"to": nb, "via": via, "tgt": tgt, "conf": round(conf, 4)}
            for (nb, via, tgt, conf) in edges
        ]
    node_count = len(graph)
    edge_count = sum(len(v) for v in graph.values()) // 2

    await cache_set(f"graph:{source_id}", {
        "nodes": list(graph.keys()),
        "edges": serializable,
        "cycles": cycles,
        "node_count": node_count,
        "edge_count": edge_count,
        "cycle_count": len(cycles),
    }, ttl=3600)

    logger.info(f"[graph] Redis: {node_count} nœuds, {edge_count} arêtes, {len(cycles)} cycles — {source_id}")

    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO relation_graph_meta
                    (source_id, node_count, edge_count, cycle_count, cycles, computed_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, NOW())
                ON CONFLICT (source_id) DO UPDATE SET
                    node_count  = EXCLUDED.node_count,
                    edge_count  = EXCLUDED.edge_count,
                    cycle_count = EXCLUDED.cycle_count,
                    cycles      = EXCLUDED.cycles,
                    computed_at = NOW()
            """, str(source_id), node_count, edge_count, len(cycles), _json_rd.dumps(cycles))
    except Exception as e:
        logger.warning(f"[graph] PostgreSQL meta update failed: {e}")


async def get_graph_from_redis(source_id: UUID) -> Optional[Dict]:
    """Charge le graphe depuis Redis. Retourne None si absent/expiré."""
    try:
        from .database import cache_get
    except ImportError:
        from database import cache_get  # type: ignore
    data = await cache_get(f"graph:{source_id}")
    if not data:
        return None
    graph: Dict[str, List] = {}
    for node, edges in data.get("edges", {}).items():
        graph[node] = [(e["to"], e["via"], e["tgt"], e["conf"]) for e in edges]
    return graph


# ══════════════════════════════════════════════════════════════════════
# PATCH P5 — Suggestions de relations alternatives
# ══════════════════════════════════════════════════════════════════════

async def get_alternative_suggestions(
    source_id: UUID,
    source_entity: str,
    source_field: str,
    current_target: str,
    top_k: int = 3,
) -> List[Dict]:
    """
    PATCH P5 — Retourne les K meilleures relations alternatives pour un champ donné,
    hors la relation actuelle. Utilisé par l'UI de validation.
    """
    try:
        from .database import get_pg_pool
    except ImportError:
        from database import get_pg_pool  # type: ignore
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT target_entity, target_field, confidence, detection_method, value_overlap
            FROM entity_relations
            WHERE source_id    = $1
              AND source_entity = $2
              AND source_field  = $3
              AND target_entity != $4
              AND confidence    >= 0.50
            ORDER BY confidence DESC
            LIMIT $5
        """, source_id, source_entity, source_field, current_target, top_k)
    return [
        {
            "target_entity":    row["target_entity"],
            "target_field":     row["target_field"],
            "confidence":       round(row["confidence"], 3),
            "detection_method": row["detection_method"],
            "value_overlap":    row["value_overlap"],
        }
        for row in rows
    ]


# ══════════════════════════════════════════════════════════════════════
# PATCH P6 — Feedback expert + trigger re-training ML
# ══════════════════════════════════════════════════════════════════════

RETRAIN_THRESHOLD = 20  # feedbacks avant re-training automatique


async def record_expert_feedback(
    source_id:  UUID,
    relation_id: UUID,
    source_entity: str,
    source_field: str,
    target_entity: str,
    target_field: str,
    feedback: str,           # 'confirmed' | 'rejected' | 'alternative'
    user_id: Optional[str] = None,
    comment: Optional[str] = None,
    alternative_target_entity: Optional[str] = None,
    alternative_target_field:  Optional[str] = None,
) -> Dict:
    """
    PATCH P6 — Enregistre le feedback expert et met à jour entity_relations.
    Retourne should_retrain=True si le seuil de re-training est atteint.
    """
    try:
        from .database import get_pg_pool
    except ImportError:
        from database import get_pg_pool  # type: ignore
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO relation_feedback
                (source_id, relation_id, source_entity, source_field,
                 target_entity, target_field, feedback, user_id, comment,
                 alternative_target_entity, alternative_target_field)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
        """, source_id, relation_id, source_entity, source_field,
            target_entity, target_field, feedback, user_id, comment,
            alternative_target_entity, alternative_target_field)

        if feedback == "confirmed":
            await conn.execute("""
                UPDATE entity_relations
                SET is_confirmed=TRUE, validated_by=$2, validated_at=NOW()
                WHERE id=$1
            """, relation_id, user_id)
        elif feedback == "rejected":
            await conn.execute("""
                UPDATE entity_relations
                SET is_confirmed=FALSE, validated_by=$2, validated_at=NOW(), reject_reason=$3
                WHERE id=$1
            """, relation_id, user_id, comment)

        cnt_row = await conn.fetchrow(
            "SELECT COUNT(*) as cnt FROM relation_feedback WHERE source_id=$1", source_id)
        total = cnt_row["cnt"]

    should_retrain = (total % RETRAIN_THRESHOLD == 0 and total > 0)
    return {
        "recorded":        True,
        "total_feedbacks": total,
        "should_retrain":  should_retrain,
        "message": (
            f"Feedback enregistré. {total} feedbacks."
            + (" Re-training ML recommandé." if should_retrain else "")
        ),
    }

# ══════════════════════════════════════════════════════════════════════
# FAUX POSITIFS RÉCURRENTS — utilitaires
# ══════════════════════════════════════════════════════════════════════

async def get_false_positive_pairs(source_id: UUID, min_rejections: int = 3) -> List[Dict]:
    """Retourne les paires rejetées >= min_rejections fois (blacklistées)."""
    try:
        from .database import get_pg_pool
    except ImportError:
        from database import get_pg_pool  # type: ignore
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT source_entity, source_field, target_entity,
                      COUNT(*) as nb_rejets,
                      MAX(created_at) as last_rejected_at
               FROM relation_feedback
               WHERE source_id=$1 AND feedback='rejected'
               GROUP BY source_entity, source_field, target_entity
               HAVING COUNT(*) >= $2
               ORDER BY COUNT(*) DESC""",
            source_id, min_rejections)
    return [
        {
            "source_entity":    r["source_entity"],
            "source_field":     r["source_field"],
            "target_entity":    r["target_entity"],
            "nb_rejets":        r["nb_rejets"],
            "last_rejected_at": r["last_rejected_at"].isoformat() if r["last_rejected_at"] else None,
            "blacklisted":      True,
        }
        for r in rows
    ]


async def remove_false_positive(
    source_id: UUID,
    source_entity: str,
    source_field: str,
    target_entity: str,
) -> Dict:
    """Réhabilite une paire blacklistée — supprime ses feedbacks négatifs."""
    try:
        from .database import get_pg_pool
    except ImportError:
        from database import get_pg_pool  # type: ignore
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """DELETE FROM relation_feedback
               WHERE source_id=$1 AND source_entity=$2
                 AND source_field=$3 AND target_entity=$4
                 AND feedback='rejected'""",
            source_id, source_entity, source_field, target_entity)
    return {
        "rehabilitated": True,
        "source_entity": source_entity,
        "source_field":  source_field,
        "target_entity": target_entity,
        "message":       "Paire réhabilitée — reproposée au prochain discover",
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