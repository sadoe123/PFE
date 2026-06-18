"""
OnePilot – data_profiler.py
§2.2.4 Profiling des données avancé — Multi-connecteur générique

Dépendances : SQLAlchemy + pyodbc (déjà installés), requests (déjà installé)
Aucune nouvelle dépendance requise — fonctionne avec ce qui est déjà dans requirements.txt
"""
from __future__ import annotations

import re
import json
import logging
import asyncio
import math
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# PATTERNS SÉMANTIQUES
# ══════════════════════════════════════════════════════════════════════

PATTERNS = {
    "email":       re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$"),
    "uuid":        re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I),
    "url":         re.compile(r"^https?://[^\s]+$"),
    "phone":       re.compile(r"^[\+\d][\d\s\-\(\)\.]{6,20}$"),
    "iban":        re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}$"),
    "siret":       re.compile(r"^\d{14}$"),
    "siren":       re.compile(r"^\d{9}$"),
    "zip_fr":      re.compile(r"^\d{5}$"),
    "iso_date":    re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?"),
    "date_fr":     re.compile(r"^\d{2}/\d{2}/\d{4}$"),
    "currency":    re.compile(r"^[\$\u20ac\xa3\xa5]\s*[\d,\.]+|[\d,\.]+\s*[\$\u20ac\xa3\xa5]$"),
    "json_str":    re.compile(r"^\s*[\{\[]"),
    "boolean_str": re.compile(r"^(true|false|yes|no|oui|non|1|0)$", re.I),
    "alpha_code":  re.compile(r"^[A-Z0-9\-_]{2,20}$"),
    "ean13":       re.compile(r"^\d{13}$"),
    "ean8":        re.compile(r"^\d{8}$"),
    "iso_country": re.compile(r"^[A-Z]{2,3}$"),       # ISO-3166 alpha-2/3
    "swift":       re.compile(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$"),
    "postal_code": re.compile(r"^\d{5}(-\d{4})?$|^[A-Z]\d[A-Z]\s?\d[A-Z]\d$"),
}

SEMANTIC_LABELS = {
    "email":       {"icon": "\u2709\ufe0f",  "label": "Email",         "color": "#818cf8"},
    "uuid":        {"icon": "\U0001f511",    "label": "UUID",          "color": "#f59e0b"},
    "url":         {"icon": "\U0001f310",    "label": "URL",           "color": "#06b6d4"},
    "phone":       {"icon": "\U0001f4de",    "label": "Telephone",     "color": "#8b5cf6"},
    "iban":        {"icon": "\U0001f3e6",    "label": "IBAN",          "color": "#10b981"},
    "siret":       {"icon": "\U0001f3e2",    "label": "SIRET",         "color": "#f97316"},
    "siren":       {"icon": "\U0001f3e2",    "label": "SIREN",         "color": "#f97316"},
    "zip_fr":      {"icon": "\U0001f4ee",    "label": "Code postal",   "color": "#ec4899"},
    "iso_date":    {"icon": "\U0001f4c5",    "label": "Date ISO",      "color": "#22d3ee"},
    "date_fr":     {"icon": "\U0001f4c5",    "label": "Date FR",       "color": "#22d3ee"},
    "currency":    {"icon": "\U0001f4b0",    "label": "Montant",       "color": "#84cc16"},
    "json_str":    {"icon": "\U0001f4e6",    "label": "JSON",          "color": "#a78bfa"},
    "boolean_str": {"icon": "\u26a1",        "label": "Booleen",       "color": "#fb923c"},
    "alpha_code":  {"icon": "\U0001f3f7\ufe0f","label": "Code metier", "color": "#94a3b8"},
    "numeric":     {"icon": "\U0001f522",    "label": "Numerique",     "color": "#34d399"},
    "text":        {"icon": "\U0001f4dd",    "label": "Texte",         "color": "#9ca3af"},
    "pk":          {"icon": "\U0001f5dd\ufe0f","label": "Cle primaire","color": "#fbbf24"},
    "fk":          {"icon": "\U0001f517",    "label": "Cle etrangere", "color": "#60a5fa"},
    "date_col":    {"icon": "\U0001f4c5",    "label": "Date",          "color": "#22d3ee"},
    "amount_col":  {"icon": "\U0001f4b0",    "label": "Montant",       "color": "#84cc16"},
    "flag_col":    {"icon": "\U0001f6a9",    "label": "Flag/Statut",   "color": "#fb923c"},
    "ean13":       {"icon": "▦",             "label": "EAN-13",        "color": "#0ea5e9"},
    "ean8":        {"icon": "▦",             "label": "EAN-8",         "color": "#0ea5e9"},
    "iso_country": {"icon": "◎",             "label": "Code pays ISO", "color": "#14b8a6"},
    "swift":       {"icon": "◈",             "label": "SWIFT/BIC",     "color": "#6366f1"},
    "postal_code": {"icon": "◉",             "label": "Code postal",   "color": "#ec4899"},
    "unknown":     {"icon": "?",             "label": "Inconnu",       "color": "#6b7280"},
}

# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════

def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    idx = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo), 4)


def detect_pattern(values: List[str]) -> Tuple[str, float]:
    if not values:
        return "unknown", 0.0
    sample = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not sample:
        return "unknown", 0.0
    counts: Dict[str, int] = {}
    for val in sample:
        for name, rx in PATTERNS.items():
            if rx.match(val):
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return "text", 1.0
    best = max(counts, key=lambda k: counts[k])
    conf = round(counts[best] / len(sample), 3)
    return (best, conf) if conf >= 0.5 else ("text", 1.0)


def classify_column(col_name: str, col_type: str,
                    is_pk: bool, is_fk: bool, pattern: str) -> str:
    if is_pk:
        return "pk"
    if is_fk:
        return "fk"
    n = col_name.lower()
    t = col_type.lower()
    if any(x in n for x in ["date", "time", "at", "created", "updated", "modified"]):
        if any(x in t for x in ["date", "time", "timestamp"]):
            return "date_col"
    if any(x in n for x in ["amount", "montant", "total", "price", "prix",
                              "cost", "revenue", "balance", "solde"]):
        return "amount_col"
    if any(x in n for x in ["flag", "is_", "active", "enabled", "status",
                              "statut", "bool", "indicator"]):
        return "flag_col"
    if pattern in SEMANTIC_LABELS and pattern not in ("text", "unknown"):
        return pattern
    if any(x in t for x in ["int", "numeric", "decimal", "float", "double"]):
        return "numeric"
    return "text"


# ══════════════════════════════════════════════════════════════════════
# CONNEXION SQLALCHEMY — réutilise pyodbc déjà installé
# ══════════════════════════════════════════════════════════════════════

def _detect_odbc_driver() -> str:
    """Détecte automatiquement le driver ODBC SQL Server disponible."""
    try:
        import pyodbc
        available = [d for d in pyodbc.drivers()
                     if "SQL Server" in d or "ODBC" in d.upper()]
        for preferred in ["ODBC Driver 18 for SQL Server",
                          "ODBC Driver 17 for SQL Server",
                          "ODBC Driver 13 for SQL Server",
                          "SQL Server"]:
            if preferred in available:
                logger.info(f"[Profiler] Driver ODBC détecté: {preferred}")
                return preferred
        if available:
            logger.info(f"[Profiler] Driver ODBC fallback: {available[0]}")
            return available[0]
    except Exception as e:
        logger.warning(f"[Profiler] pyodbc.drivers() erreur: {e}")
    return "ODBC Driver 18 for SQL Server"


def _build_url(source_dict: Dict) -> str:
    ct   = (source_dict.get("connector_type") or "").lower()
    opts = source_dict.get("options") or {}
    if isinstance(opts, str):
        try:
            opts = json.loads(opts)
        except Exception:
            opts = {}

    host = source_dict.get("host")          or opts.get("host", "localhost")
    port = source_dict.get("port")          or opts.get("port")
    db   = source_dict.get("database_name") or opts.get("database_name", "")
    user = source_dict.get("username")      or opts.get("username", "")
    pwd  = source_dict.get("password")      or opts.get("password", "")

    if any(x in ct for x in ["mssql", "sql_server", "sqlserver"]) or (
            not any(x in ct for x in ["postgres", "pg", "mysql", "sqlite",
                                       "odata", "rest", "file", "csv", "json"])):
        driver = _detect_odbc_driver()
        port   = port or 1433
        driver_enc = driver.replace(" ", "+")
        return (
            f"mssql+pyodbc://{user}:{pwd}@{host}:{port}/{db}"
            f"?driver={driver_enc}"
            "&TrustServerCertificate=yes&Encrypt=no"
        )
    if "postgres" in ct or ct == "pg":
        port = port or 5432
        return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    if "mysql" in ct:
        port = port or 3306
        return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}"
    if "sqlite" in ct:
        return f"sqlite:///{db}"
    driver = _detect_odbc_driver()
    port   = port or 1433
    driver_enc = driver.replace(" ", "+")
    return (
        f"mssql+pyodbc://{user}:{pwd}@{host}:{port}/{db}"
        f"?driver={driver_enc}"
        "&TrustServerCertificate=yes&Encrypt=no"
    )


# ══════════════════════════════════════════════════════════════════════
# PROFILING SQL SYNCHRONE (appelé dans thread)
# ══════════════════════════════════════════════════════════════════════

def _sync_profile_sql(source_dict: Dict, table_name: str,
                       fields: List[Dict], sample_size: int) -> Dict:
    from sqlalchemy import create_engine, text

    ct     = (source_dict.get("connector_type") or "").lower()
    url    = _build_url(source_dict)
    schema = (source_dict.get("schema_name") or "").strip()

    # Table qualifiée avec schéma
    if any(x in ct for x in ["mssql", "sql_server", "sqlserver"]):
        sc = schema or "dbo"
        qtable = f"[{sc}].[{table_name}]"
    elif "mysql" in ct:
        qtable = f"`{table_name}`"
    else:
        sc = schema or "public"
        qtable = f'"{sc}"."{table_name}"'

    try:
        engine = create_engine(url, pool_pre_ping=True,
                                pool_size=1, max_overflow=0,
                                connect_args={"timeout": 30})
    except Exception:
        engine = create_engine(url, pool_pre_ping=True)

    try:
        with engine.connect() as conn:
            try:
                if any(x in ct for x in ["mssql", "sql_server", "sqlserver"]):
                    cnt = conn.execute(text(f"SELECT COUNT_BIG(*) FROM {qtable}")).scalar()
                elif "mysql" in ct:
                    cnt = conn.execute(text(f"SELECT COUNT(*) FROM {qtable}")).scalar()
                else:
                    cnt = conn.execute(text(f"SELECT COUNT(*) FROM {qtable}")).scalar()
                total_rows = int(cnt or 0)
            except Exception:
                total_rows = 0

            try:
                if any(x in ct for x in ["mssql", "sql_server", "sqlserver"]):
                    sql = f"SELECT TOP {sample_size} * FROM {qtable}"
                elif "mysql" in ct:
                    sql = f"SELECT * FROM {qtable} LIMIT {sample_size}"
                else:
                    sql = f"SELECT * FROM {qtable} LIMIT {sample_size}"

                result    = conn.execute(text(sql))
                col_names = list(result.keys())
                rows      = result.fetchall()
            except Exception as e:
                engine.dispose()
                return _empty_profile(table_name, fields, f"Erreur sample: {e}")

        engine.dispose()

        columns_profile = []
        for field in fields:
            col = field["name"]
            if col not in col_names:
                continue
            idx    = col_names.index(col)
            values = [r[idx] for r in rows]
            columns_profile.append(_compute_single_column(col, field, values, total_rows))

        # Classification référentiel vs transactionnel au niveau table
        if total_rows < 500:
            table_class = "reference"
        elif total_rows > 10000:
            table_class = "transactional"
        else:
            date_cols = sum(1 for c in columns_profile
                           if c.get("semantic") in ("date_col", "iso_date", "date_fr"))
            table_class = "transactional" if date_cols >= 2 else "reference"

        return {
            "table":        table_name,
            "total_rows":   total_rows,
            "sampled_rows": len(rows),
            "table_class":  table_class,
            "columns":      columns_profile,
            "profiled_at":  datetime.utcnow().isoformat(),
            "error":        None,
        }

    except Exception as e:
        logger.error(f"[Profiler] SQL {table_name}: {e}")
        try:
            engine.dispose()
        except Exception:
            pass
        return _empty_profile(table_name, fields, str(e))


# ══════════════════════════════════════════════════════════════════════
# CALCUL STATS PAR COLONNE
# ══════════════════════════════════════════════════════════════════════

def _compute_single_column(col_name: str, field: Dict,
                            values: List[Any], total_rows: int) -> Dict:
    n        = len(values)
    is_pk    = field.get("is_primary_key", False)
    is_fk    = field.get("is_foreign_key",  False)
    col_type = field.get("type", "string")

    nulls        = sum(1 for v in values if v is None)
    null_rate    = round(nulls / n, 4) if n else 0.0
    non_nulls    = [v for v in values if v is not None]
    str_vals     = [str(v) for v in non_nulls]
    counts_map   = Counter(str_vals)
    unique_count = len(counts_map)
    unique_rate  = round(unique_count / n, 4) if n else 0.0
    duplicates   = sum(1 for c in counts_map.values() if c > 1)
    dup_rate     = round(duplicates / unique_count, 4) if unique_count else 0.0
    top_values   = [{"value": v, "count": c} for v, c in counts_map.most_common(5)]

    num_stats: Dict = {}
    numeric_vals = [f for v in non_nulls for f in [_safe_float(v)] if f is not None]
    if numeric_vals:
        numeric_vals.sort()
        mean = sum(numeric_vals) / len(numeric_vals)
        num_stats = {
            "min":    numeric_vals[0],
            "max":    numeric_vals[-1],
            "mean":   round(mean, 4),
            "median": _percentile(numeric_vals, 50),
            "p25":    _percentile(numeric_vals, 25),
            "p75":    _percentile(numeric_vals, 75),
            "p95":    _percentile(numeric_vals, 95),
            "std":    round(math.sqrt(
                sum((x - mean) ** 2 for x in numeric_vals) / len(numeric_vals)
            ), 4) if len(numeric_vals) > 1 else 0.0,
        }
        mn, mx = numeric_vals[0], numeric_vals[-1]
        if mx > mn:
            bsize   = (mx - mn) / 10
            buckets = [0] * 10
            for v in numeric_vals:
                buckets[min(int((v - mn) / bsize), 9)] += 1
            num_stats["histogram"] = {"buckets": buckets, "min": mn,
                                       "max": mx, "bucket_size": round(bsize, 4)}

    text_stats: Dict = {}
    pattern_name, pattern_conf = "unknown", 0.0
    if str_vals and col_type in ("string", "text", "varchar", "char", "nvarchar"):
        lengths = sorted(len(s) for s in str_vals)
        pattern_name, pattern_conf = detect_pattern(str_vals[:500])
        text_stats = {
            "min_length":   lengths[0]  if lengths else 0,
            "max_length":   lengths[-1] if lengths else 0,
            "avg_length":   round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "pattern":      pattern_name,
            "pattern_conf": pattern_conf,
        }
    elif numeric_vals:
        pattern_name, pattern_conf = "numeric", 1.0

    outliers_count = 0
    if numeric_vals and num_stats.get("p25") is not None and num_stats.get("p75") is not None:
        iqr = num_stats["p75"] - num_stats["p25"]
        lo  = num_stats["p25"] - 1.5 * iqr
        hi  = num_stats["p75"] + 1.5 * iqr
        outliers_count = sum(1 for v in numeric_vals if v < lo or v > hi)

    # ── Détection d'inconsistencies (normalisations différentes) ────────
    inconsistencies = 0
    if str_vals and col_type in ("string", "text", "varchar", "nvarchar"):
        normalized = [v.strip().lower() for v in str_vals]
        norm_counts = Counter(normalized)
        for norm, norm_c in norm_counts.items():
            # Cherche des variantes (casse/espaces différents) de la même valeur
            raw_variants = set(v.strip() for v in str_vals if v.strip().lower() == norm)
            if len(raw_variants) > 1:
                inconsistencies += norm_c

    # ── Détection de missing patterns (nulls groupés) ───────────────────
    missing_pattern = "random"
    if nulls > 0 and n > 10:
        null_positions = [i for i, v in enumerate(values) if v is None]
        if len(null_positions) > 1:
            gaps = [null_positions[i+1] - null_positions[i] for i in range(len(null_positions)-1)]
            avg_gap = sum(gaps) / len(gaps) if gaps else 0
            if avg_gap < 3:
                missing_pattern = "clustered"   # nulls groupés → import partiel
            elif max(gaps) < 5 if gaps else False:
                missing_pattern = "systematic"  # nulls réguliers → champ non rempli

    # ── Classification référentiel vs transactionnel ─────────────────────
    table_class = "unknown"
    if n > 0:
        if unique_rate > 0.95 and null_rate < 0.05:
            table_class = "reference"       # peu de doublons, pas de nulls → référentiel
        elif unique_rate < 0.3 and n > 1000:
            table_class = "transactional"   # beaucoup de doublons, gros volume → transactionnel
        elif n < 200 and unique_rate > 0.8:
            table_class = "reference"
        else:
            table_class = "transactional"

    semantic = classify_column(col_name, col_type, is_pk, is_fk, pattern_name)
    sem_meta = SEMANTIC_LABELS.get(semantic, SEMANTIC_LABELS["unknown"])

    quality_score = max(0, 100 - int(null_rate * 40) - int(dup_rate * 20)
                        - (min(20, int(outliers_count / n * 100)) if outliers_count and n else 0))

    issues = []
    if null_rate > 0.3:
        issues.append({"type": "high_null_rate",
                       "msg": f"{int(null_rate*100)}% de valeurs nulles",
                       "severity": "error" if null_rate > 0.8 else "warning"})
    if dup_rate > 0.5 and not is_fk and unique_count > 1:
        issues.append({"type": "duplicates",
                       "msg": f"{duplicates} valeurs dupliquees", "severity": "info"})
    if outliers_count > 0:
        issues.append({"type": "outliers",
                       "msg": f"{outliers_count} outliers (IQR)", "severity": "info"})
    if is_pk and unique_rate < 1.0:
        issues.append({"type": "pk_not_unique",
                       "msg": "Cle primaire non unique !", "severity": "error"})
    if inconsistencies > 0:
        issues.append({"type": "inconsistencies",
                       "msg": f"{inconsistencies} inconsistances (casse/espaces)",
                       "severity": "warning"})
    if missing_pattern == "clustered":
        issues.append({"type": "missing_clustered",
                       "msg": "Nulls groupes — possible import partiel",
                       "severity": "warning"})
    elif missing_pattern == "systematic":
        issues.append({"type": "missing_systematic",
                       "msg": "Nulls systematiques — champ non rempli",
                       "severity": "info"})

    return {
        "name":             col_name,
        "type":             col_type,
        "is_pk":            is_pk,
        "is_fk":            is_fk,
        "semantic":         semantic,
        "sem_icon":         sem_meta["icon"],
        "sem_label":        sem_meta["label"],
        "sem_color":        sem_meta["color"],
        "total":            n,
        "null_count":       nulls,
        "null_rate":        null_rate,
        "unique_count":     unique_count,
        "unique_rate":      unique_rate,
        "dup_rate":         dup_rate,
        "top_values":       top_values,
        "numeric_stats":    num_stats,
        "text_stats":       text_stats,
        "outliers":         outliers_count,
        "inconsistencies":  inconsistencies,
        "missing_pattern":  missing_pattern,
        "column_class":     table_class,
        "quality_score":    quality_score,
        "issues":           issues,
        "pattern":          text_stats.get("pattern", pattern_name),
        "pattern_conf":     text_stats.get("pattern_conf", pattern_conf),
    }


# ══════════════════════════════════════════════════════════════════════
# PROFILING OData / REST / FILE (synchrone → thread)
# ══════════════════════════════════════════════════════════════════════

def _sync_profile_odata(source_dict: Dict, entity_name: str,
                         fields: List[Dict], sample_size: int) -> Dict:
    import requests
    opts     = _parse_opts(source_dict)
    # base_url est un champ direct de data_sources, pas dans options
    base_url = (source_dict.get("base_url") or
                opts.get("base_url") or
                source_dict.get("host") or "")
    if not base_url:
        return _empty_profile(entity_name, fields, "URL OData manquante")
    try:
        url  = f"{base_url.rstrip('/')}/{entity_name}?$top={sample_size}&$format=json"
        data = requests.get(url, timeout=30).json()
        items = data.get("value", data) if isinstance(data, dict) else data
        return _profile_records(entity_name, fields,
                                items if isinstance(items, list) else [items])
    except Exception as e:
        return _empty_profile(entity_name, fields, str(e))


def _sync_profile_rest(source_dict: Dict, entity_name: str,
                        fields: List[Dict], sample_size: int) -> Dict:
    import requests
    opts     = _parse_opts(source_dict)
    base_url = (source_dict.get("base_url") or
                opts.get("base_url") or
                source_dict.get("host") or "")
    try:
        data  = requests.get(f"{base_url.rstrip('/')}/{entity_name}", timeout=30).json()
        items = data if isinstance(data, list) else [data]
        return _profile_records(entity_name, fields, items[:sample_size])
    except Exception as e:
        return _empty_profile(entity_name, fields, str(e))


def _sync_profile_file(source_dict: Dict, entity_name: str,
                        fields: List[Dict], sample_size: int) -> Dict:
    import csv as csv_mod, io
    opts      = _parse_opts(source_dict)
    file_path = opts.get("file_path") or opts.get("uploaded_path") or ""
    if not file_path:
        return _empty_profile(entity_name, fields, "Chemin fichier manquant")
    try:
        ext = file_path.lower().split(".")[-1]
        with open(file_path, "r", encoding="utf-8-sig") as f:
            content = f.read()
        if ext == "csv":
            reader = csv_mod.DictReader(io.StringIO(content))
            rows = [r for i, r in enumerate(reader) if i < sample_size]
        else:
            data = json.loads(content)
            rows = (data if isinstance(data, list) else [data])[:sample_size]
        return _profile_records(entity_name, fields, rows)
    except Exception as e:
        return _empty_profile(entity_name, fields, str(e))


def _parse_opts(source_dict: Dict) -> Dict:
    opts = source_dict.get("options") or {}
    if isinstance(opts, str):
        try:
            return json.loads(opts)
        except Exception:
            return {}
    return opts


def _profile_records(entity_name: str, fields: List[Dict], records: List[Dict]) -> Dict:
    if not records:
        return _empty_profile(entity_name, fields, "Aucun enregistrement")
    n = len(records)
    columns = [_compute_single_column(f["name"], f, [r.get(f["name"]) for r in records], n)
               for f in fields]

    # ── Classification référentiel vs transactionnel au niveau table ─────
    # Référentiel : peu de lignes, peu de changements (statique)
    # Transactionnel : gros volume, insertions fréquentes
    if n < 500:
        table_class = "reference"
    elif n > 10000:
        table_class = "transactional"
    else:
        # Heuristique : si beaucoup de colonnes date → transactionnel
        date_cols = sum(1 for c in columns if c.get("semantic") in ("date_col", "iso_date", "date_fr"))
        table_class = "transactional" if date_cols >= 2 else "reference"

    return {
        "table":        entity_name,
        "total_rows":   n,
        "sampled_rows": n,
        "table_class":  table_class,
        "columns":      columns,
        "profiled_at":  datetime.utcnow().isoformat(),
        "error":        None,
    }


def _empty_profile(entity_name: str, fields: List[Dict], error: str) -> Dict:
    return {"table": entity_name, "total_rows": 0, "sampled_rows": 0,
            "columns": [], "profiled_at": datetime.utcnow().isoformat(), "error": error}


# ══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE ASYNC PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

async def profile_entity(source_id: UUID, entity_name: str,
                          sample_size: int = 5000) -> Dict:
    from .database import get_pg_pool
    pool = await get_pg_pool()

    async with pool.acquire() as conn:
        source_row = await conn.fetchrow(
            """SELECT connector_type, host, port, database_name,
                      username, options, base_url, schema_name
               FROM data_sources WHERE id=$1""", source_id)
        if not source_row:
            return _empty_profile(entity_name, [], "Source introuvable")
        source_dict = dict(source_row)

        secret_rows = await conn.fetch(
            "SELECT secret_key, secret_value FROM connection_secrets WHERE source_id=$1",
            source_id)
        for sr in secret_rows:
            if sr["secret_key"] == "password":
                source_dict["password"] = sr["secret_value"]
            else:
                source_dict[sr["secret_key"]] = sr["secret_value"]
        if "password" not in source_dict:
            source_dict["password"] = ""

        entity_row = await conn.fetchrow(
            "SELECT id FROM source_entities WHERE source_id=$1 AND name=$2",
            source_id, entity_name)
        fields = []
        if entity_row:
            rows = await conn.fetch(
                """SELECT name, data_type, is_primary_key, is_foreign_key
                   FROM entity_fields WHERE entity_id=$1 ORDER BY position""",
                entity_row["id"])
            fields = [{"name": r["name"], "type": r["data_type"] or "string",
                       "is_primary_key": r["is_primary_key"],
                       "is_foreign_key": r["is_foreign_key"]} for r in rows]

    ct = (source_dict.get("connector_type") or "").lower()

    if any(x in ct for x in ["mssql", "sql_server", "sqlserver",
                               "mysql", "postgres", "sqlite", "sql"]):
        result = await asyncio.to_thread(
            _sync_profile_sql, source_dict, entity_name, fields, sample_size)
    elif "odata" in ct:
        result = await asyncio.to_thread(
            _sync_profile_odata, source_dict, entity_name, fields, sample_size)
    elif any(x in ct for x in ["rest", "api", "http"]):
        result = await asyncio.to_thread(
            _sync_profile_rest, source_dict, entity_name, fields, sample_size)
    elif any(x in ct for x in ["csv", "excel", "file", "json"]):
        result = await asyncio.to_thread(
            _sync_profile_file, source_dict, entity_name, fields, sample_size)
        # PATCH P9 — enrichissement métadonnées Excel
        if result and not result.get("error"):
            # FIX — options peut être un string JSON (asyncpg jsonb → str)
            _opts = source_dict.get("options") or {}
            if isinstance(_opts, str):
                import json as _json
                try:
                    _opts = _json.loads(_opts)
                except Exception:
                    _opts = {}
            file_path = _opts.get("file_path") or source_dict.get("base_url", "")
            if file_path and file_path.lower().endswith((".xlsx", ".xlsm", ".xls")):
                try:
                    excel_meta = extract_excel_metadata(file_path)
                    result["excel_metadata"]  = excel_meta
                    result["formula_count"]   = sum(
                        len(v) for v in excel_meta.get("formulas", {}).values()
                    )
                    if excel_meta.get("metadata", {}).get("creator"):
                        result["file_author"] = excel_meta["metadata"]["creator"]
                except Exception as _em:
                    logger.debug(f"[excel_meta] {_em}")
    else:
        result = _empty_profile(entity_name, fields,
                                f"Connecteur '{ct}' non supporte pour le profiling")

    await _cache_profile(source_id, entity_name, result)
    return result


# ══════════════════════════════════════════════════════════════════════
# FIX P2 — PROFILING COMPLET DE TOUTE LA SOURCE (toutes les entités)
# Nouvelle fonction appelée par POST /sources/{id}/profile/all
# ══════════════════════════════════════════════════════════════════════

async def profile_source_all(
    source_id:   UUID,
    sample_size: int  = 1000,
    batch_size:  int  = 5,
    skip_errors: bool = True,
) -> Dict:
    """
    Profile toutes les entités d'une source en séquence par micro-batches.
    Reprend là où il s'est arrêté (skip entités déjà profilées sans erreur).
    Retourne un résumé progressif.
    """
    from .database import get_pg_pool
    pool = await get_pg_pool()

    async with pool.acquire() as conn:
        # Toutes les entités de la source — EXCLURE les vues SQL
        all_entities = await conn.fetch(
            """SELECT name FROM source_entities
               WHERE source_id=$1
               AND (entity_type IS NULL OR entity_type != 'view')
               ORDER BY name""",
            source_id)

        # Entités déjà profilées avec succès → on les skippe
        done_entities = await conn.fetch(
            """SELECT entity_name FROM entity_profiles
               WHERE source_id=$1
               AND profile_data->>'error' IS NULL
               AND (profile_data->>'total_rows')::int > 0""",
            source_id)

    done_set     = {r["entity_name"] for r in done_entities}
    todo         = [r["name"] for r in all_entities if r["name"] not in done_set]
    total        = len(all_entities)
    already_done = len(done_set)

    logger.info(f"[profile_all] {source_id}: {already_done} déjà profilées, {len(todo)} restantes")

    success_count = already_done
    error_count   = 0
    errors        = []

    # Traitement par micro-batches pour ne pas saturer la mémoire
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        tasks = [profile_entity(source_id, name, sample_size) for name in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(batch, results):
            if isinstance(result, Exception):
                error_count += 1
                if not skip_errors:
                    raise result
                errors.append({"entity": name, "error": str(result)})
            elif result.get("error"):
                error_count += 1
                errors.append({"entity": name, "error": result["error"]})
            else:
                success_count += 1

        logger.info(
            f"[profile_all] Progression: {success_count}/{total} "
            f"({i + len(batch)}/{len(todo)} nouvelles)"
        )

    return {
        "total":         total,
        "success":       success_count,
        "errors":        error_count,
        "error_details": errors[:20],   # max 20 erreurs remontées
        "skipped":       already_done,
        "message":       f"{success_count}/{total} entités profilées avec succès",
    }


async def profile_source_summary(source_id: UUID) -> Dict:
    from .database import get_pg_pool
    pool = await get_pg_pool()
    try:
        async with pool.acquire() as conn:
            cached = await conn.fetch(
                "SELECT entity_name, profile_data, profiled_at "
                "FROM entity_profiles WHERE source_id=$1 ORDER BY entity_name",
                source_id)
    except Exception:
        return {"profiled": 0, "total_entities": 0, "avg_quality": None, "top_issues": []}

    if not cached:
        return {"profiled": 0, "total_entities": 0, "avg_quality": None, "top_issues": []}

    profiles  = [json.loads(r["profile_data"]) if isinstance(r["profile_data"], str)
                 else r["profile_data"] for r in cached]
    all_cols  = [c for p in profiles for c in p.get("columns", [])]
    avg_q     = round(sum(c["quality_score"] for c in all_cols) / len(all_cols), 1) if all_cols else None

    issues = []
    for p in profiles:
        for c in p.get("columns", []):
            for iss in c.get("issues", []):
                issues.append({"table": p["table"], "column": c["name"], **iss})
    issues.sort(key=lambda x: {"error": 0, "warning": 1, "info": 2}.get(x["severity"], 3))

    return {"profiled": len(profiles), "total_entities": len(profiles),
            "avg_quality": avg_q, "top_issues": issues[:20],
            "profiled_at": cached[-1]["profiled_at"].isoformat() if cached else None}


async def _cache_profile(source_id: UUID, entity_name: str, profile: Dict):
    from .database import get_pg_pool
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_profiles (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_id   UUID NOT NULL,
                entity_name VARCHAR(500) NOT NULL,
                profile_data JSONB NOT NULL,
                profiled_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(source_id, entity_name)
            )""")
        await conn.execute("""
            INSERT INTO entity_profiles (source_id, entity_name, profile_data, profiled_at)
            VALUES ($1, $2, $3::jsonb, NOW())
            ON CONFLICT (source_id, entity_name)
            DO UPDATE SET profile_data=$3::jsonb, profiled_at=NOW()
        """, source_id, entity_name, json.dumps(profile))


# ══════════════════════════════════════════════════════════════════════
# FIX P1 — get_cached_profile : signature harmonisée
# Accepte (source_id, entity_name) OU (source_id, entity_name, db_conn)
# main.py l'appelait avec 3 args → crash silencieux corrigé
# ══════════════════════════════════════════════════════════════════════

async def get_cached_profile(
    source_id:   UUID,
    entity_name: str,
    db_conn=None,   # ← FIX P1 : paramètre optionnel accepté mais géré en interne
) -> Optional[Dict]:
    """
    Retourne le profil mis en cache pour une entité.
    TTL différencié selon la classification :
      - référentiel    → 24h  (faible volume, peu de changements)
      - transactionnel →  1h  (gros volume, insertions fréquentes)
      - inconnu        →  6h
    """
    from .database import get_pg_pool
    pool = await get_pg_pool()
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT profile_data, profiled_at FROM entity_profiles "
                "WHERE source_id=$1 AND entity_name=$2",
                source_id, entity_name)
            if row:
                p = (json.loads(row["profile_data"])
                     if isinstance(row["profile_data"], str)
                     else dict(row["profile_data"]))

                # ── TTL selon classification ──────────────────────────
                table_class = p.get("table_class", "unknown")
                ttl_hours = {
                    "reference":     24,   # référentiel → 24h
                    "transactional":  1,   # transactionnel → 1h
                }.get(table_class, 6)     # inconnu → 6h

                profiled_at = row["profiled_at"]
                age_hours = (datetime.utcnow() - profiled_at.replace(tzinfo=None)).total_seconds() / 3600

                if age_hours > ttl_hours:
                    logger.info(
                        f"[cache] {entity_name} expiré ({age_hours:.1f}h > {ttl_hours}h "
                        f"TTL pour class={table_class}) → recalcul"
                    )
                    return None  # Cache expiré → recalcul

                p["cached_at"]   = profiled_at.isoformat()
                p["cache_ttl_h"] = ttl_hours
                p["cache_age_h"] = round(age_hours, 1)
                return p
    except Exception as e:
        logger.warning(f"[get_cached_profile] {entity_name}: {e}")
    return None

# ══════════════════════════════════════════════════════════════════════
# PATCH P7 — Histogrammes et statistiques avancées DB
# ══════════════════════════════════════════════════════════════════════

async def get_column_histogram(
    source_id: UUID,
    entity_name: str,
    column_name: str,
) -> Dict:
    """
    Retourne l'histogramme d'une colonne depuis les stats moteur DB.
    PostgreSQL : pg_stats (nécessite ANALYZE sur la table)
    MSSQL      : sys.dm_db_stats_histogram (nécessite UPDATE STATISTICS)
    """
    from .database import get_pg_pool
    pool = await get_pg_pool()

    async with pool.acquire() as conn:
        src_row = await conn.fetchrow(
            "SELECT connector_type, host, port, database_name, username, options "
            "FROM data_sources WHERE id=$1", source_id)
    if not src_row:
        return {"error": "Source introuvable"}

    ct = (src_row["connector_type"] or "").lower()

    # ── PostgreSQL : pg_stats ─────────────────────────────────────────
    if "postgres" in ct or ct == "pg":
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT null_frac, n_distinct,
                           most_common_vals::text, most_common_freqs,
                           histogram_bounds::text, correlation, avg_width
                    FROM pg_stats
                    WHERE tablename=$1 AND attname=$2
                    LIMIT 1
                """, entity_name, column_name)
            if not row:
                return {"column": column_name,
                        "error": f"Pas de stats — lancez ANALYZE {entity_name}"}
            mcv = []
            if row["most_common_vals"] and row["most_common_freqs"]:
                try:
                    vals  = row["most_common_vals"].strip("{}").split(",")
                    freqs = row["most_common_freqs"]
                    mcv   = [{"value": v.strip('"'), "frequency": round(f, 4)}
                              for v, f in zip(vals, freqs)]
                except Exception:
                    pass
            histogram = []
            if row["histogram_bounds"]:
                try:
                    bounds = row["histogram_bounds"].strip("{}").split(",")
                    n = len(bounds) - 1
                    histogram = [{"bin_start": bounds[i], "bin_end": bounds[i+1],
                                  "frequency": round(1.0/n, 4)} for i in range(n)]
                except Exception:
                    pass
            return {
                "column":             column_name,
                "null_frac":          round(float(row["null_frac"] or 0), 4),
                "n_distinct":         float(row["n_distinct"] or 0),
                "correlation":        float(row["correlation"]) if row["correlation"] else None,
                "avg_width":          int(row["avg_width"] or 0),
                "most_common_values": mcv,
                "histogram":          histogram,
                "source":             "pg_stats",
            }
        except Exception as e:
            logger.warning(f"[histogram/pg] {entity_name}.{column_name}: {e}")
            return {"column": column_name, "error": str(e)}

    # ── MSSQL : sys.dm_db_stats_histogram ────────────────────────────
    elif any(x in ct for x in ["mssql", "sql_server", "sqlserver"]):
        try:
            import pyodbc  # type: ignore
            opts = src_row["options"] or {}
            if isinstance(opts, str):
                import json as _j; opts = _j.loads(opts)
            driver = _detect_odbc_driver()
            host   = src_row["host"]          or opts.get("host", "localhost")
            port   = src_row["port"]          or opts.get("port", 1433)
            db     = src_row["database_name"] or opts.get("database_name", "")
            user   = src_row["username"]      or opts.get("username", "")
            pwd    = opts.get("password", "")
            dsn = (f"DRIVER={{{driver}}};SERVER={host},{port};DATABASE={db};"
                   f"UID={user};PWD={pwd};TrustServerCertificate=yes;")
            query = f"""
                SELECT sh.range_hi_key, sh.range_rows, sh.eq_rows,
                       sh.distinct_range_rows, sh.avg_range_rows
                FROM sys.stats s
                INNER JOIN sys.stats_columns sc
                    ON s.object_id=sc.object_id AND s.stats_id=sc.stats_id
                INNER JOIN sys.columns c
                    ON sc.object_id=c.object_id AND sc.column_id=c.column_id
                CROSS APPLY sys.dm_db_stats_histogram(s.object_id, s.stats_id) sh
                WHERE OBJECT_NAME(s.object_id)='{entity_name}' AND c.name='{column_name}'
                ORDER BY sh.step_number
            """
            def _sync():
                with pyodbc.connect(dsn, timeout=15) as mc:
                    with mc.cursor() as cur:
                        cur.execute(query)
                        return cur.fetchall()
            rows = await asyncio.to_thread(_sync)
            return {
                "column": column_name,
                "histogram": [
                    {"bin_end": str(r[0]), "range_rows": float(r[1] or 0),
                     "eq_rows": float(r[2] or 0), "distinct_range_rows": int(r[3] or 0)}
                    for r in rows
                ],
                "source": "sys.dm_db_stats_histogram",
                "note":   "Lancez UPDATE STATISTICS pour des stats fraîches.",
            }
        except Exception as e:
            logger.warning(f"[histogram/mssql] {entity_name}.{column_name}: {e}")
            return {"column": column_name, "error": str(e)}

    return {"column": column_name, "error": f"Histogramme non supporté pour '{ct}'"}


# ══════════════════════════════════════════════════════════════════════
# PATCH P8 — Dépendances entre objets DB (views→tables, procédures, triggers)
# ══════════════════════════════════════════════════════════════════════

async def get_object_dependencies(source_id: UUID) -> Dict:
    """
    Retourne le graphe de dépendances entre objets DB.
    PostgreSQL : pg_depend + pg_rewrite
    MSSQL      : sys.sql_expression_dependencies
    """
    from .database import get_pg_pool
    pool = await get_pg_pool()

    async with pool.acquire() as conn:
        src_row = await conn.fetchrow(
            "SELECT connector_type, host, port, database_name, username, options "
            "FROM data_sources WHERE id=$1", source_id)
    if not src_row:
        return {"error": "Source introuvable", "dependencies": []}

    ct   = (src_row["connector_type"] or "").lower()
    deps = []

    # ── PostgreSQL ────────────────────────────────────────────────────
    if "postgres" in ct or ct == "pg":
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT DISTINCT
                        dep_obj.relname AS object_name,
                        CASE dep_obj.relkind
                            WHEN 'v' THEN 'view'
                            WHEN 'm' THEN 'materialized_view'
                            WHEN 'r' THEN 'table'
                        END AS object_type,
                        ref_obj.relname AS depends_on,
                        CASE ref_obj.relkind
                            WHEN 'v' THEN 'view'
                            WHEN 'm' THEN 'materialized_view'
                            WHEN 'r' THEN 'table'
                        END AS depends_on_type
                    FROM pg_depend d
                    JOIN pg_rewrite r    ON d.objid    = r.oid
                    JOIN pg_class dep_obj ON r.ev_class = dep_obj.oid
                    JOIN pg_class ref_obj ON d.refobjid  = ref_obj.oid
                    JOIN pg_namespace ns  ON dep_obj.relnamespace = ns.oid
                    WHERE ns.nspname = 'public'
                      AND dep_obj.relkind IN ('v','m')
                      AND ref_obj.relkind IN ('r','v','m')
                      AND dep_obj.relname != ref_obj.relname
                    ORDER BY dep_obj.relname, ref_obj.relname
                """)
            deps = [
                {"object_name": r["object_name"], "object_type": r["object_type"] or "view",
                 "depends_on": r["depends_on"], "depends_on_type": r["depends_on_type"] or "table",
                 "dependency_type": "normal"}
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"[dependencies/pg] {e}")

    # ── MSSQL ─────────────────────────────────────────────────────────
    elif any(x in ct for x in ["mssql", "sql_server", "sqlserver"]):
        try:
            import pyodbc  # type: ignore
            opts = src_row["options"] or {}
            if isinstance(opts, str):
                import json as _j2; opts = _j2.loads(opts)
            driver = _detect_odbc_driver()
            host   = src_row["host"]          or opts.get("host", "localhost")
            port   = src_row["port"]          or opts.get("port", 1433)
            db     = src_row["database_name"] or opts.get("database_name", "")
            user   = src_row["username"]      or opts.get("username", "")
            pwd    = opts.get("password", "")
            dsn    = (f"DRIVER={{{driver}}};SERVER={host},{port};DATABASE={db};"
                      f"UID={user};PWD={pwd};TrustServerCertificate=yes;")
            TYPE_MAP = {
                "VIEW": "view", "SQL_STORED_PROCEDURE": "procedure",
                "SQL_TRIGGER": "trigger", "SQL_SCALAR_FUNCTION": "function",
                "USER_TABLE": "table",
            }
            def _sync2():
                with pyodbc.connect(dsn, timeout=15) as mc:
                    with mc.cursor() as cur:
                        cur.execute("""
                            SELECT DISTINCT OBJECT_NAME(sed.referencing_id),
                                   o.type_desc, sed.referenced_entity_name, r.type_desc
                            FROM sys.sql_expression_dependencies sed
                            JOIN sys.objects o ON o.object_id=sed.referencing_id
                            LEFT JOIN sys.objects r ON r.name=sed.referenced_entity_name
                            WHERE sed.referenced_entity_name IS NOT NULL
                              AND o.type IN ('V','P','TR','FN','IF','TF')
                            ORDER BY 1,3
                        """)
                        return cur.fetchall()
            rows2 = await asyncio.to_thread(_sync2)
            deps  = [
                {"object_name":     r[0],
                 "object_type":     TYPE_MAP.get(r[1], r[1].lower() if r[1] else "unknown"),
                 "depends_on":      r[2],
                 "depends_on_type": TYPE_MAP.get(r[3], "table") if r[3] else "table",
                 "dependency_type": "references"}
                for r in rows2 if r[0] and r[2]
            ]
        except Exception as e:
            logger.warning(f"[dependencies/mssql] {e}")

    # Cache en PostgreSQL
    if deps:
        try:
            async with pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS object_dependencies (
                        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        source_id       UUID NOT NULL,
                        object_name     VARCHAR(255),
                        object_type     VARCHAR(50),
                        depends_on      VARCHAR(255),
                        depends_on_type VARCHAR(50),
                        dependency_type VARCHAR(50),
                        computed_at     TIMESTAMP DEFAULT NOW(),
                        UNIQUE(source_id, object_name, depends_on)
                    )
                """)
                await conn.execute("DELETE FROM object_dependencies WHERE source_id=$1", source_id)
                for dep in deps:
                    await conn.execute("""
                        INSERT INTO object_dependencies
                            (source_id,object_name,object_type,depends_on,depends_on_type,dependency_type)
                        VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT DO NOTHING
                    """, source_id, dep["object_name"], dep["object_type"],
                        dep["depends_on"], dep["depends_on_type"], dep["dependency_type"])
        except Exception as e:
            logger.warning(f"[dependencies] cache: {e}")

    logger.info(f"[dependencies] {len(deps)} dépendances pour {source_id}")
    return {"source_id": str(source_id), "total": len(deps), "dependencies": deps}


# ══════════════════════════════════════════════════════════════════════
# PATCH P9 — Métadonnées Excel avancées (formules, auteur, headers/footers)
# ══════════════════════════════════════════════════════════════════════

def extract_excel_metadata(file_path: str) -> Dict:
    """
    Extrait les métadonnées avancées d'un fichier Excel via openpyxl :
    - Formules (cellule + expression)
    - Auteur, dernière modification, titre
    - Headers et footers par feuille
    - Plages nommées
    """
    result: Dict = {
        "formulas":        {},
        "named_ranges":    [],
        "headers_footers": {},
        "merged_cells":    {},
        "metadata":        {},
    }
    try:
        import openpyxl  # type: ignore
        wb    = openpyxl.load_workbook(file_path, data_only=False)
        props = wb.properties
        result["metadata"] = {
            "creator":          getattr(props, "creator",         None),
            "last_modified_by": getattr(props, "lastModifiedBy",  None),
            "created":          str(props.created)  if props.created  else None,
            "modified":         str(props.modified) if props.modified else None,
            "title":            getattr(props, "title",           None),
            "subject":          getattr(props, "subject",         None),
            "description":      getattr(props, "description",     None),
            "keywords":         getattr(props, "keywords",        None),
            "category":         getattr(props, "category",        None),
        }
        # Plages nommées
        try:
            for dn in wb.defined_names.definedName:
                result["named_ranges"].append({
                    "name":      dn.name,
                    "refers_to": str(dn.value),
                    "scope":     dn.localSheetId,
                })
        except Exception:
            pass
        # Par feuille
        for ws in wb.worksheets:
            sname = ws.title
            formulas = []
            for row in ws.iter_rows():
                for cell in row:
                    if cell.data_type == "f" and cell.value:
                        formulas.append({
                            "cell":    cell.coordinate,
                            "formula": str(cell.value),
                            "row":     cell.row,
                            "col":     cell.column,
                        })
            if formulas:
                result["formulas"][sname] = formulas
            hf: dict = {}
            try:
                def _hf_text(obj):
                    return str(obj.text) if obj and obj.text else None
                if ws.oddHeader:
                    hf["odd_header"] = {
                        "left":   _hf_text(ws.oddHeader.left),
                        "center": _hf_text(ws.oddHeader.center),
                        "right":  _hf_text(ws.oddHeader.right),
                    }
                if ws.oddFooter:
                    hf["odd_footer"] = {
                        "left":   _hf_text(ws.oddFooter.left),
                        "center": _hf_text(ws.oddFooter.center),
                        "right":  _hf_text(ws.oddFooter.right),
                    }
            except Exception:
                pass
            if hf:
                result["headers_footers"][sname] = hf
        # Cellules fusionnées par feuille
        merged: dict = {}
        for ws in wb.worksheets:
            if ws.merged_cells.ranges:
                merged[ws.title] = [
                    str(r) for r in ws.merged_cells.ranges
                ]
        if merged:
            result["merged_cells"] = merged

        wb.close()
    except ImportError:
        result["error"] = "openpyxl non installé. pip install openpyxl"
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"[excel_metadata] {file_path}: {e}")
    return result