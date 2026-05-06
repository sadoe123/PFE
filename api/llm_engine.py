"""
OnePilot – LLM Engine §2.3.3
Génération SQL via Ollama pour requêtes complexes.
Modèle : qwen2.5-coder:3b (défaut) ou mistral si disponible.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ── Configuration Ollama ────────────────────────────────────────────────────
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))

# Mots-clés qui indiquent une requête trop complexe pour les templates
COMPLEX_KEYWORDS = [
    # Dates relatives
    "mois dernier", "semaine derniere", "trimestre", "annee derniere",
    "mois precedent", "semaine precedente",
    "last month", "last week", "last year", "last quarter",
    "entre.*et", "between.*and",
    "janvier", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "aout", "septembre", "octobre", "novembre", "decembre",
    "janvier 20", "fevrier 20", "mars 20", "avril 20", "mai 20",
    "de janvier", "de fevrier", "de mars", "d.avril", "de mai",
    "premier trimestre", "deuxieme trimestre", "troisieme trimestre",
    "du premier", "du deuxieme", "du troisieme",
    # Logique complexe
    "mais pas", "except", "sauf", "not in", "not exists",
    "qui n.ont pas", "sans", "without",
    # Comparaisons avec moyenne/total
    "superieur a la moyenne", "inferieur a la moyenne",
    "superieur au total", "inferieur au total",
    "au-dessus de la moyenne", "en-dessous de la moyenne",
    "above average", "below average",
    "greater than average", "less than average",
    "higher than average", "lower than average",
    "plus que la moyenne", "moins que la moyenne",
    "greater than the", "less than the",
    "dont le total", "dont la somme", "dont le montant",
    "whose total", "whose sum",
    # Chiffre d'affaires — nécessite JOIN Order Details
    "chiffre d.affaires", "chiffre affaires", r"\bca\b",
    "revenus", "recettes", "ventes totales", "turnover",
    # Filtres sur noms propres (ex: client ALFKI)
    "client [a-z]", "produit [a-z]", "categorie [a-z]",
    # YoY / MoM
    "year over year", "yoy", "mom", "mois sur mois",
    "evolution", "progression", "variation",
    "par rapport", "versus", "compare",
    # ES — Español
    "el mes pasado", "ultimo mes", "enero", "febrero", "marzo",
    "pero no", "excepto", "sin incluir",
    "superior al promedio", "mayor que el promedio",
    # DE — Deutsch
    "letzten monat", "januar", "februar", "marz",
    "aber nicht", "ausser", "ohne",
    "uber dem durchschnitt",
]


def is_complex_query(question: str, slots_table_names: list = None) -> bool:
    """
    Détecte si une question nécessite le LLM plutôt que les templates.
    """
    q = question.lower()
    for kw in COMPLEX_KEYWORDS:
        if re.search(kw, q):
            logger.info(f"[LLM] Requête complexe détectée : '{kw}' dans '{question[:60]}'")
            return True
    return False


def _build_sql_prompt(
    question: str,
    schema: Dict[str, List[str]],
    dialect: str = "mssql",
    table_names: List[str] = None,
) -> str:
    """
    Construit le prompt pour la génération SQL.
    """
    # Filtre le schéma aux tables pertinentes
    relevant_tables = table_names or list(schema.keys())[:5]
    schema_str = ""
    for tbl in relevant_tables[:5]:
        fields = schema.get(tbl, [])
        if fields:
            schema_str += f"Table [{tbl}]: {', '.join(fields[:15])}\n"

    dialect_hint = {
        "mssql":      "SQL Server (use TOP N, square brackets [col], GETDATE())",
        "postgresql": "PostgreSQL (use LIMIT N, CURRENT_DATE)",
        "mysql":      "MySQL (use LIMIT N, NOW())",
    }.get(dialect, "SQL Server")

    # Indique les tables prioritaires si spécifiées
    table_hint = ""
    if table_names:
        table_hint = f"\nIMPORTANT: Use ONLY these tables: {', '.join(['['+t+']' for t in table_names[:3]])}"

    prompt = f"""You are a SQL expert. Generate a {dialect_hint} query.

Database schema:
{schema_str}{table_hint}
User question: {question}

Rules:
- Return ONLY the SQL query, no explanation, no markdown, no comments
- Use proper {dialect_hint} syntax
- For SQL Server: place TOP N immediately after SELECT (e.g. SELECT TOP 100 [col] FROM [table])
- NEVER put TOP at the end of the query
- Use square brackets for ALL column/table names [column], [table]
- For date filters use DATEADD, MONTH(), YEAR(), GETDATE()
- For "superieur a la moyenne": use WHERE col > (SELECT AVG(col) FROM table)
- For "mois dernier": use WHERE col >= DATEADD(MONTH,-1,GETDATE()) AND col < GETDATE()
- For "chiffre d'affaires" or "revenus" or "CA": revenue = SUM([UnitPrice]*[Quantity]) from [Order Details], join with [Orders] and [Customers]
- For Northwind revenue: SELECT c.[CompanyName], SUM(od.[UnitPrice]*od.[Quantity]) AS Revenue FROM [Customers] c JOIN [Orders] o ON c.[CustomerID]=o.[CustomerID] JOIN [Order Details] od ON o.[OrderID]=od.[OrderID] GROUP BY c.[CompanyName]
- For specific month+year like "janvier 2024": use WHERE MONTH(col)=1 AND YEAR(col)=2024
- Month names: janvier=1,fevrier=2,mars=3,avril=4,mai=5,juin=6,juillet=7,aout=8,septembre=9,octobre=10,novembre=11,decembre=12
- Always select all relevant columns, not just one column
- Use proper JOINs between tables when needed
- The tables mentioned in the schema are the ONLY tables available - use them exactly

SQL query:"""

    return prompt


def generate_sql_with_llm(
    question: str,
    schema: Dict[str, List[str]],
    dialect: str = "mssql",
    table_names: List[str] = None,
    model: str = None,
) -> Dict:
    """
    Génère du SQL via Ollama pour les requêtes complexes.
    Retourne {sql, explanation, method, model, duration_ms}
    """
    model = model or OLLAMA_MODEL
    t0 = time.time()

    prompt = _build_sql_prompt(question, schema, dialect, table_names)

    try:
        response = httpx.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,   # Faible pour SQL déterministe
                    "top_p":       0.9,
                    "num_predict": 512,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        raw_sql = data.get("response", "").strip()

        # Nettoie le SQL — enlève les blocs markdown
        sql = _clean_sql(raw_sql)

        ms = int((time.time() - t0) * 1000)
        logger.info(f"[LLM] SQL généré en {ms}ms via {model}")

        return {
            "sql":         sql,
            "explanation": f"SQL généré par LLM ({model})",
            "method":      "llm",
            "model":       model,
            "duration_ms": ms,
            "params":      {},
            "warnings":    [],
        }

    except httpx.ConnectError:
        logger.warning(f"[LLM] Ollama non disponible sur {OLLAMA_HOST}")
        return _llm_unavailable(question)
    except httpx.TimeoutException:
        logger.warning(f"[LLM] Timeout après {OLLAMA_TIMEOUT}s")
        return _llm_unavailable(question)
    except Exception as e:
        logger.error(f"[LLM] Erreur: {e}")
        return _llm_unavailable(question)


def _clean_sql(raw: str) -> str:
    """
    Nettoie la réponse du LLM pour extraire le SQL pur.
    """
    # Enlève les blocs markdown ```sql ... ```
    raw = re.sub(r"```(?:sql|SQL)?\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)

    # Enlève les lignes de commentaires
    lines = []
    for line in raw.strip().split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("--") and not stripped.startswith("#"):
            lines.append(line)

    sql = "\n".join(lines).strip()

    # Enlève le texte avant le premier SELECT/WITH
    match = re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b", sql, re.IGNORECASE)
    if match:
        sql = sql[match.start():]

    # Fix TOP N mal positionné à la fin → déplace après SELECT
    top_end = re.search(r"TOP" + r"\s+" + r"(\d+)" + r"\s*;?\s*$", sql, re.IGNORECASE)
    if top_end:
        n = top_end.group(1)
        sql = re.sub(r"TOP" + r"\s+" + r"\d+" + r"\s*;?\s*$", "", sql, flags=re.IGNORECASE).strip()
        sql = re.sub(r"SELECT", f"SELECT TOP {n}", sql, count=1, flags=re.IGNORECASE)

    # Ajoute TOP 100 si aucun TOP ni LIMIT
    has_top = re.search(r"TOP" + r"\s+" + r"\d+", sql, re.IGNORECASE)
    has_lim = re.search(r"LIMIT" + r"\s+" + r"\d+", sql, re.IGNORECASE)
    if not has_top and not has_lim:
        sql = re.sub(r"SELECT", "SELECT TOP 100", sql, count=1, flags=re.IGNORECASE)

    return sql


def _llm_unavailable(question: str) -> Dict:
    """Retourne une erreur propre si Ollama n'est pas disponible."""
    return {
        "sql":         None,
        "explanation": "LLM non disponible — Ollama non démarré",
        "method":      "llm_unavailable",
        "model":       None,
        "duration_ms": 0,
        "params":      {},
        "warnings":    ["Ollama non disponible sur host.docker.internal:11434"],
    }


def check_ollama_available() -> Dict:
    """Vérifie si Ollama est disponible et retourne les modèles."""
    try:
        resp = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return {
            "available": True,
            "host":      OLLAMA_HOST,
            "models":    models,
            "current":   OLLAMA_MODEL,
        }
    except Exception as e:
        return {
            "available": False,
            "host":      OLLAMA_HOST,
            "models":    [],
            "error":     str(e),
        }