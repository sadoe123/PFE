"""
OnePilot – LLM Engine §2.3.3
Génération SQL via Ollama pour requêtes complexes.
Modèle : qwen2.5-coder:7b (défaut) ou mistral si disponible.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

import asyncpg
import httpx

logger = logging.getLogger(__name__)

# ── Import RAG Engine Sprint 7A + 7B + 7C ──────────────────────────────────
try:
    from api.rag_engine import (
        get_schema_context              as _rag_get_schema_context,
        get_schema_context_with_graph   as _rag_get_schema_context_graph,
        get_schema_context_corrective   as _rag_get_schema_context_crag,
        validate_sql_columns            as _rag_validate_sql_columns,
        is_conceptual_question          as _rag_is_conceptual,
        inject_rag_context_into_prompt,
    )
    RAG_ENABLED       = True
    GRAPH_RAG_ENABLED = True
    CRAG_ENABLED      = True
except ImportError:
    RAG_ENABLED       = False
    GRAPH_RAG_ENABLED = False
    CRAG_ENABLED      = False


# ── Configuration Ollama ────────────────────────────────────────────────────
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
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


# ── Blacklist tables infrastructure ────────────────────────────────────────
INFRA_TABLE_PREFIXES = (
    "QRTZ_", "qrtz_", "sys", "SYS", "dt_", "DT_",
    "MSreplication", "msreplication", "sysdiagram", "__",
)

def _is_infra_table(name: str) -> bool:
    return any(name.startswith(p) for p in INFRA_TABLE_PREFIXES)


def _build_sql_prompt(
    question: str,
    schema: Dict[str, List[str]],
    dialect: str = "mssql",
    table_names: List[str] = None,
    rag_context: dict = None,
) -> str:
    """
    Construit le prompt pour la generation SQL.
    Fix 1 : Blacklist tables infra (QRTZ_*, sys*, ...)
    Fix 2 : Crochets obligatoires pour noms avec espaces
    Fix 3 : Fallback si aucune table metier trouvee
    """
    relevant_tables = [
        t for t in (table_names or list(schema.keys())[:5])
        if not _is_infra_table(t)
    ]

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

    table_hint = ""
    if relevant_tables:
        table_hint = f"\nIMPORTANT: Use ONLY these tables: {', '.join(['['+t+']' for t in relevant_tables[:3]])}"

    # ── Vues SXA métier disponibles (toujours injectées si MSSQL) ──────────
    SXA_VIEWS_HINT = """
=== VUES METIER SXA (à utiliser en priorité pour les questions business) ===
Vue [Comptes]                        : ID, CODE, DESCRIPTION, Banque, Société, Devises, Groupe_Sociétés, Groupe_de_comptes
Vue [Transactions bancaires]         : CODE, DESCRIPTION, Banque, Société, AMOUNT, [montant avec signe], CUR_ID_TRNCURRENCY, TRNDATE, VALUEDATE, ISDEBIT, Statut
Vue [Dernière integration bancaire]  : CODE, Banque, Société, Devises, CLOSINGBALANCEAMOUNT, CLOSINGBALANCEDATETIME, MontantAvecSigne
Vue [SI_Trésorerie]                  : CODE, DESCRIPTION, Banque, Société, AMOUNTI, ISDEBITI, TRNDATE
Vue [SI_Bancaire]                    : CODE, DESCRIPTION, Banque, Société, AMOUNTI, ISDEBITI, TRNDATE
Vue [Journal]                        : CODE, DESCRIPTION, Sous_Catégorie_flux, Catégorie_flux, [Montant ABS], ISDEBIT, [Date de transaction], [date de valeur], Statut, [Montant_avec_signe], Compte, Banque, Société, Devises
Vue [FINANCEMENT_BI]                 : TRN_ID, [Date début], [Date fin], Montant, [Devises financement], type_transaction, maturité, type, état, compte, Banque, Société
Vue [Ligne de financement]           : TRN_ID, CODE, DESCRIPTION, AMOUNT, Comptes, Banque, Société, SUBCATEGORY
Vue [cours marchés]                  : CUR_ID_CURRFROM, CUR_ID_CURRTO, ASK, RATEDATE
Vue [Tableaux d'amortissement]       : SCHDL_ID, compte, AMORTIZATION, INTEREST, PAYMENTDATE, Société, Banque
⚠️ RÈGLE ABSOLUE : Ces vues existent RÉELLEMENT dans la base SXA. Utilise-les directement.
   Pour les questions sur les comptes/banques/sociétés → utilise [Comptes]
   Pour les transactions → utilise [Transactions bancaires]
   Pour la trésorerie → utilise [SI_Trésorerie]
   Pour les soldes bancaires → utilise [Dernière integration bancaire]
   Pour les flux de trésorerie → utilise [Journal]
   Pour les financements → utilise [FINANCEMENT_BI]
   INTERDIT d'inventer des tables qui ne sont pas listées ci-dessus ou dans le schéma RAG.
"""

    rag_block = ""
    no_business_table = False
    if rag_context and rag_context.get("context_text"):
        rag_tables = rag_context.get("tables_found", [])
        business_tables = [t for t in rag_tables if not _is_infra_table(t)]
        if business_tables:
            rag_block = (
                "\n=== SCHEMA RAG (" + str(len(business_tables)) + " tables metier) ===\n"
                + rag_context["context_text"]
                + "\nIMPORTANT: Use ONLY tables from RAG schema above.\n"
                + "NEVER use infrastructure tables (QRTZ_*, sys*, dt_*) in SQL.\n"
                + "NEVER invent table names not listed in the schema.\n"
            )
        else:
            no_business_table = True
            logger.warning(f"[LLM] Aucune table metier dans RAG (tables: {rag_tables})")

    if no_business_table and not schema_str.strip():
        return (
            f'You are a SQL expert assistant.\n'
            f'The user asked: "{question}"\n\n'
            f'IMPORTANT: No relevant business table was found. Do NOT generate SQL. '
            f'Respond EXACTLY in French:\n'
            f'"Je ne trouve pas de donnees correspondant a cette demande dans le schema disponible."'
        )

    # Injecter les vues SXA TOUJOURS pour MSSQL — même quand le RAG a trouvé des tables
    # Le LLM doit toujours préférer les vues métier aux tables brutes
    _view_hint_block = SXA_VIEWS_HINT if dialect == "mssql" else ""

    prompt = f"""You are a SQL expert. Generate a {dialect_hint} query.
{rag_block}
{_view_hint_block}
Database schema (fallback):
{schema_str}{table_hint}
User question: {question}

Rules:
- Return ONLY the SQL query, no explanation, no markdown, no comments
- Use proper {dialect_hint} syntax
- For SQL Server: place TOP N immediately after SELECT (e.g. SELECT TOP 100 [col] FROM [table])
- NEVER put TOP at the end of the query
- CRITICAL: Use square brackets for ALL column and table names, especially names with spaces.
  Examples: [montant avec signe], [Transactions bancaires], [Derniere integration bancaire]
  NEVER write: montant avec signe (sans crochets)
- NEVER use infrastructure tables: QRTZ_*, sys*, dt_*, MSreplication*, sysdiagram*
- If question is not answerable with available tables, respond in French:
  "Je ne trouve pas de donnees correspondant a cette demande dans le schema disponible."
- CRITICAL FOR SXA VIEWS — apply these mappings STRICTLY, NO exceptions:
  * "solde trésorerie" / "trésorerie" / "solde par société" → ONLY use [SI_Trésorerie]: SELECT [Société], SUM([AMOUNTI]) AS Solde FROM [SI_Trésorerie] GROUP BY [Société]
  * "flux bancaires" / "flux par compte" / "mouvements bancaires" → ONLY use [Transactions bancaires] or [Journal]
  * "transactions bancaires" / "transactions" → ONLY use [Transactions bancaires]
  * "soldes bancaires" / "dernière intégration" / "closing balance" → ONLY use [Dernière integration bancaire]
  * "comptes avec banque" / "comptes et société" → ONLY use [Comptes]
  * "financement" / "crédit" / "emprunt" → ONLY use [FINANCEMENT_BI]
  * NEVER JOIN [SI_Trésorerie] with other tables — it already has Société, Banque, AMOUNTI
  * NEVER JOIN [Transactions bancaires] with other tables — it already has Banque, Société, AMOUNT
  * NEVER JOIN [Comptes] with other tables — it already has Banque, Société, Devises
- For date filters use DATEADD, MONTH(), YEAR(), GETDATE()
- For "superieur a la moyenne": use WHERE col > (SELECT AVG(col) FROM table)
- For "mois dernier": use WHERE col >= DATEADD(MONTH,-1,GETDATE()) AND col < GETDATE()
- For "chiffre d'affaires" or "revenus" or "CA": revenue = SUM([UnitPrice]*[Quantity]) from [Order Details]
- For specific month+year like "janvier 2024": use WHERE MONTH(col)=1 AND YEAR(col)=2024
- Month names: janvier=1,fevrier=2,mars=3,avril=4,mai=5,juin=6,juillet=7,aout=8,septembre=9,octobre=10,novembre=11,decembre=12
- Always select all relevant columns, not just one column
- Use proper JOINs between tables when needed
- The tables mentioned in the schema are the ONLY tables available

SQL query:"""

    return prompt


# ── Sprint 13 : A/B Testing — Prompt B (few-shot enrichi) ──────────────────

async def _get_few_shot_examples(pg_pool, source_id: str = None, limit: int = 5) -> list:
    """
    Charge les exemples validés (👍) depuis chat_feedback pour le few-shot.
    Filtre sur source_id si fourni — sinon prend les plus récents global.
    Retourne une liste de dicts {question, sql} si le SQL est disponible
    dans nlu_query_log.
    """
    if pg_pool is None:
        return []
    try:
        from uuid import UUID as _UUID
        query = """
            SELECT cf.question, nql.sql_generated
            FROM   chat_feedback cf
            LEFT JOIN nlu_query_log nql
                   ON nql.conversation_id = cf.conversation_id
                  AND nql.question        = cf.question
            WHERE  cf.feedback_type     = 'like'
              AND  cf.used_for_training = TRUE
              AND  cf.question          != ''
              AND  nql.sql_generated    IS NOT NULL
              AND  nql.sql_generated    != ''
        """
        params = []
        if source_id:
            try:
                query += " AND cf.source_id = $1 ORDER BY cf.created_at DESC LIMIT $2"
                params = [_UUID(source_id), limit]
            except Exception:
                query += " ORDER BY cf.created_at DESC LIMIT $1"
                params = [limit]
        else:
            query += " ORDER BY cf.created_at DESC LIMIT $1"
            params = [limit]

        async with pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [{"question": r["question"], "sql": r["sql_generated"]} for r in rows]
    except Exception as e:
        logger.debug(f"[AB Testing] Few-shot fetch error: {e}")
        return []


def _build_sql_prompt_b(
    question: str,
    schema: Dict[str, List[str]],
    dialect: str = "mssql",
    table_names: List[str] = None,
    rag_context: dict = None,
    few_shot_examples: list = None,
) -> str:
    """
    Prompt B — enrichi avec exemples few-shot tirés des questions validées (👍).
    Identique au Prompt A + bloc few-shot injecté avant la question.
    """
    # Récupérer le prompt A comme base
    prompt_a = _build_sql_prompt(question, schema, dialect, table_names, rag_context)

    # Construire le bloc few-shot
    if not few_shot_examples:
        return prompt_a  # fallback sur A si aucun exemple

    few_shot_block = "\n=== EXEMPLES VALIDÉS PAR LES UTILISATEURS (few-shot) ===\n"
    few_shot_block += "Ces exemples ont été validés comme corrects — reproduis ce style SQL.\n\n"
    for i, ex in enumerate(few_shot_examples[:5], 1):
        q = ex.get("question", "").strip()
        s = ex.get("sql", "").strip()
        if q and s:
            # Tronquer le SQL si trop long
            s_short = s[:300] + "..." if len(s) > 300 else s
            few_shot_block += f"Exemple {i}:\n  Question: {q}\n  SQL: {s_short}\n\n"
    few_shot_block += "=== FIN DES EXEMPLES ===\n"

    # Injecter le bloc few-shot juste avant "User question:"
    if "User question:" in prompt_a:
        prompt_b = prompt_a.replace(
            f"User question: {question}",
            f"{few_shot_block}User question: {question}",
            1
        )
    else:
        prompt_b = few_shot_block + prompt_a

    return prompt_b


async def generate_sql_with_llm(
    question: str,
    schema: Dict[str, List[str]],
    dialect: str = "mssql",
    table_names: List[str] = None,
    model: str = None,
    pg_pool=None,
    source_id: str = None,
) -> Dict:
    """
    Génère du SQL via Ollama pour les requêtes complexes.
    Sprint 7A : injecte le contexte RAG dans le prompt.
    """
    model = model or OLLAMA_MODEL
    t0 = time.time()

    # Sprint 7A + 7B + 7C — RAG + Graph RAG + Corrective RAG
    rag_context = None
    rag_info    = {"method": "disabled", "tables": [], "graph_tables": []}

    if RAG_ENABLED and pg_pool and source_id:
        try:
            from uuid import UUID
            sid = UUID(source_id)

            # ── Sprint 7C : Corrective RAG (sub-querying + routing) ──────────
            if CRAG_ENABLED:
                rag_context = await _rag_get_schema_context_crag(question, sid, pg_pool)

                # Routing : question conceptuelle → pas de SQL
                if rag_context.get("is_conceptual"):
                    logger.info(f"[CRAG] Routing → réponse texte (question conceptuelle)")
                    return {
                        "sql":          "",
                        "explanation":  "question_conceptuelle",
                        "method":       "crag_conceptual",
                        "model":        model or OLLAMA_MODEL,
                        "duration_ms":  0,
                        "params":       {},
                        "warnings":     [],
                        "is_conceptual": True,
                        "question":     question,
                    }

            # ── Sprint 7B : Graph RAG fallback ───────────────────────────────
            elif GRAPH_RAG_ENABLED:
                rag_context = await _rag_get_schema_context_graph(question, sid, pg_pool)
            # ── Sprint 7A : RAG standard fallback ────────────────────────────
            else:
                rag_context = await _rag_get_schema_context(question, sid, pg_pool)

            rag_info = {
                "method":       rag_context.get("method", ""),
                "tables":       rag_context.get("tables_found", []),
                "graph_tables": rag_context.get("graph_new_tables", []),
                "join_paths":   len(rag_context.get("graph_join_paths", [])),
                "sub_queries":  rag_context.get("sub_queries", []),
            }
            logger.info(
                f"[LLM+CRAG] sub_queries={rag_context.get('sub_queries',[])}"
                f" | seed={rag_context.get('graph_seed_tables',[])}"
                f" | graph={rag_context.get('graph_new_tables',[])}"
                f" | joins={len(rag_context.get('graph_join_paths',[]))}"
            )
        except Exception as e:
            logger.warning(f"[LLM+CRAG] RAG échoué : {e}")

    prompt = _build_sql_prompt(question, schema, dialect, table_names, rag_context=rag_context)

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

        # ── Sprint 7C : SQL Column Validator ─────────────────────────────────
        # Vues SXA métier — colonnes validées manuellement dans SSMS, exclues du CRAG
        _SXA_BUSINESS_VIEWS = frozenset([
            "Transactions bancaires", "Comptes", "SI_Trésorerie", "SI_Bancaire",
            "Journal", "FINANCEMENT_BI", "Dernière integration bancaire",
            "cours marchés", "Ligne de financement", "Tableaux d'amortissement",
            "VDTSSXACOMPANYRIGHT", "TH_USR",
        ])
        import re as _re_sxa
        _sql_tables_used = set(_re_sxa.findall(
            r'FROM\s+\[([^\]]+)\]|JOIN\s+\[([^\]]+)\]', sql or "", _re_sxa.IGNORECASE
        ))
        _sql_tables_flat = {t for pair in _sql_tables_used for t in pair if t}
        _uses_sxa_view = bool(_sql_tables_flat & _SXA_BUSINESS_VIEWS)

        validation_result = {}
        col_warnings = []
        if CRAG_ENABLED and pg_pool and source_id and sql and not _uses_sxa_view:
            try:
                from uuid import UUID as _UUID
                validation_result = await _rag_validate_sql_columns(
                    sql, _UUID(source_id), pg_pool
                )
                if not validation_result.get("valid", True):
                    invalid = validation_result.get("invalid_columns", [])
                    suggestions = validation_result.get("suggestions", {})
                    for col in invalid[:5]:
                        suggestion = suggestions.get(col, "")
                        warn = f"Colonne '{col}' non trouvée dans le schéma"
                        if suggestion:
                            warn += f" → suggestion : '{suggestion}'"
                        col_warnings.append(warn)
                    logger.warning(
                        f"[CRAG] SQL invalide : colonnes inexistantes {invalid}"
                    )
            except Exception as e:
                logger.warning(f"[CRAG] validate_sql_columns error: {e}")
        elif _uses_sxa_view:
            logger.info(f"[CRAG] Validation ignorée — vue SXA métier détectée: {_sql_tables_flat & _SXA_BUSINESS_VIEWS}")

        method = "crag" if CRAG_ENABLED and rag_context else ("llm+rag" if rag_context and rag_context.get("table_count") else "llm")

        return {
            "sql":              sql,
            "explanation":      f"SQL généré par LLM ({model})",
            "method":           method,
            "model":            model,
            "duration_ms":      ms,
            "params":           {},
            "warnings":         col_warnings,
            "validation":       validation_result,
            "sub_queries":      rag_context.get("sub_queries", []) if rag_context else [],
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

    # ── Fix DISTINCT + TOP — incompatibles en MSSQL ─────────────────────────
    # SELECT TOP N DISTINCT  → invalide  |  SELECT DISTINCT TOP N → invalide
    # Si DISTINCT présent → supprimer TOP N (DISTINCT déduplique déjà)
    if re.search(r'\bDISTINCT\b', sql, re.IGNORECASE):
        sql = re.sub(r'\bSELECT\b\s+TOP\s+\d+\s+DISTINCT\b',
                     'SELECT DISTINCT', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bSELECT\b\s+DISTINCT\s+TOP\s+\d+\b',
                     'SELECT DISTINCT', sql, flags=re.IGNORECASE)


    # ── Fix tables _A (audit) → table courante — recherche globale ───────────
    # Cherche TOUTES les occurrences TABLE_A dans le SQL entier
    _all_audit = re.findall(
        r'\b([A-Za-z][A-Za-z0-9_]+_A)\b(?=[\s\.\],]|$)',
        sql, re.IGNORECASE
    )
    _seen_audit = set()
    for _audit_tbl in _all_audit:
        if _audit_tbl.upper() in _seen_audit:
            continue
        _seen_audit.add(_audit_tbl.upper())
        _current_tbl = _audit_tbl[:-2]
        sql = re.sub(
            rf'\b{re.escape(_audit_tbl)}\b',
            _current_tbl,
            sql, flags=re.IGNORECASE
        )


    # ── Fix WHERE mal positionné après GROUP BY ──────────────────────────────
    # MSSQL : WHERE doit être AVANT GROUP BY
    _grp_where = re.search(
        r'(GROUP\s+BY\s+[^\n]+)\n(\s*WHERE\s+[^\n]+)',
        sql, re.IGNORECASE
    )
    if _grp_where:
        where_clause = _grp_where.group(2).strip()
        group_clause = _grp_where.group(1).strip()
        sql = sql.replace(
            _grp_where.group(0),
            where_clause + '\n' + group_clause
        )

    # ── Fix LIMIT → TOP (MySQL/PG invalide en SQL Server) ────────────────────
    _lim = re.search(r'\bLIMIT\s+(\d+)\b', sql, re.IGNORECASE)
    if _lim and not re.search(r'\bTOP\s+\d+\b', sql, re.IGNORECASE):
        n = _lim.group(1)
        sql = re.sub(r'\bLIMIT\s+\d+\b', '', sql, flags=re.IGNORECASE).strip()
        sql = re.sub(r'\bSELECT\b', f'SELECT TOP {n}', sql, count=1, flags=re.IGNORECASE)
    elif _lim:
        sql = re.sub(r'\bLIMIT\s+\d+\b', '', sql, flags=re.IGNORECASE).strip()

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



# ── Streaming LLM §2.3.3 — Sprint Point 2 ───────────────────────────────────

from typing import AsyncGenerator

async def generate_sql_stream(
    question: str,
    schema: "Dict[str, List[str]]",
    dialect: str = "mssql",
    table_names: "List[str]" = None,
    model: str = None,
    pg_pool=None,
    source_id: str = None,
    ab_variant: str = "A",
    few_shot_examples: list = None,
) -> "AsyncGenerator[Dict, None]":
    """
    Génère du SQL via Ollama en mode streaming (token par token).
    Yield des dicts typés :
      {"type": "thinking", "content": "..."}     — phase préparation/RAG
      {"type": "token",    "content": "SELECT"}  — token SQL brut du LLM
      {"type": "sql",      "content": "SELECT TOP 100 ..."}  — SQL final nettoyé
      {"type": "error",    "content": "..."}     — erreur
      {"type": "done",     "duration_ms": 1234}  — fin du stream
    """
    import time as _time
    model = model or OLLAMA_MODEL
    t0 = _time.time()

    # ── Phase 1 : CRAG + Graph RAG context ──────────────────────────────────
    rag_context = None
    if RAG_ENABLED and pg_pool and source_id:
        yield {"type": "thinking", "content": "🔍 Recherche dans le schéma RAG + Graph…"}
        try:
            from uuid import UUID as _UUID
            sid = _UUID(source_id)

            # Sprint 7C : Corrective RAG
            if CRAG_ENABLED:
                rag_context = await _rag_get_schema_context_crag(question, sid, pg_pool)

                # Routing : question conceptuelle → pas de SQL
                if rag_context.get("is_conceptual"):
                    yield {"type": "thinking", "content": "💬 Question conceptuelle → réponse texte"}
                    yield {
                        "type":          "conceptual",
                        "content":       question,
                        "is_conceptual": True,
                    }
                    yield {"type": "done", "duration_ms": 0}
                    return

                sub_queries = rag_context.get("sub_queries", [])
                if len(sub_queries) > 1:
                    yield {"type": "thinking", "content": f"🔀 Sub-queries : {len(sub_queries)} → {', '.join(sub_queries[:2])}…"}

            elif GRAPH_RAG_ENABLED:
                rag_context = await _rag_get_schema_context_graph(question, sid, pg_pool)
            else:
                rag_context = await _rag_get_schema_context(question, sid, pg_pool)

            tables_found = rag_context.get("tables_found", [])
            graph_new    = rag_context.get("graph_new_tables", [])
            join_count   = len(rag_context.get("graph_join_paths", []))
            if tables_found:
                yield {"type": "thinking", "content": f"📋 Tables seed : {', '.join(tables_found[:3])}"}
            if graph_new:
                yield {"type": "thinking", "content": f"🔗 Tables FK (Graph) : {', '.join(graph_new[:3])} | {join_count} JOIN paths"}

        except Exception as _e:
            logger.warning(f"[LLM Stream+CRAG] RAG échoué : {_e}")

    yield {"type": "thinking", "content": "\u2699\ufe0f Génération SQL…"}

    # ── Sprint 13 : A/B Testing — sélection prompt ─────────────────────────
    if ab_variant == "B" and few_shot_examples:
        prompt = _build_sql_prompt_b(
            question, schema, dialect, table_names,
            rag_context=rag_context,
            few_shot_examples=few_shot_examples,
        )
        logger.info(f"[AB Stream] Prompt B — {len(few_shot_examples)} exemples few-shot")
    else:
        prompt = _build_sql_prompt(question, schema, dialect, table_names, rag_context=rag_context)
        logger.info("[AB Stream] Prompt A — baseline")

    # ── Phase 2 : Streaming Ollama ───────────────────────────────────────
    raw_tokens = []
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as _client:
            async with _client.stream(
                "POST",
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model":  model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 512,
                    },
                },
            ) as _response:
                _response.raise_for_status()
                async for _line in _response.aiter_lines():
                    if not _line.strip():
                        continue
                    try:
                        _chunk = json.loads(_line)
                    except Exception:
                        continue
                    _token = _chunk.get("response", "")
                    if _token:
                        raw_tokens.append(_token)
                        yield {"type": "token", "content": _token}
                    if _chunk.get("done", False):
                        break

    except httpx.ConnectError:
        logger.warning(f"[LLM Stream] Ollama non disponible sur {OLLAMA_HOST}")
        yield {"type": "error", "content": "⚠️ Ollama non disponible — LLM hors ligne."}
        return
    except httpx.TimeoutException:
        logger.warning(f"[LLM Stream] Timeout après {OLLAMA_TIMEOUT}s")
        yield {"type": "error", "content": f"⏱️ Timeout LLM après {OLLAMA_TIMEOUT}s."}
        return
    except Exception as _ex:
        logger.error(f"[LLM Stream] Erreur: {_ex}")
        yield {"type": "error", "content": f"❌ Erreur LLM : {_ex}"}
        return

    # ── Phase 3 : SQL final nettoyé ──────────────────────────────────────
    raw_sql = "".join(raw_tokens).strip()
    sql_clean = _clean_sql(raw_sql)
    _ms = int((_time.time() - t0) * 1000)

    _method = "llm+rag" if rag_context and rag_context.get("table_count") else "llm"
    yield {"type": "sql",  "content": sql_clean, "method": _method, "model": model}
    yield {"type": "done", "duration_ms": _ms}


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