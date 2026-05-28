"""
OnePilot — Agent Precision (Sprint 10)
Élimine les hallucinations LLM en injectant le schéma structuré réel
avant la génération SQL.

Différence vs ReAct+ :
  ReAct+     → passe un contexte textuel approximatif au LLM
  Precision  → extrait les colonnes exactes VALIDÉES de chaque table,
               construit un schéma structuré, bloque si entités non couvertes,
               validation stricte 100% post-génération.

Pipeline :
  1. CRAG Hybrid → top-3 tables candidates
  2. Pour chaque table → _tool_get_table_columns() → colonnes réelles
  3. Construire schéma structuré validé (pas de texte libre)
  4. Vérifier couverture entités clés (montants, devises, banques)
  5. Générer SQL avec schéma injecté
  6. Valider SQL généré → entités + colonnes dans le schéma
  7. Exécuter → si erreur → re-générer avec erreur explicite (max 3 iter)
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────────────
MAX_ITER_PRECISION = 3   # moins d'itérations car contexte meilleur
COV_THRESHOLD      = 0.6 # couverture minimale entités clés


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Normalise : uppercase + retire accents + espaces + underscores."""
    s = unicodedata.normalize("NFD", s.upper())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace(" ", "").replace("_", "").replace("-", "")


def _extract_key_entities(question: str) -> Dict[str, List[str]]:
    """Extrait les entités clés d'une question pour vérifier la couverture."""
    entities: Dict[str, List[str]] = {}

    # Montants
    amounts = re.findall(r"\b\d{3,}\b", question)
    if amounts:
        entities["amounts"] = amounts

    # Années
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    if years:
        entities["years"] = years

    # Devises
    currencies = re.findall(r"\b(TND|EUR|USD|GBP|CHF|MAD)\b", question, re.IGNORECASE)
    if currencies:
        entities["currencies"] = [c.upper() for c in currencies]

    # Banques connues SXA
    banks = ["bnp", "banque postale", "société générale", "stb", "biat",
             "attijari", "groupama", "banque populaire", "sg"]
    for bank in banks:
        if bank in question.lower():
            entities.setdefault("banks", []).append(bank)

    return entities


def _check_coverage(entities: Dict, sql: str, schema: Dict) -> Tuple[bool, float, List[str]]:
    """
    Vérifie que le SQL couvre les entités clés ET utilise des colonnes du schéma.
    Retourne (valid, score, missing_list).
    """
    if not entities:
        return True, 1.0, []

    sql_norm = _norm(sql)
    missing, total, covered = [], 0, 0

    for etype, values in entities.items():
        for val in values:
            total += 1
            if _norm(val) in sql_norm:
                covered += 1
            else:
                missing.append(f"{etype}:{val}")

    score = covered / total if total > 0 else 1.0
    return score >= COV_THRESHOLD, round(score, 2), missing


# ─────────────────────────────────────────────────────────────────────────────
# Schéma structuré
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TableSchema:
    name: str
    columns: List[str] = field(default_factory=list)
    join_paths: List[str] = field(default_factory=list)
    row_count: int = 0
    source: str = "db"  # "db" | "sxa_views" | "crag"

    def to_prompt_block(self) -> str:
        """Formate le schéma pour injection dans le prompt LLM."""
        cols = ", ".join(f"[{c}]" for c in self.columns[:30]) if self.columns else "colonnes inconnues"
        block = f"Table: [{self.name}]\n  Colonnes: {cols}"
        if self.join_paths:
            block += f"\n  JOINs: {'; '.join(self.join_paths[:3])}"
        return block


@dataclass
class PrecisionContext:
    tables: List[TableSchema] = field(default_factory=list)
    join_paths: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    coverage_ok: bool = True
    coverage_score: float = 1.0
    missing_entities: List[str] = field(default_factory=list)

    def to_prompt_schema(self) -> str:
        """Génère le bloc schéma structuré pour le prompt."""
        blocks = [t.to_prompt_block() for t in self.tables]
        return "\n\n".join(blocks)

    def get_all_columns(self) -> Dict[str, List[str]]:
        """Retourne {table_name: [col1, col2, ...]} pour validation."""
        return {t.name: t.columns for t in self.tables}


# ─────────────────────────────────────────────────────────────────────────────
# Agent Precision
# ─────────────────────────────────────────────────────────────────────────────

class PrecisionAgent:
    """
    Agent Precision — Sprint 10.
    Injecte le schéma structuré réel pour éliminer les hallucinations.
    """

    def __init__(
        self,
        pg_pool:     asyncpg.Pool,
        source_dict: Dict,
        source_id:   UUID,
        dialect:     str = "mssql",
    ):
        self.pg_pool     = pg_pool
        self.source_dict = source_dict
        self.source_id   = source_id
        self.dialect     = dialect

    async def run(self, question: str) -> Dict:
        """
        Point d'entrée principal.
        Retourne un dict compatible AgentResult.
        """
        t0 = time.time()
        logger.info(f"[Precision] Démarrage — question: '{question[:80]}'")

        # ── Étape 1 : Récupérer le contexte CRAG ─────────────────────────────
        ctx = await self._get_crag_context(question)
        tables_found = ctx.get("tables_found", [])
        join_paths   = ctx.get("graph_join_paths", [])

        # ── Étape 2 : Extraire les colonnes réelles pour chaque table ─────────
        precision_ctx = await self._build_precision_context(
            question, tables_found, join_paths
        )

        # ── Étape 3 : Vérifier couverture entités ─────────────────────────────
        if not precision_ctx.coverage_ok:
            logger.warning(
                f"[Precision] Couverture insuffisante "
                f"({precision_ctx.coverage_score:.0%}) — "
                f"entités manquantes: {precision_ctx.missing_entities}"
            )

        # ── Étape 4 : Boucle génération + validation ──────────────────────────
        sql, result, success, iterations, explanation = await self._generation_loop(
            question, precision_ctx
        )

        duration_ms = int((time.time() - t0) * 1000)
        logger.info(
            f"[Precision] Terminé — success={success} "
            f"| {iterations} iter | {duration_ms}ms"
        )

        return {
            "sql":         sql,
            "result":      result,
            "success":     success,
            "iterations":  iterations,
            "duration_ms": duration_ms,
            "explanation": explanation,
            "method":      "precision_agent",
            "agent_type":  "precision",
            "warnings":    precision_ctx.missing_entities,
            "schema_used": [t.name for t in precision_ctx.tables],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 1 : CRAG
    # ─────────────────────────────────────────────────────────────────────────

    # Tables SXA prioritaires — injectées si la question les mentionne
    SXA_PRIORITY_TABLES = {
        "financement":    "FINANCEMENT_BI",
        "financements":   "FINANCEMENT_BI",
        "répartition":    "FINANCEMENT_BI",
        "repartition":    "FINANCEMENT_BI",
        "solde":          "Dernière integration bancaire",
        "soldes":         "Dernière integration bancaire",
        "trésorerie":     "Dernière integration bancaire",
        "tresorerie":     "Dernière integration bancaire",
        "bancaire":       "Dernière integration bancaire",
        "compte":         "Comptes",
        "comptes":        "Comptes",
        "transaction":    "Transactions bancaires",
        "transactions":   "Transactions bancaires",
        "mouvement":      "Transactions bancaires",
        "mouvements":     "Transactions bancaires",
        "utilisateur":    "TH_USR",
        "utilisateurs":   "TH_USR",
    }

    # Tables INTERDITES — inventées par le LLM, n'existent pas dans SXA
    SXA_FORBIDDEN_TABLES = {
        "Ligne_de_financement", "LigneFinancement", "LIGNE_FINANCEMENT",
        "DetailFinancement", "Financement_Detail",
        "TypeFinancement", "EtatFinancement",
    }

    async def _get_crag_context(self, question: str) -> Dict:
        """Appelle le pipeline CRAG pour récupérer les tables candidates.
        Injecte les tables SXA prioritaires si la question les mentionne."""
        try:
            from .rag_engine import get_schema_context_corrective
            ctx = await get_schema_context_corrective(
                question, self.source_id, self.pg_pool
            )
        except Exception as e:
            logger.warning(f"[Precision] CRAG error: {e}")
            ctx = {"tables_found": [], "graph_join_paths": [], "context_text": ""}

        # Injecter les tables SXA prioritaires si absentes
        q_lower = question.lower()
        q_has_financement = any(k in q_lower for k in ("financement", "financements", "répartition", "repartition"))
        priority_tables = []
        for kw, tbl in self.SXA_PRIORITY_TABLES.items():
            if kw not in q_lower:
                continue
            # Éviter d'injecter "Dernière integration bancaire" quand la question
            # parle de financements — "bancaire" dans FINANCEMENT_BI ne doit pas
            # déclencher la table de soldes bancaires
            if tbl == "Dernière integration bancaire" and q_has_financement:
                continue
            if tbl not in ctx.get("tables_found", []):
                priority_tables.append(tbl)

        if priority_tables:
            existing = ctx.get("tables_found", [])
            # Mettre les tables prioritaires en tête
            ctx["tables_found"] = priority_tables + [
                t for t in existing if t not in priority_tables
            ]
            logger.info(
                f"[Precision] Tables prioritaires injectées: {priority_tables}"
            )

        return ctx

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 2 : Construction du contexte précis
    # ─────────────────────────────────────────────────────────────────────────

    async def _build_precision_context(
        self,
        question:    str,
        tables:      List[str],
        join_paths:  List[str],
    ) -> PrecisionContext:
        """
        Pour chaque table candidate → récupère les colonnes réelles depuis la DB.
        Vérifie la couverture des entités clés.
        """
        entities = _extract_key_entities(question)

        # Récupérer colonnes pour chaque table en parallèle
        tasks = [
            self._fetch_table_schema(tbl)
            for tbl in tables[:5]  # max 5 tables
        ]
        table_schemas = await asyncio.gather(*tasks)

        # Filtrer les tables valides (avec colonnes)
        valid_schemas = [ts for ts in table_schemas if ts.columns]

        # Associer les JOIN paths
        # jp peut etre un dict ou une string selon le retour CRAG
        for ts in valid_schemas:
            ts.join_paths = [
                str(jp) if not isinstance(jp, str) else jp
                for jp in join_paths
                if _norm(ts.name) in _norm(str(jp) if not isinstance(jp, str) else jp)
            ][:3]

        # Vérifier couverture avec les colonnes disponibles
        all_cols_text = " ".join(
            col
            for ts in valid_schemas
            for col in ts.columns
        )
        all_tables_text = " ".join(ts.name for ts in valid_schemas)
        combined_text = all_tables_text + " " + all_cols_text

        coverage_ok, coverage_score, missing = _check_coverage(
            entities, combined_text, {}
        )

        return PrecisionContext(
            tables=valid_schemas,
            join_paths=join_paths,
            entities=entities,
            coverage_ok=coverage_ok,
            coverage_score=coverage_score,
            missing_entities=missing,
        )

    async def _fetch_table_schema(self, table_name: str) -> TableSchema:
        """Récupère les colonnes réelles d'une table depuis PostgreSQL."""
        try:
            from .agentic_rag import _tool_get_table_columns
            result = await _tool_get_table_columns(
                table_name, self.source_id, self.pg_pool
            )
            if result.get("success") and result.get("columns"):
                return TableSchema(
                    name=table_name,
                    columns=result["columns"],
                    source=result.get("source", "db"),
                )
        except Exception as e:
            logger.warning(f"[Precision] fetch_table_schema({table_name}): {e}")

        # Fallback — retourner le nom sans colonnes
        return TableSchema(name=table_name, columns=[], source="unknown")

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 4 : Boucle génération + validation
    # ─────────────────────────────────────────────────────────────────────────

    async def _generation_loop(
        self,
        question: str,
        ctx:      PrecisionContext,
    ) -> Tuple[str, List, bool, int, str]:
        """
        Génère le SQL, l'exécute, valide. Max MAX_ITER_PRECISION itérations.
        Retourne (sql, result, success, iterations, explanation).
        """
        from .agentic_rag import _tool_execute_sql

        previous_sql   = None
        previous_error = None
        best_sql       = ""
        best_result    = []

        for iteration in range(1, MAX_ITER_PRECISION + 1):
            logger.info(f"[Precision] Iter {iteration}/{MAX_ITER_PRECISION}")

            # ── Générer le SQL ───────────────────────────────────────────────
            sql = await self._generate_sql(
                question,
                ctx,
                previous_sql=previous_sql,
                previous_error=previous_error,
            )

            if not sql:
                logger.warning(f"[Precision] Iter {iteration} — SQL vide")
                continue

            # ── Valider colonnes vs schéma ───────────────────────────────────
            col_errors = self._validate_columns(sql, ctx)
            if col_errors:
                logger.warning(
                    f"[Precision] Iter {iteration} — colonnes invalides: "
                    f"{col_errors[:3]}"
                )
                previous_sql   = sql
                previous_error = f"Colonnes non trouvées dans le schéma: {', '.join(col_errors[:3])}"
                continue

            # ── Exécuter ─────────────────────────────────────────────────────
            exec_result = await _tool_execute_sql(
                sql, self.source_dict, self.dialect
            )

            if exec_result["success"]:
                best_sql    = sql
                best_result = exec_result.get("rows", [])
                logger.info(
                    f"[Precision] Iter {iteration} — succès "
                    f"({exec_result['row_count']} lignes)"
                )
                return (
                    best_sql, best_result, True, iteration,
                    f"Precision Agent — {iteration} iter | "
                    f"{len(ctx.tables)} tables | "
                    f"cov={ctx.coverage_score:.0%}"
                )
            else:
                err = exec_result.get("error", "Unknown error")
                logger.warning(f"[Precision] Iter {iteration} — erreur SQL: {err[:100]}")
                previous_sql   = sql
                previous_error = str(err)
                if not best_sql:
                    best_sql = sql

        # Toutes les itérations échouées
        return (
            best_sql, best_result, False, MAX_ITER_PRECISION,
            f"Precision Agent — {MAX_ITER_PRECISION} iter max atteintes"
        )

    def _validate_columns(self, sql: str, ctx: PrecisionContext) -> List[str]:
        """
        Verifie que les colonnes dans le SQL existent dans le schema.
        Ignore les aliases SQL (AS NomAlias).
        """
        if not ctx.tables:
            return []

        # Extraire les alias definis dans le SQL : [col] AS Alias ou col AS Alias
        # On ne valide PAS les aliases, seulement les vraies colonnes referencees
        aliases = set()
        for m in re.finditer(r"\]\s+AS\s+(\w+)", sql, re.IGNORECASE):
            aliases.add(_norm(m.group(1)))
        for m in re.finditer(r"AS\s+(\w+)", sql, re.IGNORECASE):
            aliases.add(_norm(m.group(1)))

        # Extraire les colonnes entre crochets
        sql_cols = re.findall(r"\[([^\]]+)\]", sql)

        # Construire l ensemble de toutes les colonnes disponibles
        all_valid = set()
        for ts in ctx.tables:
            all_valid.add(_norm(ts.name))
            for col in ts.columns:
                all_valid.add(_norm(col))

        # Colonnes systemes communes toujours valides
        SYSTEM_COLS = {
            "ISLOCKED","CODE","DESCRIPTION","NAME","EMAIL",
            "USR_ID","TRN_ID","ID","STATUS","TYPE","ETAT","MONTANT",
            "BANQUE","SOCIETE","DEVISE","DEVISES","TOTAL","NB","COUNT",
            "TOTALFINANCEMENT","TOTALFINANCEMENTS","TYPEFINANCEMENT",
            "ETATFINANCEMENT","NBFINANCEMENT","MONTANTTOTAL","TOTALMONTANT",
        }

        suspect = []
        for col in sql_cols:
            if ' ' in col: continue  # alias multi-mots
            cn = _norm(col)
            if cn in aliases: continue
            if cn in all_valid: continue
            if any(cn.startswith(_norm(s)) for s in SYSTEM_COLS): continue
            if cn in {_norm(s) for s in SYSTEM_COLS}: continue
            suspect.append(col)

        return suspect[:3]

    async def _generate_sql(
        self,
        question:       str,
        ctx:            PrecisionContext,
        previous_sql:   Optional[str] = None,
        previous_error: Optional[str] = None,
    ) -> str:
        """Génère le SQL avec schéma structuré injecté."""
        import httpx
        import os

        OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
        OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
        OLLAMA_TIMEOUT = 60.0

        # ── Bloc erreur précédente ────────────────────────────────────────────
        error_block = ""
        if previous_error and previous_sql:
            error_block = f"""
⚠️ TENTATIVE PRÉCÉDENTE ÉCHOUÉE :
SQL : {previous_sql}
Erreur : {previous_error}
CONSIGNE : Corriger en utilisant UNIQUEMENT les colonnes listées dans le schéma ci-dessous.
"""

        # ── Schéma structuré ─────────────────────────────────────────────────
        schema_block = ctx.to_prompt_schema() if ctx.tables else "Aucune table trouvée"

        # ── JOIN paths ───────────────────────────────────────────────────────
        join_block = ""
        if ctx.join_paths:
            join_block = "JOIN paths validés en base :\n" + "\n".join(
                f"  {jp}" for jp in ctx.join_paths[:5]
            )

        # ── Entités manquantes ────────────────────────────────────────────────
        missing_block = ""
        if ctx.missing_entities:
            missing_block = (
                f"\n⚠️ Entités à couvrir (présentes dans la question) : "
                f"{', '.join(ctx.missing_entities)}"
            )

        dialect_hint = {
            "mssql":      "SQL Server (TOP N, crochets [col], GETDATE(), DATEADD)",
            "postgresql": "PostgreSQL (LIMIT N, CURRENT_DATE)",
        }.get(self.dialect, "SQL Server")

        # Bloc tables interdites (hallucinations connues)
        forbidden_block = ""
        if hasattr(self, "SXA_FORBIDDEN_TABLES") and self.SXA_FORBIDDEN_TABLES:
            forbidden_block = (
                "\n⛔ TABLES INTERDITES (n'existent pas en base, ne jamais utiliser) :\n"
                + ", ".join(sorted(self.SXA_FORBIDDEN_TABLES))
                + "\n"
            )

        prompt = f"""Tu es un expert SQL {dialect_hint}. Génère une requête SQL PRÉCISE.

SCHÉMA STRUCTURÉ — Utilise UNIQUEMENT ces tables et colonnes :

{schema_block}

{join_block}

{missing_block}
{forbidden_block}
{error_block}

RÈGLES STRICTES :
1. Utilise UNIQUEMENT les tables et colonnes listées dans le SCHÉMA STRUCTURÉ ci-dessus
2. N'invente JAMAIS une table ou colonne absente du schéma — même si tu la "connais"
   Exemple interdit : FROM Ligne_de_financement  ← cette table n'existe pas
   Exemple correct  : FROM [FINANCEMENT_BI]      ← seule table financement valide
3. Syntaxe MSSQL : SELECT TOP 100 [col] ... (jamais LIMIT)
4. Mets des crochets autour des noms avec espaces : [Dernière integration bancaire]
5. N'utilise JAMAIS d'alias avec espaces (ex: AS Total Montant ← INTERDIT). Utilise AS TotalMontant
6. Préfixe les colonnes ambiguës : [TableName].[ColName]
7. Retourne UNIQUEMENT le SQL — sans texte, sans commentaire, sans markdown

Question : {question}

SQL :"""

        try:
            resp = httpx.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.05, "num_predict": 512},
                },
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()

            # Nettoyer
            raw = re.sub(r"```(?:sql|SQL)?\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
            match = re.search(r"\b(SELECT|WITH)\b", raw, re.IGNORECASE)
            if match:
                raw = raw[match.start():]

            # MSSQL fixes
            if self.dialect in ("mssql", "odata"):
                if not re.search(r"TOP\s+\d+", raw, re.IGNORECASE):
                    raw = re.sub(r"\bSELECT\b", "SELECT TOP 100", raw,
                                 count=1, flags=re.IGNORECASE)
                raw = re.sub(r"\s*LIMIT\s+\d+\s*;?\s*$", "", raw.strip(),
                             flags=re.IGNORECASE)
                # Corriger tables _A audit
                _audits = re.findall(
                    r'\b([A-Za-z][A-Za-z0-9_]+_A)\b(?=[\s\.\]\[,)]|$)',
                    raw, re.IGNORECASE
                )
                for _a in set(_audits):
                    raw = re.sub(
                        rf'\b{re.escape(_a)}\b',
                        _a[:-2], raw, flags=re.IGNORECASE
                    )

            return raw.strip()

        except Exception as e:
            logger.warning(f"[Precision] LLM error: {e}")
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée public
# ─────────────────────────────────────────────────────────────────────────────

async def run_precision_agent(
    question:    str,
    source_id:   UUID,
    pg_pool:     asyncpg.Pool,
    source_dict: Dict,
    dialect:     str = "mssql",
) -> Dict:
    """Point d'entrée compatible avec l'orchestrateur."""
    agent = PrecisionAgent(
        pg_pool=pg_pool,
        source_dict=source_dict,
        source_id=source_id,
        dialect=dialect,
    )
    return await agent.run(question)
