"""
OnePilot — Orchestrateur Sprint 9
Multi-Agent RAG : coordination intelligente des agents spécialisés.

Architecture :
    Question
        ↓
    Orchestrateur.run()
        ├── Étape 1 : Règles déterministes (< 1ms)
        │       ├── question composée ? (ET / VS / comparaison)
        │       ├── filtres dynamiques ? (_has_dynamic_filters)
        │       └── pattern direct ? (_find_direct_sql)
        ├── Étape 2 : LLM si ambigu (~1s)
        │       └── classification intention
        │
        └── Dispatch vers l'agent approprié :
                ├── DirectSQL    → Sprint 8.5/8.6 (< 500ms)
                ├── MultiQuery   → Sprint 9 NOUVEAU (parallèle)
                ├── Retrieval    → Sprint 7A/7B/7C CRAG évolué
                └── ReAct+       → Sprint 8 enrichi

Agents :
    - DirectSQLAgent   : patterns connus → bypass LLM
    - MultiQueryAgent  : questions composées → décomposition + parallèle + fusion
    - RetrievalAgent   : CRAG évolué avec seuil qualité + re-query
    - ReActPlusAgent   : ReAct enrichi avec contexte cumulatif + validation stricte
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TYPES ET ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class AgentType(str, Enum):
    DIRECT_SQL   = "direct_sql"
    MULTI_QUERY  = "multi_query"
    REACT_PLUS   = "react_plus"
    PRECISION    = "precision"
    UNKNOWN      = "unknown"


@dataclass
class SubQuery:
    """Une sous-question extraite d'une question composée."""
    text:        str
    intent:      str = "generate_sql"   # generate_sql | aggregate | filter
    sql:         str = ""
    result:      List[Dict] = field(default_factory=list)
    success:     bool = False
    duration_ms: int = 0
    error:       str = ""


@dataclass
class OrchestratorResult:
    """Résultat unifié retourné par l'orchestrateur."""
    success:      bool
    sql:          str                          # SQL principal (ou premier SQL si multi)
    sqls:         List[str] = field(default_factory=list)  # tous les SQLs si multi-query
    result:       List[Dict] = field(default_factory=list)
    explanation:  str = ""
    method:       str = ""                    # agent utilisé
    agent_type:   AgentType = AgentType.UNKNOWN
    iterations:   int = 1
    duration_ms:  int = 0
    warnings:     List[str] = field(default_factory=list)
    sub_queries:  List[SubQuery] = field(default_factory=list)  # si multi-query


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTEURS — Règles déterministes (Étape 1)
# ─────────────────────────────────────────────────────────────────────────────

# Séparateurs de questions composées
_COMPOSITE_PATTERNS = [
    # Conjonctions explicites
    r'\bet\b(?:\s+(?:le|la|les|l\'|aussi|également|de plus))?\s+(?:le|la|les|l\'|affiche|montre|liste|donne|total|solde|nombre)',
    r'\bainsi\s+que\b',
    r'\bde\s+plus\b',
    r'\ben\s+même\s+temps\b',
    r'\bpar\s+ailleurs\b',
    # Comparaisons
    r'\bvs\.?\b',
    r'\bversus\b',
    r'\bcompare[rz]?\b',
    r'\bcomparaison\b',
    r'\bdifférence\s+entre\b',
    r'\bcontraste\b',
    # Listes de données différentes
    r'\bliste[rz]?\s+.{3,40}\s+et\s+(?:aussi\s+)?(?:les?|la|le)\b',
    r'\baffiche[rz]?\s+.{3,40}\s+et\s+(?:aussi\s+)?(?:les?|la|le)\b',
    # Patterns "X et leurs Y" / "X et les Y associes" — Sprint 13
    r'\bet\s+leurs?\s+\w+',
    r'\bet\s+les\s+\w+\s+associe',
    r'\bsolde\s+.{3,30}\s+et\s+total\b',
    r'\bsolde\s+.{3,30}\s+et\s+(?:les?|la|le)\b',
]

_COMPOSITE_RE = [re.compile(p, re.IGNORECASE) for p in _COMPOSITE_PATTERNS]

# Séparateurs forts — toujours une question composée
_STRONG_SEPARATORS = re.compile(
    r'\b(et\s+(?:aussi|également|en\s+plus|leurs?\s+\w+)|'
    r'ainsi\s+que|'
    r'en\s+plus\s+de\s+(?:ça|cela)|'
    r'vs\.?|versus|'
    r'comparer?|comparaison\s+entre)\b',
    re.IGNORECASE
)


def detect_composite_question(question: str) -> Tuple[bool, List[str]]:
    """
    Détecte si une question est composée de plusieurs sous-questions.

    Stratégie :
      1. Séparateurs forts (vs, versus, comparer) → toujours composée
      2. Patterns regex → probablement composée
      3. Décomposition heuristique en sous-questions

    Returns:
        (is_composite: bool, sub_questions: List[str])
    """
    q = question.strip()

    # ── Séparateurs forts ────────────────────────────────────────────────────
    strong = _STRONG_SEPARATORS.search(q)
    if strong:
        parts = _split_composite(q, strong.group(0))
        if len(parts) >= 2:
            return True, parts

    # ── Patterns composés ────────────────────────────────────────────────────
    for pattern_re in _COMPOSITE_RE:
        if pattern_re.search(q):
            parts = _split_by_et(q)
            if len(parts) >= 2 and all(len(p.split()) >= 2 for p in parts):
                return True, parts

    return False, [q]


def _split_composite(question: str, separator: str) -> List[str]:
    """Découpe une question sur un séparateur fort — supporte N sous-questions."""
    sep_lower = separator.lower().strip()
    q_lower   = question.lower()

    # Cas : séparateur "vs" ou "versus" → split récursif sur TOUS les "vs"
    # Supporte N comparaisons : "BNP vs SG vs Groupama vs BPM" → 4 parties
    if sep_lower in ("vs", "vs.", "versus"):
        raw_parts = re.split(r'\s+vs\.?\s+', question, flags=re.IGNORECASE)
        parts = [p.strip().rstrip(',').strip() for p in raw_parts if len(p.strip().split()) >= 1]
        if len(parts) >= 2:
            return parts
        return [question]

    # Cas : "comparer X vs Y [vs Z...]" → split récursif sur tous les "vs"
    if sep_lower.startswith("compar"):
        raw_parts = re.split(r'\s+vs\.?\s+', question, flags=re.IGNORECASE)
        if len(raw_parts) >= 2:
            # Retirer le préfixe "compare" de la première partie
            first = re.sub(r'^comparer?\s+', '', raw_parts[0], flags=re.IGNORECASE).strip()
            rest  = [p.strip() for p in raw_parts[1:]]
            parts = [p for p in [first] + rest if len(p.split()) >= 1]
            if len(parts) >= 2:
                return parts
        return [question]

    idx = q_lower.find(sep_lower)
    if idx == -1:
        return [question]
    part1 = question[:idx].strip().rstrip(',').strip()
    part2 = question[idx + len(separator):].strip()
    return [p for p in [part1, part2] if len(p.split()) >= 2]
def _split_by_et(question: str) -> List[str]:
    """
    Découpe une question sur 'et' en vérifiant que les deux parties
    sont des questions valides (au moins 3 tokens chacune).
    """
    # Chercher ' et ' entouré de contexte substantiel
    parts = re.split(r'\s+et\s+(?=(?:le|la|les|leurs?|l\'|affiche|montre|liste|donne|total|solde|nombre|les\s+\w))', question, flags=re.IGNORECASE)
    if len(parts) >= 2:
        return [p.strip() for p in parts if len(p.split()) >= 2]
    return [question]


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATEUR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Cerveau central de OnePilot Sprint 9.
    Analyse la question et dispatch vers l'agent le plus approprié.
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

    async def run(self, question: str, nlu_intent: str = "") -> OrchestratorResult:
        """
        Point d'entrée principal.
        Analyse → Route → Exécute → Retourne.

        Args:
            question   : question en langage naturel
            nlu_intent : intent détecté par le NLU en amont (optionnel).
                         Si llm_explain → l'Orchestrateur refuse de générer du SQL.
                         list_fields / describe_entity sont autorisés (peuvent donner du SQL utile).
        """
        t0 = time.time()
        logger.info(f"[Orchestrator] Démarrage — question: '{question[:80]}'")

        # ── Guard intent NLU : si non-SQL → refus immédiat ──────────────────────
        # Evite que l'Orchestrateur génère du SQL pour des questions conceptuelles
        # même si un pattern DirectSQL matche un mot-clé de la question.
        # Seul llm_explain est non-SQL garanti.
        # greeting/help retirés : confiance faible (0.5-0.6) peut faussement
        # classifier des questions courtes comme "utilisateurs bloques".
        _NON_SQL_INTENTS_ORCH = {"llm_explain"}
        if nlu_intent in _NON_SQL_INTENTS_ORCH:
            logger.info(
                f"[Orchestrator] Refus — intent NLU={nlu_intent} "
                f"(non-SQL) : '{question[:60]}'"
            )
            return OrchestratorResult(
                success=False,
                sql="",
                explanation=(
                    f"Cette question est de nature explicative (intent={nlu_intent}). "
                    f"Elle sera traitée par le module LLM_EXPLAIN."
                ),
                method="refused_non_sql_intent",
                agent_type=AgentType.UNKNOWN,
            )

        from .agentic_rag import (
            _find_direct_sql,
            _has_dynamic_filters,
            _tool_execute_sql,
            _tool_validate_result,
            AgentResult,
            SXA_DIRECT_SQL as _sds_orch,
        )

        # ── Étape 0A PRIORITÉ ABSOLUE : Question composée → MultiQuery ──────
        # DOIT être avant DirectSQL pour éviter que "compare BNP vs SG vs Groupama"
        # soit intercepté par le pattern partiel "financements bnp".
        # Exception : si la question ENTIÈRE matche exactement un pattern DirectSQL
        # (ex: "compare utilisateurs actifs vs bloques" a un pattern dédié).
        import unicodedata as _ud_orch, re as _re_orch
        _q_norm_orch = _ud_orch.normalize('NFD', question.lower().strip())
        _q_norm_orch = ''.join(c for c in _q_norm_orch if _ud_orch.category(c) != 'Mn')
        _COMP_ORCH = _re_orch.compile(
            r'\b(vs\.?|versus|comparer?)\b|\bet\s+(?:leurs?|les\s+\w+\s+associe)',
            _re_orch.IGNORECASE
        )
        _is_comp_orch = bool(_COMP_ORCH.search(question))

        # Chercher un match EXACT (≥ 70% de couverture ET cohérence composé)
        _exact_direct_orch = None
        _exact_pat_orch = None
        if _is_comp_orch:
            for _pe_o, _se_o in sorted(_sds_orch.items(), key=lambda x: -len(x[0])):
                _pe_norm_o = _ud_orch.normalize('NFD', _pe_o.lower())
                _pe_norm_o = ''.join(c for c in _pe_norm_o if _ud_orch.category(c) != 'Mn')
                _ratio_o = len(_pe_norm_o) / max(len(_q_norm_orch), 1)
                _pat_comp_o = bool(_COMP_ORCH.search(_pe_norm_o))
                if _pe_norm_o == _q_norm_orch:
                    _exact_direct_orch = _se_o
                    _exact_pat_orch = _pe_o
                    break
                elif _pe_norm_o in _q_norm_orch and _ratio_o >= 0.70 and _pat_comp_o:
                    _exact_direct_orch = _se_o
                    _exact_pat_orch = _pe_o
                    break

        if _is_comp_orch and _exact_direct_orch:
            # Pattern exact composé trouvé → DirectSQL direct
            logger.info(
                f"[Orchestrator] Route → DirectSQL exact composé "
                f"(pattern='{_exact_pat_orch}')"
            )
            result = await self._run_direct_sql(
                question, _exact_direct_orch, _exact_pat_orch, 1.0, t0,
                _tool_execute_sql, _tool_validate_result
            )
            result.duration_ms = int((time.time() - t0) * 1000)
            return result
        elif _is_comp_orch:
            # Question composée SANS pattern exact → MultiQuery
            is_composite, sub_questions = detect_composite_question(question)
            if is_composite and len(sub_questions) >= 2:
                logger.info(
                    f"[Orchestrator] Question composée détectée — "
                    f"{len(sub_questions)} sous-questions : {sub_questions}"
                )
                result = await self._run_multi_query(question, sub_questions, t0)
                result.duration_ms = int((time.time() - t0) * 1000)
                return result

        # ── Étape 0B PRIORITÉ ABSOLUE : Pattern direct connu → bypass tout ──
        matched_pattern, direct_sql, match_score = _find_direct_sql(question)
        if matched_pattern and direct_sql:
            logger.info(
                f"[Orchestrator] Route → DirectSQL PRIORITÉ "
                f"(pattern='{matched_pattern}', score={match_score:.2f})"
            )
            result = await self._run_direct_sql(
                question, direct_sql, matched_pattern, match_score, t0,
                _tool_execute_sql, _tool_validate_result
            )
            result.duration_ms = int((time.time() - t0) * 1000)
            return result

        # ── Étape 0C : Question complexe → Agent Precision ──────────────────
        is_complex = _is_complex_question(question)
        if is_complex:
            logger.info(f"[Orchestrator] Route → Precision (question complexe)")
            result = await self._run_precision(question, t0)
            result.duration_ms = int((time.time() - t0) * 1000)
            return result

        # ── Étape 1A : Question composée (non vs/compare) ────────────────────
        is_composite, sub_questions = detect_composite_question(question)
        if is_composite and len(sub_questions) >= 2:
            logger.info(
                f"[Orchestrator] Question composée détectée — "
                f"{len(sub_questions)} sous-questions : {sub_questions}"
            )
            result = await self._run_multi_query(question, sub_questions, t0)
            result.duration_ms = int((time.time() - t0) * 1000)
            return result

        # ── Étape 1B : Pattern direct (fallback après composite) ─────────────
        matched_pattern, direct_sql, match_score = _find_direct_sql(question)
        if matched_pattern and direct_sql:
            logger.info(
                f"[Orchestrator] Route → DirectSQL "
                f"(pattern='{matched_pattern}', score={match_score:.2f})"
            )
            result = await self._run_direct_sql(
                question, direct_sql, matched_pattern, match_score, t0,
                _tool_execute_sql, _tool_validate_result
            )
            result.duration_ms = int((time.time() - t0) * 1000)
            return result

        # ── Étape 1C : Filtres dynamiques → ReAct+ ───────────────────────────
        has_dynamic, dynamic_reasons = _has_dynamic_filters(question)
        if has_dynamic:
            logger.info(
                f"[Orchestrator] Route → ReAct+ "
                f"(filtres dynamiques={dynamic_reasons})"
            )
        else:
            logger.info(f"[Orchestrator] Route → ReAct+ (question inconnue)")

        result = await self._run_react_plus(question, t0)
        result.duration_ms = int((time.time() - t0) * 1000)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Agent DirectSQL
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_direct_sql(
        self,
        question:       str,
        sql:            str,
        pattern:        str,
        score:          float,
        t0:             float,
        execute_fn,
        validate_fn,
    ) -> OrchestratorResult:
        """Exécute un SQL direct validé — bypass LLM total."""
        exec_result = await execute_fn(sql, self.source_dict, self.dialect)

        if exec_result["success"]:
            validation = validate_fn(exec_result["rows"], question, sql)
            if validation["valid"] or exec_result["row_count"] == 0:
                return OrchestratorResult(
                    success=True,
                    sql=sql,
                    sqls=[sql],
                    result=exec_result["rows"],
                    explanation=f"Requête directe optimisée (pattern='{pattern}', score={score:.2f})",
                    method="orchestrator_direct_sql",
                    agent_type=AgentType.DIRECT_SQL,
                    iterations=1,
                )
        # Fallback → ReAct+
        logger.warning(f"[Orchestrator] DirectSQL échoué — fallback ReAct+")
        return await self._run_react_plus(question, t0)

    # ─────────────────────────────────────────────────────────────────────────
    # Agent MultiQuery
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_multi_query(
        self,
        question:      str,
        sub_questions: List[str],
        t0:            float,
    ) -> OrchestratorResult:
        """
        Exécute plusieurs sous-questions en PARALLÈLE puis fusionne.
        C'est la nouveauté principale du Sprint 9.
        """
        from .multi_query_agent import MultiQueryAgent
        agent = MultiQueryAgent(
            pg_pool=self.pg_pool,
            source_dict=self.source_dict,
            source_id=self.source_id,
            dialect=self.dialect,
        )
        return await agent.run(question, sub_questions)

    # ─────────────────────────────────────────────────────────────────────────
    # Agent Precision (Sprint 10)
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_precision(
        self,
        question: str,
        t0:       float,
    ) -> OrchestratorResult:
        """
        Agent Precision Sprint 10 : schema structure reel -> 0 hallucination.
        """
        from .precision_agent import run_precision_agent

        prec_result = await run_precision_agent(
            question=question,
            source_id=self.source_id,
            pg_pool=self.pg_pool,
            source_dict=self.source_dict,
            dialect=self.dialect,
        )

        return OrchestratorResult(
            success=prec_result["success"],
            sql=prec_result["sql"],
            sqls=[prec_result["sql"]] if prec_result["sql"] else [],
            result=prec_result.get("result", []),
            explanation=prec_result.get("explanation", ""),
            method=f"orchestrator_precision",
            agent_type=AgentType.PRECISION,
            iterations=prec_result.get("iterations", 1),
            warnings=prec_result.get("warnings", []),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Agent ReAct+
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_react_plus(
        self,
        question: str,
        t0:       float,
    ) -> OrchestratorResult:
        """
        ReAct enrichi Sprint 9 :
          - CRAG avec seuil qualité (re-query si score < -3.0)
          - Contexte cumulatif entre itérations
          - Validation stricte entités (Option C évolué)
          - Colonnes exactes injectées dès la 1ère erreur
        """
        from .agentic_rag import run_agentic_rag, AgentResult

        agent_result: AgentResult = await run_agentic_rag(
            question=question,
            source_id=self.source_id,
            pg_pool=self.pg_pool,
            source_dict=self.source_dict,
            dialect=self.dialect,
        )

        return OrchestratorResult(
            success=agent_result.success,
            sql=agent_result.sql,
            sqls=[agent_result.sql] if agent_result.sql else [],
            result=agent_result.result,
            explanation=agent_result.explanation,
            method=f"orchestrator_react_plus({agent_result.method})",
            agent_type=AgentType.REACT_PLUS,
            iterations=agent_result.iterations,
            warnings=agent_result.warnings,
        )



# ─────────────────────────────────────────────────────────────────────────────
# Detection question complexe -> Agent Precision
# ─────────────────────────────────────────────────────────────────────────────

def _is_complex_question(question: str) -> bool:
    """
    Detecte si une question necessite l'Agent Precision.
    Criteres : jointures multi-tables, agregations complexes,
               analyses croisees sans pattern direct.
    """
    import re as _re
    q = question.lower()

    COMPLEX_KW = [
        "par banque et par devise", "par devise et par banque",
        "par banque et par type", "par type et par banque",
        "par banque et par societe", "par societe et par banque",
        "par type et par etat", "par etat et par type",
        "par type de transaction et", "et par type de transaction",
        "par banque et", "et par banque",
        "par devise et", "et par devise",
        "evolution de", "tendance des", "projection de",
        "superieur a la moyenne", "inferieur a la moyenne",
        "superieur a", "inferieur a",
        "depasse", "depassent", "au-dessus de",
        "fois la moyenne", "fois le montant", "fois la valeur",
        "solde bancaire depasse", "solde depasse",
        "financement depasse", "montant depasse",
        "par rapport au", "en proportion",
        "les plus", "les moins",
        "avec leurs", "associes a", "lies a",
        "evolution mensuelle", "cumul annuel",
        "par trimestre", "sur les 12 derniers",
        "total et nombre", "montant et compte",
        "somme et moyenne", "repartition et total",
        "nombre de financements par",
        "repartition des financements par",
        "evolution des financements par",
        "repartition des", "analyse de la repartition",
        "analyse des", "ventilation des",
    ]
    for kw in COMPLEX_KW:
        import unicodedata as _ud
        def _norm(s):
            s = _ud.normalize('NFD', s.lower())
            return ''.join(c for c in s if _ud.category(c) != 'Mn')
        if _norm(kw) in _norm(q):
            return True

    has_bank     = any(b in q for b in ["bnp", "societe generale", "postale", "stb", "biat"])
    has_currency = bool(_re.search(r"\b(tnd|eur|usd)\b", q))
    has_year     = bool(_re.search(r"\b20\d{2}\b", q))
    has_multigroup = any(k in q for k in [
        "par banque et", "par devise et", "par type et",
        "par societe et", "et par banque", "et par devise",
        "et par type", "et par societe",
    ])

    if sum([has_bank, has_currency, has_year]) >= 2:
        return True
    if has_multigroup:
        return True
    if has_year:
        agg_kw = ["total des", "somme des", "moyenne des", "nombre de",
                  "repartition des", "evolution des", "par devise", "par banque",
                  "par type", "par societe"]
        if any(k in q for k in agg_kw):
            return True

    return False

# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PUBLIC
# ─────────────────────────────────────────────────────────────────────────────

async def run_orchestrator(
    question:    str,
    source_id:   UUID,
    pg_pool:     asyncpg.Pool,
    source_dict: Dict,
    dialect:     str = "mssql",
    nlu_intent:  str = "",
) -> OrchestratorResult:
    """
    Point d'entrée public — appelé depuis main.py en remplacement de run_agentic_rag.

    Args:
        question    : question en langage naturel
        source_id   : UUID de la source de données
        pg_pool     : pool PostgreSQL (métadonnées OnePilot)
        source_dict : config de la source
        dialect     : dialecte SQL (mssql, postgresql, mysql, odata)
        nlu_intent  : intent NLU détecté en amont (évite SQL sur questions explicatives)

    Returns:
        OrchestratorResult avec sql, result, method, sub_queries si multi
    """
    orchestrator = Orchestrator(
        pg_pool=pg_pool,
        source_dict=source_dict,
        source_id=source_id,
        dialect=dialect,
    )
    return await orchestrator.run(question, nlu_intent=nlu_intent)


def orchestrator_result_to_dict(result: OrchestratorResult) -> Dict:
    """Sérialise OrchestratorResult en dict JSON-compatible pour l'API."""
    return {
        "success":     result.success,
        "sql":         result.sql,
        "sqls":        result.sqls,
        "result":      result.result,
        "explanation": result.explanation,
        "method":      result.method,
        "agent_type":  result.agent_type.value,
        "iterations":  result.iterations,
        "duration_ms": result.duration_ms,
        "warnings":    result.warnings,
        "sub_queries": [
            {
                "text":        sq.text,
                "intent":      sq.intent,
                "sql":         sq.sql,
                "success":     sq.success,
                "duration_ms": sq.duration_ms,
                "error":       sq.error,
            }
            for sq in result.sub_queries
        ],
    }
