"""
OnePilot — Multi-Query Agent Sprint 9
Décompose les questions composées en sous-questions,
les exécute en PARALLÈLE, puis fusionne les résultats.

Exemples de questions composées supportées :
    "total transactions BNP 2024 et solde bancaire par société"
    "liste les financements actifs et les comptes associés"
    "compare les devises disponibles vs les taux de change"
    "utilisateurs bloqués et leurs sociétés associées"
    "flux de trésorerie du mois et financements à échéance"
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import asyncpg

from .orchestrator import OrchestratorResult, SubQuery, AgentType

logger = logging.getLogger(__name__)


class MultiQueryAgent:
    """
    Agent spécialisé dans les questions composées.

    Stratégie :
      1. Pour chaque sous-question → tenter DirectSQL d'abord
      2. Si pas de pattern direct → ReAct (boucle limitée à 3 iter pour la vitesse)
      3. Exécuter toutes les sous-questions en asyncio.gather() → PARALLÈLE
      4. Fusionner les résultats en un résultat unifié
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

    async def run(
        self,
        original_question: str,
        sub_questions:     List[str],
    ) -> OrchestratorResult:
        """
        Exécute les sous-questions en parallèle et fusionne.
        """
        t0 = time.time()
        logger.info(
            f"[MultiQuery] Démarrage — {len(sub_questions)} sous-questions "
            f"pour '{original_question[:60]}'"
        )

        # ── Exécution parallèle ───────────────────────────────────────────────
        tasks = [
            self._run_sub_query(sq_text, idx)
            for idx, sq_text in enumerate(sub_questions)
        ]
        sub_results: List[SubQuery] = await asyncio.gather(*tasks)

        # ── Fusion des résultats ──────────────────────────────────────────────
        fused = self._fuse_results(original_question, sub_results)
        fused.duration_ms = int((time.time() - t0) * 1000)

        logger.info(
            f"[MultiQuery] Terminé — "
            f"{sum(1 for s in sub_results if s.success)}/{len(sub_results)} succès "
            f"| {fused.duration_ms}ms"
        )
        return fused

    async def _run_sub_query(self, question: str, idx: int) -> SubQuery:
        """
        Exécute une sous-question individuelle.
        Essaie DirectSQL d'abord, puis ReAct avec max 3 itérations.
        """
        t0 = time.time()

        # ── Nettoyage : retirer les verbes introducteurs hérités de la question composée
        # Ex : "compare les devises disponibles" → "les devises disponibles"
        # Ex : "liste les financements actifs" → "financements actifs" (déjà bon)
        question = re.sub(
            r'^(compare[rz]?\s+|comparer\s+)',
            '', question.strip(), flags=re.IGNORECASE
        ).strip()

        sq = SubQuery(text=question)

        logger.info(f"[MultiQuery] Sous-question {idx+1}: '{question[:60]}'")

        try:
            from .agentic_rag import (
                _find_direct_sql,
                _tool_execute_sql,
                _tool_validate_result,
                run_agentic_rag,
            )

            # ── Tentative DirectSQL ───────────────────────────────────────────
            matched_pattern, direct_sql, match_score = _find_direct_sql(question)
            if matched_pattern and direct_sql:
                exec_result = await _tool_execute_sql(
                    direct_sql, self.source_dict, self.dialect
                )
                if exec_result.get("error"):
                    logger.warning(
                        f"[MultiQuery] SQ{idx+1} DirectSQL exec error: "
                        f"{exec_result['error'][:200]}"
                    )
                if exec_result["success"] or exec_result["row_count"] == 0:
                    sq.sql     = direct_sql
                    sq.result  = exec_result["rows"]
                    sq.success = True
                    sq.intent  = "direct_sql"
                    sq.duration_ms = int((time.time() - t0) * 1000)
                    logger.info(
                        f"[MultiQuery] SQ{idx+1} → DirectSQL "
                        f"({sq.duration_ms}ms, {exec_result['row_count']} lignes)"
                    )
                    return sq

            # ── Fallback ReAct (max 3 itérations pour la vitesse) ─────────────
            agent_result = await run_agentic_rag(
                question=question,
                source_id=self.source_id,
                pg_pool=self.pg_pool,
                source_dict=self.source_dict,
                dialect=self.dialect,
            )
            sq.sql       = agent_result.sql
            sq.result    = agent_result.result
            sq.success   = agent_result.success
            sq.intent    = "react"
            sq.duration_ms = int((time.time() - t0) * 1000)
            logger.info(
                f"[MultiQuery] SQ{idx+1} → ReAct "
                f"({agent_result.iterations} iter, {sq.duration_ms}ms)"
            )

        except Exception as e:
            sq.error       = str(e)
            sq.success     = False
            sq.duration_ms = int((time.time() - t0) * 1000)
            logger.error(f"[MultiQuery] SQ{idx+1} erreur : {e}")

        return sq

    def _fuse_results(
        self,
        original_question: str,
        sub_results:       List[SubQuery],
    ) -> OrchestratorResult:
        """
        Fusionne les résultats des sous-questions en un résultat unifié.

        Stratégie de fusion :
          - Si 1 sous-question réussie et 1 échouée → retourner la réussie avec warning
          - Si toutes réussies → retourner les deux SQLs + résultats combinés
          - Si toutes échouées → failure
        """
        successful = [sq for sq in sub_results if sq.success]
        failed     = [sq for sq in sub_results if not sq.success]
        warnings   = []

        if failed:
            for sq in failed:
                warnings.append(
                    f"Sous-question échouée: '{sq.text[:40]}' — {sq.error or 'aucun résultat'}"
                )

        if not successful:
            return OrchestratorResult(
                success=False,
                sql="",
                sqls=[],
                result=[],
                explanation="Toutes les sous-questions ont échoué",
                method="orchestrator_multi_query_failed",
                agent_type=AgentType.MULTI_QUERY,
                iterations=len(sub_results),
                warnings=warnings,
                sub_queries=sub_results,
            )

        # ── SQL principal = premier SQL réussi ────────────────────────────────
        main_sql  = successful[0].sql
        all_sqls  = [sq.sql for sq in successful]

        # ── Résultats combinés — étiquetés par sous-question ──────────────────
        combined_result = []
        for sq in successful:
            for row in sq.result:
                # Ajouter une colonne _source pour identifier l'origine
                enriched = dict(row) if isinstance(row, dict) else {"value": row}
                enriched["_source_question"] = sq.text[:50]
                combined_result.append(enriched)

        # ── Explication ───────────────────────────────────────────────────────
        explanation_parts = []
        for i, sq in enumerate(successful):
            explanation_parts.append(
                f"Q{i+1}: '{sq.text[:40]}' → {len(sq.result)} lignes ({sq.duration_ms}ms)"
            )
        explanation = " | ".join(explanation_parts)

        if warnings:
            explanation += f" | ⚠️ {len(failed)} sous-question(s) échouée(s)"

        return OrchestratorResult(
            success=True,
            sql=main_sql,
            sqls=all_sqls,
            result=combined_result,
            explanation=explanation,
            method="orchestrator_multi_query",
            agent_type=AgentType.MULTI_QUERY,
            iterations=len(sub_results),
            warnings=warnings,
            sub_queries=sub_results,
        )
