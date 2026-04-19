"""
OnePilot – Preference Learner §2.3.4
Apprentissage des préférences utilisateur depuis l'historique des choix.
- Scoring de probabilité pour chaque interprétation
- Apprentissage par slot_key + valeur choisie
- Fallback sur règles métier prédéfinies
- Vocabulaire vocal : "deux cents" → "200"
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# RÈGLES MÉTIER PRÉDÉFINIES (fallback)
# ══════════════════════════════════════════════════════════════════════

BUSINESS_RULES: Dict[str, Dict] = {
    # Champs montant : priorité HT sur TTC
    "amount_field": {
        "keywords_ht":  ["ht", "excl", "hors taxe", "before tax", "net"],
        "keywords_ttc": ["ttc", "incl", "toutes taxes", "with tax", "gross"],
        "default":      "ht",
        "reason":       "Par défaut, les montants sont HT (hors taxes)",
    },
    # Période temporelle : année fiscale vs calendaire
    "date_period": {
        "fiscal_keywords": ["fiscal", "exercice", "fy", "financial year"],
        "calendar_keywords": ["calendaire", "calendar", "civil"],
        "default": "calendar",
        "reason": "Par défaut, période calendaire (jan→déc)",
    },
    # Agrégation : SUM vs COUNT vs AVG
    "aggregation": {
        "sum_keywords":   ["total", "somme", "sum", "montant", "chiffre"],
        "count_keywords": ["nombre", "count", "combien", "fois", "occurrences"],
        "avg_keywords":   ["moyenne", "average", "avg", "moyen"],
        "default":        "SUM",
        "reason":         "Par défaut, agrégation SUM sur les champs numériques",
    },
    # Tri : DESC sur les métriques numériques
    "sort_order": {
        "desc_keywords": ["top", "plus", "meilleur", "highest", "max"],
        "asc_keywords":  ["moins", "minimum", "lowest", "min"],
        "default":       "DESC",
        "reason":        "Par défaut, tri décroissant (plus grand en premier)",
    },
}

# Vocabulaire vocal numérique (FR)
VOICE_NUMBER_MAP: Dict[str, str] = {
    "zéro": "0", "un": "1", "deux": "2", "trois": "3", "quatre": "4",
    "cinq": "5", "six": "6", "sept": "7", "huit": "8", "neuf": "9",
    "dix": "10", "vingt": "20", "trente": "30", "quarante": "40",
    "cinquante": "50", "soixante": "60", "cent": "100", "mille": "1000",
    "deux cents": "200", "trois cents": "300", "cinq cents": "500",
    "un million": "1000000", "deux millions": "2000000",
}


# ══════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Interpretation:
    """Une interprétation possible d'un slot ambigu."""
    value:       str
    score:       float          # 0.0 → 1.0
    source:      str            # "history" | "rule" | "frequency" | "default"
    reason:      str = ""
    usage_count: int = 0


@dataclass
class UserPreference:
    """Préférence apprise depuis l'historique."""
    user_id:    str
    source_id:  str
    slot_key:   str
    value:      str
    count:      int = 1
    last_used:  Optional[str] = None
    score:      float = 0.5


# ══════════════════════════════════════════════════════════════════════
# PREFERENCE LEARNER
# ══════════════════════════════════════════════════════════════════════

class PreferenceLearner:
    """
    Apprend les préférences utilisateur depuis l'historique des choix.
    Combine : historique DB + règles métier + fréquence d'usage.
    """

    def __init__(self, pg_pool=None, redis_client=None):
        self.pg    = pg_pool
        self.redis = redis_client

    # ── 1. Scorer les interprétations ────────────────────────────────

    async def score_interpretations(
        self,
        slot_key:    str,
        options:     List[str],
        question:    str,
        user_id:     str = "default",
        source_id:   str = "",
    ) -> List[Interpretation]:
        """
        Retourne les options triées par score décroissant.
        Score = combinaison : historique (0.5) + règle métier (0.3) + fréquence (0.2)
        """
        interpretations = []

        # Historique utilisateur
        history_scores = await self._get_history_scores(
            user_id, source_id, slot_key, options
        )

        # Règles métier
        rule_scores = self._apply_business_rules(slot_key, options, question)

        for opt in options:
            hist_score  = history_scores.get(opt, 0.0)
            rule_score  = rule_scores.get(opt, 0.0)
            usage_count = await self._get_usage_count(user_id, source_id, slot_key, opt)

            # Score combiné pondéré
            combined = (
                hist_score  * 0.50 +
                rule_score  * 0.30 +
                min(usage_count / 10.0, 1.0) * 0.20
            )

            source = "default"
            reason = ""
            if hist_score > 0.5:
                source = "history"
                reason = f"Choisi {usage_count} fois dans l'historique"
            elif rule_score > 0.5:
                source = "rule"
                reason = self._get_rule_reason(slot_key, opt, question)
            elif usage_count > 0:
                source = "frequency"
                reason = f"Utilisé {usage_count} fois"

            interpretations.append(Interpretation(
                value       = opt,
                score       = round(combined, 3),
                source      = source,
                reason      = reason,
                usage_count = usage_count,
            ))

        # Trie par score décroissant
        interpretations.sort(key=lambda x: x.score, reverse=True)
        return interpretations

    def score_interpretations_sync(
        self,
        slot_key:  str,
        options:   List[str],
        question:  str,
    ) -> List[Interpretation]:
        """
        Version synchrone (sans DB) — utilise uniquement les règles métier.
        Utilisée quand pg_pool n'est pas disponible.
        """
        rule_scores = self._apply_business_rules(slot_key, options, question)
        interpretations = []
        for opt in options:
            score  = rule_scores.get(opt, 0.1)
            source = "rule" if score > 0.1 else "default"
            reason = self._get_rule_reason(slot_key, opt, question) if score > 0.1 else ""
            interpretations.append(Interpretation(
                value=opt, score=round(score, 3),
                source=source, reason=reason,
            ))
        interpretations.sort(key=lambda x: x.score, reverse=True)
        return interpretations

    # ── 2. Enregistrer un choix utilisateur ──────────────────────────

    async def record_choice(
        self,
        user_id:   str,
        source_id: str,
        slot_key:  str,
        value:     str,
        question:  str = "",
    ) -> bool:
        """
        Enregistre le choix de l'utilisateur en DB.
        Upsert : incrémente le compteur si déjà présent.
        """
        if not self.pg:
            logger.warning("[PreferenceLearner] Pas de pg_pool, choix non enregistré")
            return False

        try:
            async with self.pg.acquire() as conn:
                await conn.execute("""
                    INSERT INTO user_preferences
                        (user_id, source_id, slot_key, value,
                         usage_count, last_used, question_sample)
                    VALUES ($1, $2, $3, $4, 1, NOW(), $5)
                    ON CONFLICT (user_id, source_id, slot_key, value)
                    DO UPDATE SET
                        usage_count  = user_preferences.usage_count + 1,
                        last_used    = NOW(),
                        question_sample = EXCLUDED.question_sample
                """, user_id, source_id, slot_key, value, question[:200])
            logger.info(
                f"[PreferenceLearner] Choix enregistré — "
                f"user={user_id} slot={slot_key} value={value}"
            )
            return True
        except Exception as e:
            logger.warning(f"[PreferenceLearner] Record error: {e}")
            return False

    # ── 3. Récupérer la meilleure préférence ─────────────────────────

    async def get_best_preference(
        self,
        user_id:   str,
        source_id: str,
        slot_key:  str,
        options:   List[str],
        question:  str = "",
        threshold: float = 0.6,
    ) -> Optional[str]:
        """
        Retourne la meilleure option si son score dépasse le seuil.
        Retourne None si aucune option ne dépasse le seuil → clarification nécessaire.
        """
        scored = await self.score_interpretations(
            slot_key, options, question, user_id, source_id
        )
        if scored and scored[0].score >= threshold:
            logger.info(
                f"[PreferenceLearner] Préférence automatique → "
                f"{scored[0].value} (score={scored[0].score})"
            )
            return scored[0].value
        return None

    def get_best_preference_sync(
        self,
        slot_key:  str,
        options:   List[str],
        question:  str = "",
        threshold: float = 0.5,
    ) -> Optional[str]:
        """Version synchrone sans DB."""
        scored = self.score_interpretations_sync(slot_key, options, question)
        if scored and scored[0].score >= threshold:
            return scored[0].value
        return None

    # ── 4. Résoudre automatiquement sans clarification ────────────────

    async def auto_resolve(
        self,
        slot_key:  str,
        options:   List[str],
        question:  str,
        user_id:   str = "default",
        source_id: str = "",
    ) -> Tuple[Optional[str], List[Interpretation]]:
        """
        Tente de résoudre automatiquement sans demander à l'utilisateur.
        Retourne (valeur_choisie, toutes_les_interprétations).
        valeur_choisie = None si la clarification est nécessaire.
        """
        scored = await self.score_interpretations(
            slot_key, options, question, user_id, source_id
        )

        if not scored:
            return None, []

        best = scored[0]
        second = scored[1] if len(scored) > 1 else None

        # Résolution automatique si :
        # 1. Score suffisamment élevé (> 0.65)
        # 2. Écart suffisant avec le 2ème choix (> 0.2)
        if best.score >= 0.65 and (
            second is None or (best.score - second.score) >= 0.2
        ):
            return best.value, scored

        return None, scored

    # ── 5. Normalisation vocabulaire vocal ───────────────────────────

    @staticmethod
    def normalize_voice_input(text: str) -> str:
        """
        Normalise les nombres écrits en toutes lettres vers des chiffres.
        "donnez moi les deux cents premiers clients" → "donnez moi les 200 premiers clients"
        """
        text_lower = text.lower()
        for word, digit in sorted(
            VOICE_NUMBER_MAP.items(),
            key=lambda x: -len(x[0])   # plus long en premier
        ):
            if word in text_lower:
                text_lower = text_lower.replace(word, digit)
        return text_lower

    @staticmethod
    def normalize_ambiguous_voice(text: str) -> List[str]:
        """
        Détecte les ambiguïtés vocales et retourne les interprétations possibles.
        "deux cents" → ["200", "2 cents"]
        "un million" → ["1000000", "1 millions"]
        """
        interpretations = []

        # Patterns ambigus communs
        ambiguous_patterns = [
            (r'\bdeux cents\b',   ["200", "2 cent(s)"]),
            (r'\bvingt et un\b',  ["21", "20 et 1"]),
            (r'\bun million\b',   ["1000000", "1 million"]),
            (r'\bcinq cents\b',   ["500", "5 cent(s)"]),
            (r'\bdix mille\b',    ["10000", "10 mille"]),
        ]

        for pattern, variants in ambiguous_patterns:
            if re.search(pattern, text.lower()):
                interpretations.extend(variants)

        return list(set(interpretations)) if interpretations else [text]

    # ── 6. Préférences utilisateur sauvegardées ───────────────────────

    async def get_user_preferences(
        self,
        user_id:   str,
        source_id: str = "",
        limit:     int = 50,
    ) -> List[UserPreference]:
        """Retourne toutes les préférences apprises pour un utilisateur."""
        if not self.pg:
            return []
        try:
            async with self.pg.acquire() as conn:
                where = "WHERE user_id = $1"
                params = [user_id]
                if source_id:
                    where += " AND source_id = $2"
                    params.append(source_id)

                rows = await conn.fetch(
                    f"""SELECT user_id, source_id, slot_key, value,
                               usage_count, last_used
                        FROM user_preferences
                        {where}
                        ORDER BY usage_count DESC
                        LIMIT {limit}""",
                    *params
                )
            return [
                UserPreference(
                    user_id   = r["user_id"],
                    source_id = r["source_id"],
                    slot_key  = r["slot_key"],
                    value     = r["value"],
                    count     = r["usage_count"],
                    last_used = r["last_used"].isoformat() if r["last_used"] else None,
                    score     = min(r["usage_count"] / 10.0, 1.0),
                )
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"[PreferenceLearner] Get preferences error: {e}")
            return []

    async def reset_user_preferences(
        self,
        user_id:   str,
        source_id: str = "",
        slot_key:  Optional[str] = None,
    ) -> int:
        """Efface les préférences d'un utilisateur."""
        if not self.pg:
            return 0
        try:
            async with self.pg.acquire() as conn:
                if slot_key:
                    result = await conn.execute(
                        "DELETE FROM user_preferences WHERE user_id=$1 AND slot_key=$2",
                        user_id, slot_key
                    )
                elif source_id:
                    result = await conn.execute(
                        "DELETE FROM user_preferences WHERE user_id=$1 AND source_id=$2",
                        user_id, source_id
                    )
                else:
                    result = await conn.execute(
                        "DELETE FROM user_preferences WHERE user_id=$1",
                        user_id
                    )
            return int(result.split()[-1]) if result else 0
        except Exception as e:
            logger.warning(f"[PreferenceLearner] Reset error: {e}")
            return 0

    # ── Helpers privés ────────────────────────────────────────────────

    async def _get_history_scores(
        self,
        user_id:   str,
        source_id: str,
        slot_key:  str,
        options:   List[str],
    ) -> Dict[str, float]:
        """Retourne les scores historiques depuis la DB."""
        if not self.pg:
            return {}
        try:
            async with self.pg.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT value, usage_count
                    FROM user_preferences
                    WHERE user_id  = $1
                      AND slot_key = $2
                      AND value    = ANY($3)
                    ORDER BY usage_count DESC
                """, user_id, slot_key, options)

            if not rows:
                return {}

            total = sum(r["usage_count"] for r in rows)
            return {
                r["value"]: round(r["usage_count"] / total, 3)
                for r in rows
            }
        except Exception as e:
            logger.warning(f"[PreferenceLearner] History scores error: {e}")
            return {}

    async def _get_usage_count(
        self,
        user_id:   str,
        source_id: str,
        slot_key:  str,
        value:     str,
    ) -> int:
        """Retourne le nombre d'utilisations d'une valeur."""
        if not self.pg:
            return 0
        try:
            async with self.pg.acquire() as conn:
                count = await conn.fetchval("""
                    SELECT COALESCE(usage_count, 0)
                    FROM user_preferences
                    WHERE user_id  = $1
                      AND slot_key = $2
                      AND value    = $3
                """, user_id, slot_key, value)
            return count or 0
        except Exception:
            return 0

    @staticmethod
    def _apply_business_rules(
        slot_key: str,
        options:  List[str],
        question: str,
    ) -> Dict[str, float]:
        """
        Applique les règles métier prédéfinies pour scorer les options.
        Retourne {option: score}.
        """
        q_lower = question.lower()
        scores: Dict[str, float] = {}

        if slot_key == "field_names":
            # Règle montant : HT vs TTC
            rule = BUSINESS_RULES["amount_field"]
            for opt in options:
                opt_lower = opt.lower()
                if any(kw in opt_lower or kw in q_lower for kw in rule["keywords_ht"]):
                    scores[opt] = 0.75
                elif any(kw in opt_lower or kw in q_lower for kw in rule["keywords_ttc"]):
                    scores[opt] = 0.60

        elif slot_key == "table_names":
            # Priorité aux tables transactionnelles sur les tables d'audit (_A suffix)
            for opt in options:
                if opt.endswith("_A") or opt.endswith("_AUDIT"):
                    scores[opt] = 0.20
                elif any(kw in opt.lower() for kw in ["order", "invoice", "sale", "customer", "product"]):
                    scores[opt] = 0.80
                else:
                    scores[opt] = 0.40

        elif slot_key == "intent":
            # Règle agrégation
            rule = BUSINESS_RULES["aggregation"]
            for opt in options:
                opt_lower = opt.lower()
                if any(kw in q_lower for kw in rule["sum_keywords"]) and "sum" in opt_lower:
                    scores[opt] = 0.80
                elif any(kw in q_lower for kw in rule["count_keywords"]) and "count" in opt_lower:
                    scores[opt] = 0.80
                elif any(kw in q_lower for kw in rule["avg_keywords"]) and "avg" in opt_lower:
                    scores[opt] = 0.80

        # Score par défaut pour les options non scorées
        for opt in options:
            if opt not in scores:
                scores[opt] = 0.10

        return scores

    @staticmethod
    def _get_rule_reason(slot_key: str, value: str, question: str) -> str:
        """Retourne la raison de la règle appliquée."""
        if slot_key == "field_names":
            rule = BUSINESS_RULES["amount_field"]
            if any(kw in value.lower() for kw in rule["keywords_ht"]):
                return rule["reason"]
        if slot_key == "table_names":
            if not (value.endswith("_A") or value.endswith("_AUDIT")):
                return "Tables transactionnelles prioritaires sur les tables d'audit"
        return "Règle métier par défaut"