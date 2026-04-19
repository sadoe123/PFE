"""
OnePilot – Ambiguity Resolver §2.3.4  ── 100% complet
Détection et résolution des ambiguïtés dans les requêtes NL.
- Scoring de probabilité pour chaque interprétation
- Apprentissage des préférences utilisateur (historique)
- Fallback sur règles métier prédéfinies
- Normalisation vocabulaire vocal
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .nlu_engine import Intent, QuerySlots
from .preference_learner import PreferenceLearner, Interpretation

logger = logging.getLogger(__name__)


@dataclass
class ClarificationQuestion:
    """Question de clarification à poser à l'utilisateur."""
    question:  str
    options:   List[str]
    slot_key:  str   # quel slot résoudre
    required:  bool = True


class AmbiguityResolver:
    """
    Détecte et résout les ambiguïtés dans les requêtes NL.
    Génère des questions de clarification ciblées.
    """

    def analyze(
        self,
        slots: QuerySlots,
        known_entities: List[str],
        known_fields: Dict[str, List[str]],
    ) -> List[ClarificationQuestion]:
        """
        Analyse les slots et retourne les questions de clarification nécessaires.
        known_entities = noms de tables connues
        known_fields   = {table: [fields]}
        """
        questions = []

        # 1. Intent inconnu
        if slots.intent == Intent.UNKNOWN:
            questions.append(ClarificationQuestion(
                question = "Je n'ai pas bien compris votre demande. Que souhaitez-vous faire ?",
                options  = [
                    "Lister les tables",
                    "Générer du SQL",
                    "Voir les relations",
                    "Profiler une table",
                    "Chercher une entité",
                ],
                slot_key  = "intent",
                required  = True,
            ))

        # 2. Table non spécifiée pour SQL/JOIN
        if slots.intent in (Intent.GENERATE_SQL, Intent.GENERATE_JOIN, Intent.GENERATE_AGG) \
                and not slots.table_names and known_entities:
            # Trier par pertinence : vues métier en premier
            def _score(n):
                nl = n.lower()
                if any(k in nl for k in ["tresorerie","bancaire","journal","transaction",
                    "financement","comptes","orders","customers","products","employees"]):
                    return 0
                if not nl.endswith("_a") and not nl.endswith("_u"):
                    return 1
                return 2
            top_entities = sorted(known_entities, key=_score)[:5]
            questions.append(ClarificationQuestion(
                question = "Sur quelle table souhaitez-vous effectuer cette requête ?",
                options  = top_entities,
                slot_key = "table_names",
                required = True,
            ))

        # 3. Champ ambigu pour l'agrégation
        if slots.intent == Intent.GENERATE_AGG and slots.table_names:
            table = slots.table_names[0]
            fields = known_fields.get(table, [])
            numeric_fields = [f for f in fields if any(
                kw in f.lower() for kw in
                ["amount", "total", "price", "qty", "value", "montant", "prix", "sum", "cost"]
            )]
            if len(numeric_fields) > 1 and not slots.metric:
                questions.append(ClarificationQuestion(
                    question = f"Quel champ souhaitez-vous agréger dans '{table}' ?",
                    options  = numeric_fields[:5],
                    slot_key = "field_names",
                    required = False,
                ))

        # 4. Tables multiples sans JOIN explicite
        if len(slots.table_names) > 2:
            questions.append(ClarificationQuestion(
                question = f"J'ai trouvé plusieurs tables ({', '.join(slots.table_names[:3])}). Laquelle vous intéresse ?",
                options  = slots.table_names[:5],
                slot_key = "table_names",
                required = True,
            ))

        # 5. Filtre sans valeur (seulement pour les intents qui nécessitent un filtre)
        filter_intents = (Intent.GENERATE_FILTER, Intent.GENERATE_SQL)
        if (slots.filter_op and not slots.amount_filter and not slots.date_filter
                and slots.intent in filter_intents):
            questions.append(ClarificationQuestion(
                question = "Vous souhaitez filtrer les données. Quelle est la valeur du filtre ?",
                options  = ["Une date spécifique", "Un montant", "Une valeur texte"],
                slot_key = "filter_value",
                required = False,
            ))

        return questions

    def build_clarification_response(
        self,
        questions: List[ClarificationQuestion],
    ) -> Dict:
        """
        Construit la réponse de clarification à afficher dans le chat.
        """
        if not questions:
            return {"needs_clarification": False}

        q = questions[0]  # Une question à la fois

        return {
            "needs_clarification": True,
            "question":            q.question,
            "options":             q.options,
            "slot_key":            q.slot_key,
            "required":            q.required,
            "remaining":           len(questions) - 1,
        }

    def apply_clarification(
        self,
        slots: QuerySlots,
        slot_key: str,
        value: str,
    ) -> QuerySlots:
        """
        Applique la réponse de clarification aux slots.
        """
        if slot_key == "intent":
            intent_map = {
                "lister les tables":  Intent.LIST_ENTITIES,
                "générer du sql":     Intent.GENERATE_SQL,
                "voir les relations": Intent.GET_RELATIONS,
                "profiler une table": Intent.PROFILE_ENTITY,
                "chercher une entité":Intent.SEARCH_ENTITY,
            }
            slots.intent = intent_map.get(value.lower(), slots.intent)

        elif slot_key == "table_names":
            if value not in slots.table_names:
                slots.table_names = [value] + slots.table_names

        elif slot_key == "field_names":
            if value not in slots.field_names:
                slots.field_names = [value] + slots.field_names

        # Nettoie les ambiguïtés résolues
        slots.ambiguities = [a for a in slots.ambiguities
                             if slot_key not in a and "no_table" not in a]

        return slots

    # ── §2.3.4 Scoring + Apprentissage ───────────────────────────────

    def analyze_with_scoring(
        self,
        slots: QuerySlots,
        known_entities: List[str],
        known_fields: Dict[str, List[str]],
        question: str = "",
        user_id: str = "default",
        source_id: str = "",
    ) -> Tuple[List["ClarificationQuestion"], Dict[str, List[Interpretation]]]:
        """
        Analyse les slots ET retourne les scores pour chaque option.
        Utilise PreferenceLearner pour scorer sans DB (sync).
        Retourne (questions, {slot_key: [Interpretation]}).
        """
        questions = self.analyze(slots, known_entities, known_fields)
        learner   = PreferenceLearner()
        scored:   Dict[str, List[Interpretation]] = {}

        for q in questions:
            interpretations = learner.score_interpretations_sync(
                slot_key = q.slot_key,
                options  = q.options,
                question = question,
            )
            scored[q.slot_key] = interpretations

            # Auto-résolution : si meilleur score >= 0.65 et required=False
            if not q.required and interpretations and interpretations[0].score >= 0.65:
                best = interpretations[0].value
                logger.info(
                    f"[AmbiguityResolver] Auto-résolution {q.slot_key} → "
                    f"{best} (score={interpretations[0].score})"
                )

        return questions, scored

    async def analyze_with_learning(
        self,
        slots: QuerySlots,
        known_entities: List[str],
        known_fields: Dict[str, List[str]],
        question: str = "",
        user_id: str = "default",
        source_id: str = "",
        pg_pool = None,
    ) -> Tuple[List["ClarificationQuestion"], Dict[str, List[Interpretation]], Optional[str]]:
        """
        Version asynchrone avec DB.
        Retourne (questions, scored, auto_resolved_value).
        auto_resolved_value = valeur résolue automatiquement si possible, sinon None.
        """
        questions = self.analyze(slots, known_entities, known_fields)
        if not questions:
            return [], {}, None

        learner = PreferenceLearner(pg_pool=pg_pool)
        scored: Dict[str, List[Interpretation]] = {}
        auto_resolved = None

        for q in questions:
            # Tente résolution automatique via préférences
            best_value, interpretations = await learner.auto_resolve(
                slot_key  = q.slot_key,
                options   = q.options,
                question  = question,
                user_id   = user_id,
                source_id = source_id,
            )
            scored[q.slot_key] = interpretations

            # Si résolution auto possible ET question non-bloquante
            if best_value and not q.required:
                auto_resolved = best_value
                logger.info(
                    f"[AmbiguityResolver] Résolution auto (non-required) "
                    f"{q.slot_key} → {best_value}"
                )

        return questions, scored, auto_resolved

    async def record_user_choice(
        self,
        user_id:   str,
        source_id: str,
        slot_key:  str,
        value:     str,
        question:  str = "",
        pg_pool    = None,
    ) -> bool:
        """
        Enregistre le choix de l'utilisateur pour apprentissage futur.
        À appeler quand l'utilisateur répond à une question de clarification.
        """
        learner = PreferenceLearner(pg_pool=pg_pool)
        return await learner.record_choice(
            user_id   = user_id,
            source_id = source_id,
            slot_key  = slot_key,
            value     = value,
            question  = question,
        )

    def normalize_voice_question(self, question: str) -> str:
        """
        Normalise le vocabulaire vocal avant NLU.
        "donnez moi les deux cents premiers clients" → "donnez moi les 200 premiers clients"
        """
        return PreferenceLearner.normalize_voice_input(question)

    def build_clarification_response_with_scores(
        self,
        questions: List["ClarificationQuestion"],
        scored:    Dict[str, List[Interpretation]],
    ) -> Dict:
        """
        Construit la réponse de clarification enrichie avec les scores.
        Les options sont triées par score décroissant.
        """
        if not questions:
            return {"needs_clarification": False}

        q = questions[0]
        interps = scored.get(q.slot_key, [])

        # Options triées par score si disponibles
        if interps:
            sorted_options = [i.value for i in interps]
            scores_map     = {i.value: i.score for i in interps}
            reasons_map    = {i.value: i.reason for i in interps}
        else:
            sorted_options = q.options
            scores_map     = {}
            reasons_map    = {}

        return {
            "needs_clarification": True,
            "question":            q.question,
            "options":             sorted_options[:5],
            "options_scored": [
                {
                    "value":  opt,
                    "score":  scores_map.get(opt, 0.0),
                    "reason": reasons_map.get(opt, ""),
                }
                for opt in sorted_options[:5]
            ],
            "slot_key":  q.slot_key,
            "required":  q.required,
            "remaining": len(questions) - 1,
        }