"""
OnePilot – Ambiguity Resolver §2.3.4
Détection et résolution des ambiguïtés dans les requêtes NL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .nlu_engine import Intent, QuerySlots

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
            top_entities = known_entities[:5]
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
