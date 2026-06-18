from __future__ import annotations
"""
OnePilot – NLU Engine v2 §2.3.1
Pipeline NLU avec vrais modèles ML :
  - SpaCy fr_core_news_sm : NER, lemmatisation, tokenisation
  - Sentence-BERT multilingue : embeddings sémantiques pour intent classification
  - Intent classification hybride : BERT similarity + regex fallback
  - Support FR/EN/ES/DE via modèle multilingue
"""

# ── Configuration cache HuggingFace ──────────────────────────────────────────
import os as _os_hf
_os_hf.environ.setdefault("HF_HOME", "/tmp/hf_cache")
_os_hf.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")
_os_hf.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/hf_cache")
# ─────────────────────────────────────────────────────────────────────────────


import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# IMPORTS MODÈLES ML (lazy loading pour ne pas ralentir le démarrage)
# ══════════════════════════════════════════════════════════════════════

_spacy_model  = None
_bert_model   = None
_intent_embeddings: Optional[Dict[str, Any]] = None



# ══════════════════════════════════════════════════════════════════════
# FASTTEXT INTENT CLASSIFIER
# ══════════════════════════════════════════════════════════════════════

FASTTEXT_TRAIN_DATA = """
__label__generate_aggregate total par categorie
__label__generate_aggregate total par groupe
__label__generate_aggregate total par type
__label__generate_aggregate total par periode
__label__generate_aggregate total par dimension
__label__generate_aggregate somme par entite
__label__generate_aggregate somme par groupe
__label__generate_aggregate somme par type
__label__generate_aggregate moyenne par categorie
__label__generate_aggregate moyenne par groupe
__label__generate_aggregate repartition par type
__label__generate_aggregate repartition par categorie
__label__generate_aggregate repartition par dimension
__label__generate_aggregate top 10 par valeur
__label__generate_aggregate top 5 par montant
__label__generate_aggregate top N par critere
__label__generate_aggregate classement par valeur
__label__generate_aggregate classement par montant
__label__generate_aggregate meilleures entites par valeur
__label__generate_aggregate les plus grandes valeurs
__label__generate_aggregate ranking by value
__label__generate_aggregate top 10 by amount
__label__generate_aggregate total by category
__label__generate_aggregate sum by group
__label__generate_aggregate average by type
__label__generate_aggregate distribution by dimension
__label__generate_aggregate top records by value
__label__generate_aggregate group by attribute
__label__generate_aggregate aggregate by field
__label__generate_aggregate count by status
__label__correct_sql c'est faux
__label__correct_sql c'est pas bon
__label__correct_sql corrige le sql
__label__correct_sql il manque un crochet
__label__correct_sql manque les crochets
__label__correct_sql ajoute un filtre
__label__correct_sql ajoute la condition
__label__correct_sql remplace par
__label__correct_sql change la colonne
__label__correct_sql erreur dans le sql
__label__correct_sql le sql est faux
__label__correct_sql mauvaise colonne
__label__correct_sql mauvaise table
__label__correct_sql il faut corriger
__label__correct_sql modifie la requete
__label__correct_sql mets les crochets
__label__correct_sql ajoute top 100
__label__correct_sql enleve le filtre
__label__correct_sql change le where
__label__correct_sql ca marche pas
__label__correct_sql pas le bon resultat
__label__correct_sql wrong sql
__label__correct_sql fix the query
__label__correct_sql incorrect query
__label__correct_sql add filter
__label__correct_sql add brackets
__label__correct_sql wrong column
__label__correct_sql missing bracket
__label__correct_sql update the sql
__label__correct_sql change the table
__label__generate_aggregate transactions superieures a 50000
__label__generate_aggregate transactions inferieures a 10000
__label__generate_aggregate montants superieurs a 100000
__label__generate_aggregate transactions en 2023
__label__generate_aggregate transactions en 2024
__label__generate_aggregate transactions par banque en 2023
__label__generate_aggregate financements dont la maturite depasse 5 ans
__label__generate_aggregate financements superieurs a 1000000
__label__generate_aggregate encaissements superieurs a 50000
__label__generate_aggregate decaissements superieurs a 50000
__label__generate_aggregate liste des transactions superieures
__label__generate_aggregate total des transactions superieures a la moyenne
__label__show_dashboard dashboard par categorie
__label__show_dashboard dashboard par type
__label__show_dashboard dashboard par periode
__label__show_dashboard dashboard par dimension
__label__show_dashboard dashboard par groupe
__label__show_dashboard affiche dashboard par entite
__label__show_dashboard montre dashboard par dimension
__label__show_dashboard dashboard analyse par groupe
__label__show_dashboard cree dashboard par valeur
__label__show_dashboard visualise par categorie
__label__show_dashboard genere dashboard par type
__label__show_dashboard tableau de bord par categorie
__label__show_dashboard tableau de bord par periode
__label__show_dashboard dashboard des donnees
__label__show_dashboard show dashboard by category
__label__show_dashboard generate dashboard by type
__label__show_dashboard create dashboard by period
__label__show_dashboard display dashboard for entity
__label__show_dashboard dashboard overview
__label__show_dashboard visualize data dashboard
__label__list_entities liste les tables
__label__list_entities montre les entites
__label__list_entities quelles sont les tables
__label__list_entities liste les entites disponibles
__label__list_entities montre moi les sources
__label__list_entities quelles entites existent
__label__list_entities affiche toutes les tables
__label__list_entities liste les objets disponibles
__label__list_entities show tables
__label__list_entities list all entities
__label__list_entities what tables exist
__label__list_entities show available sources
__label__count_entities combien de lignes
__label__count_entities combien d enregistrements
__label__count_entities nombre de lignes dans la table
__label__count_entities taille de la table
__label__count_entities nombre total d enregistrements
__label__count_entities compter les enregistrements
__label__count_entities how many rows
__label__count_entities count records
__label__count_entities how many entries
__label__count_entities total number of records
__label__count_entities count all rows in table
__label__count_entities how many items exist
__label__count_entities combien de tables
__label__count_entities combien d entites
__label__count_entities nombre de tables dans la source
__label__count_entities combien de tables dans sxa
__label__count_entities combien d objets dans la source
__label__count_entities nombre total de tables
__label__count_entities how many tables
__label__count_entities how many entities
__label__count_entities count tables in source
__label__count_entities nombre de tables disponibles
__label__count_entities combien y a t il de tables
__label__count_entities combien de tables sont disponibles
__label__count_entities quel est le nombre de tables
__label__count_entities donnez moi le nombre de tables
__label__count_entities nombre d entites indexees
__label__count_entities combien de vues dans la source
__label__count_entities total tables in database
__label__count_entities how many tables in source
__label__list_fields champs de la table
__label__list_fields colonnes de cette entite
__label__list_fields quelles colonnes disponibles
__label__list_fields liste des colonnes
__label__list_fields champs disponibles
__label__list_fields quels attributs existent
__label__list_fields affiche les champs
__label__list_fields liste les attributs
__label__list_fields quels champs sont des cles primaires
__label__list_fields quelles sont les cles primaires
__label__list_fields liste les cles primaires
__label__list_fields primary keys de la table
__label__list_fields quels champs sont indexes
__label__list_fields fields of the table
__label__list_fields columns of this entity
__label__list_fields what fields are available
__label__list_fields list all columns
__label__list_fields show primary keys
__label__describe_entity decris la table
__label__describe_entity info sur cette table
__label__describe_entity description de l entite
__label__describe_entity que contient cette table
__label__describe_entity explique cette entite
__label__describe_entity decris cette source
__label__describe_entity describe the entity
__label__describe_entity what is this table
__label__describe_entity explain this entity
__label__describe_entity tell me about this table
__label__describe_entity describe this source
__label__get_relations relations de la table
__label__get_relations liens entre tables
__label__get_relations dependances entre entites
__label__get_relations connexions entre tables
__label__get_relations relations entre deux entites
__label__get_relations quelles sont les relations
__label__get_relations foreign keys
__label__get_relations cles etrangeres
__label__get_relations dependances de la table
__label__get_relations relations disponibles
__label__get_relations show relations
__label__get_relations list foreign keys
__label__get_relations entity dependencies
__label__get_relations table connections
__label__get_relations quelles tables sont liees a
__label__get_relations quelles tables sont liees
__label__get_relations tables liees a cette table
__label__get_relations tables reliees a
__label__get_relations quelles entites sont connectees
__label__get_relations montre les liaisons de
__label__get_relations quelles sont les tables liees
__label__get_relations tables en relation avec
__label__get_relations relations de cette entite
__label__get_relations liens de la table
__label__get_relations quelles tables pointent vers
__label__get_relations what tables are linked to
__label__get_relations tables linked to
__label__get_relations show links for table
__label__get_relations quelles fk pour cette table
__label__get_relations montre les fk de
__label__get_relations foreign keys of table
__label__get_relations relations entrantes et sortantes
__label__generate_join jointure entre deux tables
__label__generate_join joindre deux entites
__label__generate_join relier deux tables
__label__generate_join join two tables
__label__generate_join joindre les tables
__label__generate_join jointure entre entites
__label__generate_join combine two tables
__label__generate_join merge tables together
__label__generate_join join entities
__label__generate_sql genere du SQL
__label__generate_sql ecris une requete
__label__generate_sql donne moi les donnees brutes
__label__generate_sql requete SQL pour cette table
__label__generate_sql generate SQL query
__label__generate_sql write a query
__label__generate_sql get raw data
__label__generate_sql SQL request for table
__label__profile_entity profil de la table
__label__profile_entity statistiques de la table
__label__profile_entity qualite des donnees
__label__profile_entity analyse la table
__label__profile_entity data quality check
__label__profile_entity table statistics
__label__profile_entity data profile
__label__profile_entity analyze data quality
__label__profile_entity show table statistics
__label__search_entity cherche la table
__label__search_entity trouve une entite
__label__search_entity recherche cette table
__label__search_entity find this table
__label__search_entity search for entity
__label__search_entity look for table
__label__search_entity find entity by name
__label__find_path chemin entre deux tables
__label__find_path chemin entre entites
__label__find_path comment relier ces tables
__label__find_path path between tables
__label__find_path how to join these entities
__label__find_path find path between tables
__label__find_path shortest path between entities
__label__greeting bonjour
__label__greeting hello
__label__greeting salut
__label__greeting hi there
__label__greeting hey
__label__greeting bonsoir
__label__help aide moi
__label__help help me
__label__help que peux tu faire
__label__help what can you do
__label__help comment tu fonctionnes
__label__help aide
__label__help how do you work
"""

_fasttext_model = None

def _get_fasttext():
    global _fasttext_model
    if _fasttext_model is None:
        try:
            import fasttext, tempfile, os
            # Entraîne le modèle sur les données d'exemple
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write(FASTTEXT_TRAIN_DATA)
                tmp = f.name
            _fasttext_model = fasttext.train_supervised(
                tmp, epoch=50, lr=0.5, wordNgrams=2, dim=100, verbose=0
            )
            os.unlink(tmp)
            logger.info("[NLU v3] FastText entraîné")
        except Exception as e:
            logger.warning(f"[NLU v3] FastText non disponible: {e}")
            _fasttext_model = False
    return _fasttext_model if _fasttext_model else None



def retrain_fasttext_with_feedback(extra_examples: list[tuple[str, str]]) -> dict:
    """
    Réentraîne FastText en ajoutant les exemples validés par feedback.
    extra_examples = [(question, intent), ...]
    Recharge le modèle en mémoire immédiatement.
    Retourne {"status", "examples_added", "total_examples", "accuracy"}.
    """
    global _fasttext_model
    try:
        import fasttext, tempfile, os

        # ── Construit le dataset augmenté ──────────────────────────
        base_lines = [
            l for l in FASTTEXT_TRAIN_DATA.strip().split("\n")
            if l.strip().startswith("__label__")
        ]

        # Déduplique les nouveaux exemples
        existing_texts = set(
            " ".join(l.split()[1:]) for l in base_lines
        )
        new_lines = []
        for question, intent in extra_examples:
            q_clean = question.strip().lower()
            if q_clean and q_clean not in existing_texts:
                label = intent.replace("Intent.", "").lower()
                new_lines.append(f"__label__{label}  {q_clean}")
                existing_texts.add(q_clean)

        all_lines = base_lines + new_lines
        total = len(all_lines)

        # ── Entraîne le nouveau modèle ──────────────────────────────
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("\n".join(all_lines))
            tmp = f.name

        new_model = fasttext.train_supervised(
            tmp, epoch=50, lr=0.5, wordNgrams=2, dim=100, verbose=0
        )
        os.unlink(tmp)

        # ── Recharge en mémoire (hot reload) ───────────────────────
        _fasttext_model = new_model

        # Réinitialise le pipeline NLU pour prendre le nouveau modèle
        global _nlu_pipeline
        _nlu_pipeline = None  # Force reinit au prochain appel

        logger.info(
            f"[FastText Retrain] ✅ Réentraîné — "
            f"+{len(new_lines)} exemples, total={total}"
        )

        return {
            "status":          "ok",
            "examples_added":  len(new_lines),
            "total_examples":  total,
            "model_reloaded":  True,
        }

    except Exception as e:
        logger.error(f"[FastText Retrain] Erreur: {e}")
        return {"status": "error", "message": str(e)}


class FastTextClassifier:
    """Classification rapide d'intent via FastText (2ms)."""

    def classify(self, text: str) -> Tuple[str, float]:
        ft = _get_fasttext()
        if not ft:
            return Intent.UNKNOWN, 0.0
        try:
            # Nettoie le texte pour FastText
            clean = text.lower().replace("\n", " ").strip()
            labels, probs = ft.predict(clean, k=1)
            if labels:
                intent = labels[0].replace("__label__", "")
                return intent, float(probs[0])
        except Exception as e:
            logger.warning(f"[FastText] Error: {e}")
        return Intent.UNKNOWN, 0.0


# ══════════════════════════════════════════════════════════════════════
# ROBERTA CLASSIFIER (haute précision)
# ══════════════════════════════════════════════════════════════════════

_roberta_model = None
_roberta_embeddings = None

def _get_roberta():
    global _roberta_model, _roberta_embeddings
    if _roberta_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            _roberta_model = SentenceTransformer(
                "xlm-roberta-base", local_files_only=True
            )
            # Calcule les embeddings des exemples
            embeddings = {}
            for intent_name, examples in INTENT_EXAMPLES.items():
                vecs = _roberta_model.encode(examples, normalize_embeddings=True)
                embeddings[intent_name] = np.mean(vecs, axis=0)
            _roberta_embeddings = embeddings
            logger.info("[NLU v3] RoBERTa XLM chargé")
        except Exception as e:
            logger.warning(f"[NLU v3] RoBERTa non disponible: {e}")
            _roberta_model = False
    return (_roberta_model, _roberta_embeddings) if _roberta_model else (None, None)


class RoBERTaClassifier:
    """Classification haute précision via RoBERTa XLM multilingue (50ms)."""

    def classify(self, text: str) -> Tuple[str, float]:
        import numpy as np
        model, embeddings = _get_roberta()
        if not model or not embeddings:
            return Intent.UNKNOWN, 0.0
        try:
            q_vec = model.encode([text], normalize_embeddings=True)[0]
            scores = {
                intent: float(np.dot(q_vec, vec))
                for intent, vec in embeddings.items()
            }
            best = max(scores, key=scores.__getitem__)
            return best, round(scores[best], 3)
        except Exception as e:
            logger.warning(f"[RoBERTa] Error: {e}")
        return Intent.UNKNOWN, 0.0



def _get_spacy():
    global _spacy_model
    if _spacy_model is None:
        try:
            import spacy
            _spacy_model = spacy.load("fr_core_news_sm")
            logger.info("[NLU v2] SpaCy fr_core_news_sm chargé")
        except Exception as e:
            logger.warning(f"[NLU v2] SpaCy non disponible: {e}")
            _spacy_model = False
    return _spacy_model if _spacy_model else None


def _get_bert():
    global _bert_model
    if _bert_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _bert_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2",
                local_files_only=True,
            )
            logger.info("[NLU v2] BERT multilingue chargé")
        except Exception as e:
            logger.warning(f"[NLU v2] BERT non disponible: {e}")
            _bert_model = False
    return _bert_model if _bert_model else None


# ══════════════════════════════════════════════════════════════════════
# INTENTS avec exemples d'entraînement multilingues
# ══════════════════════════════════════════════════════════════════════

class Intent:
    COUNT_ENTITIES   = "count_entities"
    LIST_ENTITIES    = "list_entities"
    DESCRIBE_ENTITY  = "describe_entity"
    LIST_FIELDS      = "list_fields"
    GET_RELATIONS    = "get_relations"
    DESCRIBE_SOURCE  = "describe_source"
    GENERATE_SQL     = "generate_sql"
    GENERATE_JOIN    = "generate_join"
    GENERATE_FILTER  = "generate_filter"
    GENERATE_AGG     = "generate_aggregate"
    PROFILE_ENTITY   = "profile_entity"
    SEARCH_ENTITY    = "search_entity"
    COMPARE_ENTITIES = "compare_entities"
    FIND_PATH        = "find_path"
    SHOW_DASHBOARD   = "show_dashboard"
    GREETING         = "greeting"
    HELP             = "help"
    UNKNOWN          = "unknown"
    LLM_EXPLAIN      = "llm_explain"   # Questions explicatives → LLM
    CORRECT_SQL      = "correct_sql"    # Correction du dernier SQL généré


# Phrases d'exemple par intent (FR + EN + ES + DE)
INTENT_EXAMPLES: Dict[str, List[str]] = {
    Intent.COUNT_ENTITIES: [
        "combien de tables", "nombre d'entités", "how many tables",
        "combien de lignes", "how many rows", "count records",
        "cuántas tablas", "wie viele Tabellen",
    ],
    Intent.LIST_ENTITIES: [
        "liste les tables", "montre les entités", "show tables",
        "affiche les tables", "quelles sont les tables",
        "list all entities", "mostrar tablas", "Tabellen anzeigen",
    ],
    Intent.DESCRIBE_ENTITY: [
        "décris la table", "info sur", "describe the table",
        "détail de l'entité", "what is this table", "describe entity",
        "información sobre", "Tabelle beschreiben",
    ],
    Intent.LIST_FIELDS: [
        "champs de la table", "colonnes de", "fields of",
        "attributs de", "columns in", "liste les champs",
        "campos de", "Felder von",
        # Clés primaires / primaires
        "clés primaires", "cles primaires", "primary key", "primary keys",
        "clé primaire", "cle primaire", "quels champs", "quelles colonnes",
        "quels sont les champs", "quelles sont les colonnes",
        "champs disponibles", "liste des champs", "liste des colonnes",
        "clés étrangères", "foreign key", "indexes", "index de",
    ],
    Intent.GET_RELATIONS: [
        "relations de", "liens entre", "foreign keys",
        "dépendances", "relationships", "clés étrangères",
        "relaciones de", "Beziehungen von",
    ],
    Intent.GENERATE_SQL: [
        "génère du SQL", "écris une requête", "generate SQL query",
        "write a query", "donne moi les données", "show me the data",
        "generar SQL", "SQL generieren",
    ],
    Intent.GENERATE_JOIN: [
        "jointure entre", "join entre", "rejoindre les tables",
        "joindre", "join tables", "merge tables",
        "unir tablas", "Tabellen verbinden",
    ],
    Intent.GENERATE_AGG: [
        "total par", "somme par", "moyenne par", "sum by",
        "group by", "agréger par", "total des ventes",
        "total por", "Summe nach",
    ],
    Intent.GENERATE_FILTER: [
        "filtre par date", "where clause", "filtrer les données",
        "données du mois", "filter by", "données supérieures à",
        "filtrar por", "filtern nach",
    ],
    Intent.PROFILE_ENTITY: [
        "profil de la table", "statistiques de", "stats de",
        "distribution des valeurs", "profile table", "data quality",
        "perfil de tabla", "Tabellenprofil",
    ],
    Intent.SEARCH_ENTITY: [
        "cherche la table", "trouve l'entité", "find table",
        "search for", "où est la table", "localiser",
        "buscar tabla", "Tabelle suchen",
    ],
    Intent.FIND_PATH: [
        "chemin entre", "comment relier", "path between",
        "comment joindre", "how to join", "lien entre deux tables",
        "camino entre", "Pfad zwischen",
    ],
    Intent.GREETING: [
        "bonjour", "salut", "hello", "hi", "coucou",
        "hola", "hallo", "bonsoir",
    ],
    Intent.HELP: [
        "aide", "help", "que peux-tu faire", "what can you do",
        "commandes disponibles", "ayuda", "Hilfe",
    ],
    Intent.DESCRIBE_SOURCE: [
        "info sur la source", "résumé de la source", "describe source",
        "structure de la base", "información de la fuente",
    ],
    Intent.SHOW_DASHBOARD: [
        "génère un dashboard", "montre le dashboard", "crée un dashboard",
        "tableau de bord", "visualise les données", "dashboard des ventes",
        "generate dashboard", "show dashboard", "create dashboard",
        "visualisation des données", "graphiques des ventes",
        "analyse visuelle", "dashboard analyse",
    ],
}

# Patterns regex de fallback (gardés du v1)
INTENT_PATTERNS: Dict[str, List[str]] = {
    Intent.COUNT_ENTITIES: [
        r"combien\s+(de\s+)?(tables?|entit|lignes?|rows?|enregistrements?)",
        r"how\s+many\s+(tables?|rows?|records?)",
        r"nb\s+(de\s+)?(lignes?|enregistrements?)",
        r"taille\s+(de\s+)?(\w+)",
    ],
    Intent.LIST_ENTITIES: [
        r"list[ez]?\s*(les?\s+)?(tables?|entit[eé])",
        r"montr[ez]?\s*(les?\s+)?(tables?|entit[eé])",
        r"quell?es?\s+(sont\s+)?(les?\s+)?(tables?|entit[eé])",
        r"affich[ez]?\s*(les?\s+)?(tables?|entit[eé])",
    ],
    Intent.GENERATE_JOIN: [
        r"jointure\s+(entre\s+)?(\w+)\s+(et|and)\s+(\w+)",
        r"join\s+(entre\s+)?(\w+)\s+(et|and|with)\s+(\w+)",
        r"joindre\s+(\w+)\s+(et|and|avec|[aà])\s+(\w+)",
    ],
    Intent.GENERATE_AGG: [
        r"total\s+(de\s+|des?\s+|par\s+)?",
        r"somme\s+(des?\s+|par\s+)?",
        r"moyenne\s+(des?\s+|par\s+)?",
        r"group\s+by",
        r"sum\s+(of\s+|by\s+)?",
    ],
    Intent.LIST_FIELDS: [
        r"quels?\s+champs?\s+(sont|se|de|dans)",
        r"cl[eé]s?\s+primaires?",
        r"primary\s+keys?",
        r"cl[eé]s?\s+[eé]trang[eè]res?",
        r"foreign\s+keys?",
        r"champs?\s+(de\s+la\s+table|disponibles?|de\s+|d[''`])",
        r"colonnes?\s+(de\s+|d[''`]|disponibles?)",
        r"liste\s+(les?\s+)?champs?",
        r"liste\s+(les?\s+)?colonnes?",
        r"liste\s+(les?\s+)?cl[eé]",
        r"liste\s+(les?\s+)?index",
        r"quelles?\s+colonnes?",
        r"index\s+(de\s+|sur\s+)",
        r"cl[eé]s?\s+(de\s+)?la\s+",
        r"affich[ez]?\s+(les?\s+)?cl[eé]",
    ],
    Intent.GET_RELATIONS: [
        r"relations?\s+(de\s+|of\s+)?(\w+)?",
        r"liens?\s+(de\s+)?(\w+)?",
        r"foreign\s+key",
    ],
    Intent.GENERATE_SQL: [
        r"g[eé]n[eè]re\s+(du\s+|une?\s+)?sql",
        r"select\s+.*\s+from",
        r"donne\s+(moi\s+)?(les?\s+)?donn[eé]es",
    ],
    Intent.GREETING: [
        r"^(bonjour|salut|hello|hi|coucou|bonsoir|hola|hallo)[\s!]*$",
    ],
    Intent.HELP: [
        r"aide|help|comment|que\s+peux.tu|what\s+can\s+you",
    ],
}


# ══════════════════════════════════════════════════════════════════════
# INTENT CLASSIFIER BERT
# ══════════════════════════════════════════════════════════════════════

class BERTIntentClassifier:
    """
    Classification d'intent par similarité cosinus avec BERT multilingue.
    """

    def __init__(self):
        self._embeddings: Optional[Dict] = None

    def _build_embeddings(self):
        """Calcule les embeddings des phrases d'exemple une seule fois."""
        import numpy as np
        bert = _get_bert()
        if not bert:
            return None

        embeddings = {}
        for intent_name, examples in INTENT_EXAMPLES.items():
            vecs = bert.encode(examples, normalize_embeddings=True)
            embeddings[intent_name] = np.mean(vecs, axis=0)

        logger.info(f"[NLU v2] Embeddings calculés pour {len(embeddings)} intents")
        return embeddings

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classe l'intent via similarité cosinus BERT.
        Retourne (intent, confidence).
        """
        import numpy as np

        bert = _get_bert()
        if not bert:
            return Intent.UNKNOWN, 0.0

        # Calcule les embeddings au premier appel
        if self._embeddings is None:
            self._embeddings = self._build_embeddings()
        if not self._embeddings:
            return Intent.UNKNOWN, 0.0

        # Encode la question
        q_vec = bert.encode([text], normalize_embeddings=True)[0]

        # Calcule la similarité avec chaque intent
        scores = {}
        for intent_name, intent_vec in self._embeddings.items():
            similarity = float(np.dot(q_vec, intent_vec))
            scores[intent_name] = similarity

        best_intent = max(scores, key=scores.__getitem__)
        best_score  = scores[best_intent]

        # Seuil minimum de confiance
        if best_score < 0.3:
            return Intent.UNKNOWN, best_score

        return best_intent, round(best_score, 3)


# ══════════════════════════════════════════════════════════════════════
# SPACY NER ENGINE
# ══════════════════════════════════════════════════════════════════════

class SpaCyNERExtractor:
    """
    Extraction d'entités nommées via SpaCy fr_core_news_sm.
    Détecte : dates, montants, organisations, lieux, personnes.
    """

    def extract(self, text: str) -> List[Dict]:
        """Extrait les entités nommées du texte via SpaCy."""
        nlp = _get_spacy()
        if not nlp:
            return []

        doc = nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "text":  ent.text,
                "label": ent.label_,  # DATE, MONEY, ORG, LOC, PER, MISC
                "start": ent.start_char,
                "end":   ent.end_char,
                "normalized": self._normalize(ent.text, ent.label_),
            })

        return entities

    def lemmatize(self, text: str) -> str:
        """Lemmatise le texte via SpaCy."""
        nlp = _get_spacy()
        if not nlp:
            return text
        doc = nlp(text)
        return " ".join([
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct
        ])

    def _normalize(self, text: str, label: str) -> Any:
        if label == "DATE":
            return {"type": "date_raw", "value": text}
        elif label == "MONEY":
            num = re.sub(r"[^\d.,]", "", text).replace(",", ".")
            try:
                return {"type": "amount", "value": float(num)}
            except Exception:
                return {"type": "amount", "value": text}
        return text


# ══════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTOR (regex + SpaCy combinés)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedEntity:
    entity_type: str
    value:       str
    normalized:  Any
    start:       int
    end:         int
    confidence:  float = 1.0
    source:      str = "regex"  # "regex" ou "spacy" ou "bert"


DATE_PATTERNS = [
    (r"\b(aujourd'hui|today)\b",           "today"),
    (r"\b(hier|yesterday)\b",              "yesterday"),
    (r"\b(cette?\s+semaine|this\s+week)\b","this_week"),
    (r"\b(ce\s+mois|this\s+month)\b",      "this_month"),
    (r"\b(cette?\s+ann[eé]e|this\s+year)\b","this_year"),
    (r"\b(l[''`]an\s+dernier|last\s+year)\b","last_year"),
    (r"\b(\d{4})\b",                        "year"),
    (r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b","date"),
]

AMOUNT_PATTERNS = [
    (r"\b(\d[\d\s,.]*)(\s*)(k€|k\$|milliers?|thousands?)\b","amount_k"),
    (r"\b(\d[\d\s,.]*)(\s*)(m€|m\$|millions?)\b",           "amount_m"),
    (r"\b(\d[\d\s,.]*)(\s*)(€|\$|eur|usd|euros?|dollars?)\b","amount"),
    (r"\b(\d+(?:[.,]\d+)?)\b",                               "number"),
]

FILTER_PATTERNS = [
    (r"\b(sup[eé]rieur\s+[aà]|greater\s+than|more\s+than|>)\b","gt"),
    (r"\b(inf[eé]rieur\s+[aà]|less\s+than|lower\s+than|<)\b",  "lt"),
    (r"\b([eé]gal\s+[aà]|equal\s+to|=)\b",                     "eq"),
    (r"\b(entre|between)\b",                                     "between"),
    (r"\b(top\s+\d+|les?\s+\d+\s+premiers?)\b",                "top_n"),
]

METRIC_KEYWORDS = {
    "total": "SUM", "somme": "SUM", "sum": "SUM",
    "moyenne": "AVG", "average": "AVG", "avg": "AVG",
    "maximum": "MAX", "max": "MAX",
    "minimum": "MIN", "min": "MIN",
    "nombre": "COUNT", "count": "COUNT", "combien": "COUNT",
}


class HybridEntityExtractor:
    """Extraction d'entités combinant SpaCy NER + patterns regex."""

    def __init__(self):
        self.spacy_ner = SpaCyNERExtractor()

    def extract(self, text: str, known_entities: List[str] = None) -> List[ExtractedEntity]:
        results = []
        t = text.lower()

        # 1. SpaCy NER (dates, montants natifs)
        spacy_ents = self.spacy_ner.extract(text)
        for ent in spacy_ents:
            if ent["label"] in ("DATE", "MONEY"):
                entity_type = "date" if ent["label"] == "DATE" else "amount"
                results.append(ExtractedEntity(
                    entity_type = entity_type,
                    value       = ent["text"],
                    normalized  = ent["normalized"],
                    start       = ent["start"],
                    end         = ent["end"],
                    source      = "spacy",
                ))

        # 2. Patterns regex dates (complément)
        for pattern, date_type in DATE_PATTERNS:
            for m in re.finditer(pattern, t, re.IGNORECASE):
                if not any(e.start <= m.start() <= e.end for e in results):
                    results.append(ExtractedEntity(
                        entity_type = "date",
                        value       = m.group(0),
                        normalized  = self._normalize_date(m.group(0), date_type),
                        start       = m.start(),
                        end         = m.end(),
                        source      = "regex",
                    ))

        # 3. Patterns montants
        for pattern, amount_type in AMOUNT_PATTERNS:
            for m in re.finditer(pattern, t, re.IGNORECASE):
                if not any(e.start <= m.start() <= e.end for e in results):
                    results.append(ExtractedEntity(
                        entity_type = "amount",
                        value       = m.group(0),
                        normalized  = self._normalize_amount(m.group(0), amount_type),
                        start       = m.start(),
                        end         = m.end(),
                        source      = "regex",
                    ))

        # 4. Opérateurs filtre
        for pattern, op in FILTER_PATTERNS:
            for m in re.finditer(pattern, t, re.IGNORECASE):
                results.append(ExtractedEntity(
                    entity_type = "filter_op",
                    value       = m.group(0),
                    normalized  = op,
                    start       = m.start(),
                    end         = m.end(),
                    source      = "regex",
                ))

        # 5. Métriques SQL
        for keyword, sql_func in METRIC_KEYWORDS.items():
            for m in re.finditer(r"\b" + keyword + r"\b", t, re.IGNORECASE):
                results.append(ExtractedEntity(
                    entity_type = "metric",
                    value       = m.group(0),
                    normalized  = sql_func,
                    start       = m.start(),
                    end         = m.end(),
                    source      = "regex",
                ))

        # 6. Entités connues (noms de tables/entités de la source)
        if known_entities:
            for ent in known_entities:
                # Pattern qui gère les underscores (CS_BLTF, GS_BNKBR, etc.)
                esc = re.escape(ent.lower())
                pattern = r"(?<![\w])" + esc + r"(?![\w])"
                for m in re.finditer(pattern, t, re.IGNORECASE):
                    results.append(ExtractedEntity(
                        entity_type = "table",
                        value       = m.group(0),
                        normalized  = ent,
                        start       = m.start(),
                        end         = m.end(),
                        confidence  = 1.0,
                        source      = "schema",
                    ))

        results.sort(key=lambda e: e.start)
        return results

    def _normalize_date(self, value: str, date_type: str) -> Dict:
        now = datetime.now()
        if date_type == "today":
            return {"type": "date", "from": now.date().isoformat(), "to": now.date().isoformat()}
        elif date_type == "yesterday":
            d = (now - timedelta(days=1)).date()
            return {"type": "date", "from": d.isoformat(), "to": d.isoformat()}
        elif date_type == "this_week":
            start = (now - timedelta(days=now.weekday())).date()
            return {"type": "range", "from": start.isoformat(), "to": now.date().isoformat()}
        elif date_type == "this_month":
            return {"type": "range", "from": f"{now.year}-{now.month:02d}-01", "to": now.date().isoformat()}
        elif date_type == "this_year":
            return {"type": "range", "from": f"{now.year}-01-01", "to": now.date().isoformat()}
        elif date_type == "last_year":
            y = now.year - 1
            return {"type": "range", "from": f"{y}-01-01", "to": f"{y}-12-31"}
        elif date_type == "year":
            y = int(value)
            return {"type": "range", "from": f"{y}-01-01", "to": f"{y}-12-31"}
        return {"type": "raw", "value": value}

    def _normalize_amount(self, value: str, amount_type: str) -> float:
        num = re.sub(r"[^\d.,]", "", value).replace(",", ".").strip(".")
        try:
            n = float(num)
            if amount_type == "amount_k":
                return n * 1000
            elif amount_type == "amount_m":
                return n * 1_000_000
            return n
        except Exception:
            return 0.0


# ══════════════════════════════════════════════════════════════════════
# SLOTS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class QuerySlots:
    intent:         str = Intent.UNKNOWN
    source_name:    Optional[str] = None
    table_names:    List[str] = field(default_factory=list)
    field_names:    List[str] = field(default_factory=list)
    date_filter:    Optional[Dict] = None
    amount_filter:  Optional[Dict] = None
    metric:         Optional[str] = None
    group_by:       Optional[str] = None
    filter_op:      Optional[str] = None
    top_n:          Optional[int] = None
    raw_entities:   List[ExtractedEntity] = field(default_factory=list)
    confidence:     float = 0.0
    ambiguities:    List[str] = field(default_factory=list)
    lemmatized:     Optional[str] = None
    nlu_method:     str = "hybrid"  # "bert", "regex", "hybrid"


# ══════════════════════════════════════════════════════════════════════
# CONTEXT MANAGER
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ConversationTurn:
    question:  str
    intent:    str
    slots:     QuerySlots
    answer:    str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ContextManager:
    def __init__(self, max_turns: int = 10):
        self.turns:       List[ConversationTurn] = []
        self.max_turns    = max_turns
        self.last_table:  Optional[str] = None
        self.last_field:  Optional[str] = None
        self.last_source: Optional[str] = None
        # Clarification state — stocke la question en attente
        self.pending_clarification: Optional[Dict] = None
        # Slots en attente de clarification
        self.pending_slots: Optional[Any] = None
        # Question originale avant clarification
        self.pending_question: Optional[str] = None

    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
        if turn.slots.table_names:
            self.last_table = turn.slots.table_names[0]
        if turn.slots.field_names:
            self.last_field = turn.slots.field_names[0]

    def resolve_references(self, text: str) -> str:
        resolved = text
        pronouns = [r"\b(elle|il|cette?\s+table|cette?\s+entit[eé]|it)\b"]
        for p in pronouns:
            if re.search(p, text, re.IGNORECASE) and self.last_table:
                resolved = re.sub(p, self.last_table, resolved, flags=re.IGNORECASE)
        return resolved

    def get_context_summary(self) -> str:
        if not self.turns:
            return ""
        last = self.turns[-1]
        return f"Contexte: {last.intent} sur {self.last_table or 'N/A'}. Q: {last.question[:60]}"


# ══════════════════════════════════════════════════════════════════════
# NLU PIPELINE v2 — Hybride BERT + Regex + SpaCy
# ══════════════════════════════════════════════════════════════════════

class NLUPipeline:
    """
    Pipeline NLU v2 hybride :
    1. Normalisation + Lemmatisation SpaCy
    2. Résolution contextuelle
    3. Intent via BERT (similarité cosinus) + fallback regex
    4. NER via SpaCy + regex
    5. Slot filling
    6. Détection ambiguïtés
    """

    def __init__(self):
        self.fasttext_classifier = FastTextClassifier()
        self.bert_classifier     = BERTIntentClassifier()
        self.roberta_classifier  = RoBERTaClassifier()
        self.entity_extractor    = HybridEntityExtractor()
        self.spacy_ner           = SpaCyNERExtractor()
        self._compiled_patterns: Dict[str, List] = {}
        self._compile_patterns()

    def _compile_patterns(self):
        for intent_name, patterns in INTENT_PATTERNS.items():
            self._compiled_patterns[intent_name] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def normalize(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"qu[''`]", "que ", text)
        text = re.sub(r"l[''`]", "la ", text)
        text = re.sub(r"d[''`]", "de ", text)
        text = re.sub(r"\s+", " ", text)
        # ── Correction fautes de frappe courantes ──────────────
        _typos = {
            r"\bdashbord\b":    "dashboard",
            r"\bdashbord\b":    "dashboard",
            r"\bashboard\b":   "dashboard",
            r"\bdashbord\b":   "dashboard",
            r"\bdashboard\b":   "dashboard",   # normalise casse
            r"\bcamember\b":    "camembert",
            r"\bcamembert\b":   "camembert",
            r"\bvisualise\b":   "visualise",
            r"\bvisualisation\b": "visualisation",
        }
        t_lower = text.lower()
        for pat, repl in _typos.items():
            t_lower = re.sub(pat, repl, t_lower, flags=re.IGNORECASE)
        # Préserve la casse originale sauf pour les mots corrigés
        text = t_lower
        return text

    def detect_intent_regex(self, text: str) -> Tuple[str, float]:
        t = text.lower()
        scores: Dict[str, float] = {}
        for intent_name, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(t):
                    scores[intent_name] = scores.get(intent_name, 0) + 1.0
        if not scores:
            return Intent.UNKNOWN, 0.0
        best = max(scores, key=scores.__getitem__)
        return best, min(scores[best] / 2.0, 1.0)

    def detect_intent(self, text: str) -> Tuple[str, float, str]:
        """
        Pipeline 3 niveaux :
        0. Regex prioritaire  → patterns critiques à confiance maximale
        1. FastText (2ms)     → si confidence >= 0.7 : réponse immédiate
        2. BERT MiniLM (10ms) → si confidence >= 0.55
        3. RoBERTa XLM (50ms) → précision maximale
        4. Regex fallback     → dernier recours
        """
        # Niveau 0 : Regex prioritaire pour patterns critiques
        # Ces patterns ont priorité ABSOLUE sur tous les modèles ML
        _PRIORITY_PATTERNS = [
            # LIST_FIELDS — clés, champs, colonnes, index
            (Intent.LIST_FIELDS, [
                r"cl[eé]s?\s+primaires?",
                r"primary\s+keys?",
                r"cl[eé]s?\s+[eé]trang[eè]res?",
                r"foreign\s+keys?",
                r"liste\s+(les?\s+)?cl[eé]",
                r"liste\s+(les?\s+)?index",
                r"quels?\s+champs?\s+(sont|de|dans)",
                r"champs?\s+disponibles?",
                r"colonnes?\s+disponibles?",
            ]),
            # LLM_EXPLAIN — questions explicatives/conceptuelles → LLM
            (Intent.LLM_EXPLAIN, [
                r"^explique[sz]?\s+(moi\s+)?comment",
                r"^explique[sz]?\s+(moi\s+)?",
                r"^comment\s+(fonctionne|marche|utilise)",
                r"^qu[''e]est[- ]ce\s+(qu[''e]|que)",
                r"^c[''e]est\s+quoi\s+",
                r"^quelle\s+est\s+la\s+diff[eé]rence",
                r"^quels?\s+sont\s+les\s+(avantages?|inconv[eé]nients?|concepts?|principes?)",
                r"(erp|sap|dynamics|odoo|sage)\s+(fonctionne|marche|est)",
                r"^d[eé]finis?\s+(moi\s+)?",
                r"^que\s+signifie\s+",
                r"^expliqu",
                r"comment\s+(fonctionne|marche)",
            ]),
            # GREETING — exactement un mot de salutation
            (Intent.GREETING, [
                r"^(bonjour|salut|hello|hi|coucou|bonsoir|hola|hallo)[\s!]*$",
            ]),
            # SHOW_DASHBOARD — PRIORITÉ ABSOLUE si "dashboard" présent
            # Doit être AVANT GENERATE_AGG pour éviter que "dashboard top 10" → GENERATE_AGG
            (Intent.SHOW_DASHBOARD, [
                r"dash\w*board",        # dashboard, dashbord, dash board...
                r"\bashboard\b",        # ashboard (faute frappe fréquente)
                r"tableau\s+de\s+bord",
                r"(génère|créer?|show|create|generate)\s+(une?\s+)?visualis",
            ]),
            # GENERATE_AGG — top N, classement, ranking (seulement si pas dashboard)
            (Intent.GENERATE_AGG, [
                r"^top\s+\d+",
                r"^les?\s+\d+\s+(premiers?|meilleurs?|plus)",
                r"classement\s+(des?|par)",
                r"ranking\s+by",
            ]),
        ]
        t_lower = text.lower().strip()
        for priority_intent, patterns in _PRIORITY_PATTERNS:
            for pat in patterns:
                if re.search(pat, t_lower, re.IGNORECASE):
                    logger.debug(f"[NLU] Regex prioritaire: {priority_intent} sur '{t_lower[:40]}'")
                    return priority_intent, 1.0, "regex_priority"

        # Niveau 1 : FastText (ultra-rapide)
        ft_intent, ft_conf = self.fasttext_classifier.classify(text)
        if ft_conf >= 0.7:
            logger.debug(f"[NLU v3] FastText: {ft_intent} {ft_conf:.2f}")
            return ft_intent, ft_conf, "fasttext"

        # Niveau 2 : BERT MiniLM
        bert_intent, bert_conf = self.bert_classifier.classify(text)
        if bert_conf >= 0.55:
            if ft_intent == bert_intent and ft_conf >= 0.5:
                boosted = min(bert_conf + 0.1, 1.0)
                return bert_intent, boosted, "fasttext+bert"
            return bert_intent, bert_conf, "bert"

        # Niveau 3 : RoBERTa
        rob_intent, rob_conf = self.roberta_classifier.classify(text)
        if rob_conf >= 0.45:
            return rob_intent, rob_conf, "roberta"

        # Niveau 4 : Regex fallback
        regex_intent, regex_conf = self.detect_intent_regex(text)
        if regex_conf > 0.0:
            return regex_intent, regex_conf, "regex"

        # Dernier recours
        all_results = [
            (ft_intent, ft_conf, "fasttext"),
            (bert_intent, bert_conf, "bert"),
            (rob_intent, rob_conf, "roberta"),
        ]
        best = max(all_results, key=lambda x: x[1])
        if best[1] >= 0.3:
            return best
        return Intent.UNKNOWN, 0.0, "none"

    def extract_group_by(self, text: str) -> Optional[str]:
        patterns = [
            r"par\s+(\w+)", r"by\s+(\w+)",
            r"group\s+(?:by\s+)?(\w+)",
            r"group[eé]\s+par\s+(\w+)",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    def extract_top_n(self, text: str) -> Optional[int]:
        patterns = [
            r"top\s+(\d+)", r"les?\s+(\d+)\s+premiers?",
            r"(\d+)\s+premiers?", r"first\s+(\d+)",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def process(
        self,
        text: str,
        context: Optional[ContextManager] = None,
        known_entities: Optional[List[str]] = None,
    ) -> QuerySlots:
        """Pipeline complet v2."""

        # 1. Normalisation
        text = self.normalize(text)

        # 2. Résolution contextuelle
        if context:
            text = context.resolve_references(text)

        # 3. Lemmatisation SpaCy (pour améliorer matching)
        lemmatized = self.spacy_ner.lemmatize(text)

        # 4. Intent detection hybride BERT + regex
        intent, confidence, method = self.detect_intent(text)

        # Si lemmatisé donne un meilleur résultat, l'utiliser
        if lemmatized and lemmatized != text:
            lem_intent, lem_conf, lem_method = self.detect_intent(lemmatized)
            if lem_conf > confidence:
                intent, confidence, method = lem_intent, lem_conf, f"lemma+{lem_method}"

        # 5. Extraction d'entités hybride SpaCy + regex
        entities = self.entity_extractor.extract(text, known_entities or [])

        # 6. Slot filling
        slots = QuerySlots(
            intent       = intent,
            confidence   = confidence,
            raw_entities = entities,
            lemmatized   = lemmatized,
            nlu_method   = method,
        )

        slots.table_names = [
            e.normalized for e in entities if e.entity_type == "table"
        ]

        date_entities = [e for e in entities if e.entity_type == "date"]
        if date_entities:
            slots.date_filter = date_entities[0].normalized

        amount_entities = [e for e in entities if e.entity_type == "amount"]
        if amount_entities:
            slots.amount_filter = {
                "value": amount_entities[0].normalized,
                "op":    next((e.normalized for e in entities if e.entity_type == "filter_op"), "eq"),
            }

        metric_entities = [e for e in entities if e.entity_type == "metric"]
        if metric_entities:
            slots.metric = metric_entities[0].normalized

        slots.group_by = self.extract_group_by(text)
        slots.top_n    = self.extract_top_n(text)

        filter_ops = [e for e in entities if e.entity_type == "filter_op"]
        if filter_ops:
            slots.filter_op = filter_ops[0].normalized

        # 7. Extraction field_names (CamelCase non-table, non-groupBy)
        import re as _re2
        capitalized = _re2.findall(r'[A-Z][a-z]+[A-Z][a-zA-Z]+', text)
        grp = slots.group_by or ''
        excl_ends = ('ID','Id','Key','Code','Type','Country','City','Region','Name','Date','Status')
        for word in capitalized:
            if (word not in slots.table_names and word not in slots.field_names
                    and word != grp
                    and not any(word.endswith(x) for x in excl_ends)):
                slots.field_names.insert(0, word)

        logger.info(
            f"[NLU v2] intent={intent} conf={confidence:.2f} method={method} "
            f"tables={slots.table_names} metric={slots.metric}"
        )

        return slots


# ══════════════════════════════════════════════════════════════════════
# SINGLETONS
# ══════════════════════════════════════════════════════════════════════

_nlu_pipeline: Optional[NLUPipeline] = None

def get_nlu_pipeline() -> NLUPipeline:
    global _nlu_pipeline
    if _nlu_pipeline is None:
        _nlu_pipeline = NLUPipeline()
        logger.info("[NLU v2] Pipeline initialisé (BERT + SpaCy + Regex)")
    return _nlu_pipeline


_contexts: Dict[str, ContextManager] = {}

def get_context(conv_id: str) -> ContextManager:
    if conv_id not in _contexts:
        _contexts[conv_id] = ContextManager()
    return _contexts[conv_id]

def clear_context(conv_id: str):
    _contexts.pop(conv_id, None)