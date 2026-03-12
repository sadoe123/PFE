"""
OnePilot - Phase 3 : ML Relation Detector (XGBoost)
====================================================
Utilise les FK explicites (explicit_fk + view_join) comme données d'entraînement
et les features de profiling (data_profiler.py) pour prédire de nouvelles relations.

Features d'entrée (par paire source_field → target_field) :
  Structurelles :
    - suffix_score        : col source finit par ID/Code/Num/Key → probable FK
    - prefix_match        : même préfixe table (ex: CS_ ↔ CS_)
    - name_similarity     : Levenshtein normalisé noms colonnes
    - entity_similarity   : similarité noms entités
    - type_compat         : types SQL compatibles (int↔int, varchar↔varchar)

  Statistiques (profiling) :
    - src_cardinality     : nb valeurs uniques / nb lignes source
    - tgt_cardinality     : nb valeurs uniques / nb lignes cible
    - src_null_rate       : taux de nulls colonne source
    - tgt_null_rate       : taux de nulls colonne cible
    - src_avg_len         : longueur moy valeurs source (varchar)
    - tgt_avg_len         : longueur moy valeurs cible
    - range_overlap       : chevauchement min/max (numériques)

  Contextuelles :
    - src_table_size      : nb lignes table source (log10)
    - tgt_table_size      : nb lignes table cible (log10)
    - tgt_is_hub          : cible est une table hub (référencée N fois)
    - module_same         : même préfixe module (AA_, GS_, etc.)

Seuil de confiance minimal pour émettre une prédiction : 0.65
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# STRUCTURES
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MLPrediction:
    source_entity:    str
    source_field:     str
    target_entity:    str
    target_field:     str
    confidence:       float
    detection_method: str = "ml_xgboost"
    relation_type:    str = "many_to_one"
    features:         Dict = dc_field(default_factory=dict)


@dataclass
class TrainingRecord:
    features: Dict[str, float]
    label:    int   # 1 = relation confirmée, 0 = non-relation


# ──────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────

_FK_SUFFIXES = re.compile(
    r"(id|code|cd|num|no|key|ref|fk|type|tp|stat|sts|ccy|cur|cmp|usr|grp|seq|idx|lnk)$",
    re.IGNORECASE,
)
_PREFIX_RE = re.compile(r"^([A-Z]{2,4})_", re.IGNORECASE)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            curr[j] = min(prev[j] + 1, curr[j-1] + 1,
                          prev[j-1] + (0 if a[i-1] == b[j-1] else 1))
        prev = curr
    return prev[lb]


def _name_similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    if a == b:
        return 1.0
    max_len = max(len(a), len(b), 1)
    dist = _levenshtein(a, b)
    return max(0.0, 1.0 - dist / max_len)


def _module_prefix(name: str) -> str:
    m = _PREFIX_RE.match(name)
    return m.group(1).upper() if m else ""


def _suffix_score(col: str) -> float:
    """Score 0-1 selon si le nom de colonne ressemble à une FK."""
    col_low = col.lower().replace("_", "")
    if _FK_SUFFIXES.search(col_low):
        return 1.0
    # colonne entière = ID / code
    if col_low in ("id", "code", "key", "num", "ref"):
        return 0.9
    return 0.1


def _type_compat(src_type: str, tgt_type: str) -> float:
    """Compatibilité de types SQL (normalisés)."""
    def norm(t: str) -> str:
        t = (t or "").lower()
        if any(x in t for x in ("int", "long", "number", "numeric", "decimal", "float", "double", "real")):
            return "numeric"
        if any(x in t for x in ("char", "text", "string", "varchar", "nvarchar", "nchar")):
            return "string"
        if any(x in t for x in ("date", "time", "timestamp")):
            return "datetime"
        return "other"
    nt, ns = norm(tgt_type), norm(src_type)
    if nt == ns:
        return 1.0
    # numeric ↔ string : parfois les codes sont stockés des deux côtés
    if {nt, ns} == {"numeric", "string"}:
        return 0.3
    return 0.0


def _build_features(
    src_entity: str, src_field: str, src_field_meta: Dict,
    tgt_entity: str, tgt_field: str, tgt_field_meta: Dict,
    src_profile: Optional[Dict], tgt_profile: Optional[Dict],
    hub_counts: Dict[str, int],
) -> Dict[str, float]:

    feats: Dict[str, float] = {}

    # ── Structurelles ──
    feats["suffix_score"]     = _suffix_score(src_field)
    feats["name_similarity"]  = _name_similarity(src_field, tgt_field)
    feats["entity_similarity"]= _name_similarity(src_entity, tgt_entity)

    sm = _module_prefix(src_entity)
    tm = _module_prefix(tgt_entity)
    feats["module_same"]      = 1.0 if (sm and sm == tm) else 0.0
    feats["prefix_match"]     = 1.0 if (sm and tm and sm == tm) else 0.5

    feats["type_compat"] = _type_compat(
        src_field_meta.get("data_type", ""),
        tgt_field_meta.get("data_type", ""),
    )

    # ── Hub ──
    max_hub = max(hub_counts.values(), default=1)
    feats["tgt_is_hub"] = min(1.0, hub_counts.get(tgt_entity, 0) / max(max_hub, 1))

    # ── Profiling source ──
    if src_profile:
        col_stats = src_profile.get("columns", {}).get(src_field, {})
        row_count = max(src_profile.get("row_count", 1), 1)
        unique    = col_stats.get("distinct_count", 0) or 0
        feats["src_cardinality"] = min(1.0, unique / row_count)
        feats["src_null_rate"]   = col_stats.get("null_rate", 0.0) or 0.0
        feats["src_avg_len"]     = min(1.0, (col_stats.get("avg_length", 0) or 0) / 50.0)
        feats["src_table_size"]  = math.log10(max(row_count, 1)) / 8.0
    else:
        feats.update(src_cardinality=0.5, src_null_rate=0.0,
                     src_avg_len=0.5, src_table_size=0.5)

    # ── Profiling cible ──
    if tgt_profile:
        col_stats = tgt_profile.get("columns", {}).get(tgt_field, {})
        row_count = max(tgt_profile.get("row_count", 1), 1)
        unique    = col_stats.get("distinct_count", 0) or 0
        feats["tgt_cardinality"] = min(1.0, unique / row_count)
        feats["tgt_null_rate"]   = col_stats.get("null_rate", 0.0) or 0.0
        feats["tgt_avg_len"]     = min(1.0, (col_stats.get("avg_length", 0) or 0) / 50.0)
        feats["tgt_table_size"]  = math.log10(max(row_count, 1)) / 8.0
    else:
        feats.update(tgt_cardinality=0.5, tgt_null_rate=0.0,
                     tgt_avg_len=0.5, tgt_table_size=0.5)

    # ── Range overlap (si disponible) ──
    src_min = src_profile.get("columns", {}).get(src_field, {}).get("min_value") if src_profile else None
    src_max = src_profile.get("columns", {}).get(src_field, {}).get("max_value") if src_profile else None
    tgt_min = tgt_profile.get("columns", {}).get(tgt_field, {}).get("min_value") if tgt_profile else None
    tgt_max = tgt_profile.get("columns", {}).get(tgt_field, {}).get("max_value") if tgt_profile else None
    try:
        sm_v, sx_v = float(src_min), float(src_max)
        tm_v, tx_v = float(tgt_min), float(tgt_max)
        overlap = max(0.0, min(sx_v, tx_v) - max(sm_v, tm_v))
        total   = max(max(sx_v, tx_v) - min(sm_v, tm_v), 1.0)
        feats["range_overlap"] = overlap / total
    except (TypeError, ValueError, ZeroDivisionError):
        feats["range_overlap"] = 0.5  # inconnu → neutre

    return feats


# ──────────────────────────────────────────────────────────────────────
# FEATURE VECTOR (ordre stable pour XGBoost)
# ──────────────────────────────────────────────────────────────────────

FEATURE_ORDER = [
    "suffix_score", "name_similarity", "entity_similarity",
    "module_same", "prefix_match", "type_compat",
    "src_cardinality", "src_null_rate", "src_avg_len", "src_table_size",
    "tgt_cardinality", "tgt_null_rate", "tgt_avg_len", "tgt_table_size",
    "tgt_is_hub", "range_overlap",
]


def _to_vector(feats: Dict[str, float]) -> List[float]:
    return [feats.get(k, 0.0) for k in FEATURE_ORDER]


# ──────────────────────────────────────────────────────────────────────
# MODÈLE DE FALLBACK (heuristique pondérée) si XGBoost absent
# ──────────────────────────────────────────────────────────────────────

_WEIGHTS = {
    "suffix_score":     0.25,
    "name_similarity":  0.20,
    "type_compat":      0.15,
    "src_cardinality":  0.10,
    "tgt_is_hub":       0.10,
    "range_overlap":    0.08,
    "module_same":      0.06,
    "entity_similarity":0.04,
    "src_null_rate":   -0.02,  # null élevé = moins fiable
}


def _heuristic_score(feats: Dict[str, float]) -> float:
    score = sum(feats.get(k, 0.0) * w for k, w in _WEIGHTS.items())
    # Normalise [0,1]
    return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────

async def train_model(
    source_id: UUID,
    db,           # asyncpg connection pool
    profiles: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """
    Entraîne un modèle XGBoost (ou heuristique si xgboost absent) sur les
    relations CONFIRMÉES de la source (explicit_fk, view_join, validated).
    Retourne un objet modèle sérialisé + métriques.
    """
    try:
        import xgboost as xgb
        import numpy as np
        HAS_XGB = True
        logger.info("[ML] XGBoost disponible — entraînement réel")
    except ImportError:
        HAS_XGB = False
        logger.warning("[ML] XGBoost absent — mode heuristique pondérée activé")

    # 1. Charger les relations confirmées (positives)
    positive_rows = await db.fetch("""
        SELECT source_entity, source_field, target_entity, target_field,
               detection_method, confidence
        FROM entity_relations
        WHERE source_id = $1
          AND (detection_method IN ('explicit_fk','view_join')
               OR (is_confirmed = TRUE AND detection_method != 'ml_xgboost'))
    """, source_id)

    if len(positive_rows) < 10:
        return {
            "status": "insufficient_data",
            "message": f"Seulement {len(positive_rows)} relations confirmées — minimum 10 requis pour l'entraînement",
            "positive_count": len(positive_rows),
            "model_type": "none",
        }

    # 2. Charger métadonnées champs via source_entities + entity_fields
    fields_rows = await db.fetch("""
        SELECT se.name AS entity_name, ef.name AS field_name,
               ef.data_type
        FROM source_entities se
        JOIN entity_fields ef ON ef.entity_id = se.id
        WHERE se.source_id = $1
        ORDER BY se.name, ef.position
    """, source_id)

    fields_meta: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for row in fields_rows:
        fields_meta[row["entity_name"]][row["field_name"]] = {
            "data_type": row["data_type"] or "",
        }

    # 3. Calcul hub counts (tables les plus référencées = hubs)
    hub_counts: Dict[str, int] = defaultdict(int)
    for row in positive_rows:
        hub_counts[row["target_entity"]] += 1

    # 4. Construire exemples positifs
    records: List[TrainingRecord] = []
    for row in positive_rows:
        src_e, src_f = row["source_entity"], row["source_field"]
        tgt_e, tgt_f = row["target_entity"], row["target_field"]
        src_fm = fields_meta.get(src_e, {}).get(src_f, {})
        tgt_fm = fields_meta.get(tgt_e, {}).get(tgt_f, {})
        sp = (profiles or {}).get(src_e)
        tp = (profiles or {}).get(tgt_e)
        feats = _build_features(src_e, src_f, src_fm, tgt_e, tgt_f, tgt_fm, sp, tp, hub_counts)
        records.append(TrainingRecord(features=feats, label=1))

    # 5. Générer exemples négatifs (paires aléatoires non-confirmées)
    import random
    all_entities = list(fields_meta.keys())
    positive_set = {(r["source_entity"], r["source_field"], r["target_entity"], r["target_field"])
                    for r in positive_rows}
    neg_count = 0
    attempts  = 0
    neg_target = min(len(records) * 3, 500)  # ratio 1:3 max

    random.seed(42)
    while neg_count < neg_target and attempts < neg_target * 10:
        attempts += 1
        se = random.choice(all_entities)
        te = random.choice(all_entities)
        if se == te:
            continue
        se_fields = list(fields_meta.get(se, {}).keys())
        te_fields = list(fields_meta.get(te, {}).keys())
        if not se_fields or not te_fields:
            continue
        sf = random.choice(se_fields)
        tf = random.choice(te_fields)
        if (se, sf, te, tf) in positive_set:
            continue
        # Exclure les colonnes qui ressemblent à des FK (sinon trop facile à distinguer)
        if _suffix_score(sf) > 0.8 and _name_similarity(sf, tf) > 0.7:
            continue
        src_fm = fields_meta.get(se, {}).get(sf, {})
        tgt_fm = fields_meta.get(te, {}).get(tf, {})
        sp = (profiles or {}).get(se)
        tp = (profiles or {}).get(te)
        feats = _build_features(se, sf, src_fm, te, tf, tgt_fm, sp, tp, hub_counts)
        records.append(TrainingRecord(features=feats, label=0))
        neg_count += 1

    logger.info(f"[ML] Training: {len(records)} exemples ({len(positive_rows)} positifs, {neg_count} négatifs)")

    if HAS_XGB:
        import numpy as np
        X = np.array([_to_vector(r.features) for r in records], dtype=float)
        y = np.array([r.label for r in records], dtype=float)

        # Split train/val 80/20
        n     = len(X)
        idx   = list(range(n))
        random.shuffle(idx)
        split = int(n * 0.8)
        tr_idx, va_idx = idx[:split], idx[split:]

        dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx], feature_names=FEATURE_ORDER)
        dval   = xgb.DMatrix(X[va_idx], label=y[va_idx], feature_names=FEATURE_ORDER)

        params = {
            "objective":       "binary:logistic",
            "eval_metric":     "auc",
            "max_depth":       5,
            "learning_rate":   0.1,
            "n_estimators":    200,
            "subsample":       0.8,
            "colsample_bytree":0.8,
            "min_child_weight":2,
            "seed":            42,
            "verbosity":       0,
        }
        model = xgb.train(
            params, dtrain, num_boost_round=200,
            evals=[(dval, "val")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )
        val_preds = model.predict(dval)
        val_labels = y[va_idx]
        # AUC approximée
        pairs = sorted(zip(val_preds, val_labels), reverse=True)
        pos_count = int(sum(val_labels))
        neg_c     = len(val_labels) - pos_count
        auc = 0.0
        if pos_count and neg_c:
            tp_acc = 0
            for _, lbl in pairs:
                if lbl == 1:
                    tp_acc += 1
                else:
                    auc += tp_acc
            auc /= (pos_count * neg_c)

        # Feature importance
        importance = model.get_score(importance_type="gain")
        model_bytes = model.save_raw()

        return {
            "status":          "trained",
            "model_type":      "xgboost",
            "positive_count":  len(positive_rows),
            "negative_count":  neg_count,
            "total_examples":  len(records),
            "val_auc":         round(auc, 4),
            "best_iteration":  model.best_iteration,
            "feature_importance": importance,
            "_model_obj":      model,   # objet en mémoire pour predict
            "_hub_counts":     hub_counts,
            "_fields_meta":    fields_meta,
        }
    else:
        # Heuristique : calibration sur les positifs
        pos_scores  = [_heuristic_score(r.features) for r in records if r.label == 1]
        neg_scores  = [_heuristic_score(r.features) for r in records if r.label == 0]
        avg_pos     = sum(pos_scores) / max(len(pos_scores), 1)
        avg_neg     = sum(neg_scores) / max(len(neg_scores), 1)
        return {
            "status":          "trained",
            "model_type":      "heuristic",
            "positive_count":  len(positive_rows),
            "negative_count":  neg_count,
            "total_examples":  len(records),
            "avg_pos_score":   round(avg_pos, 4),
            "avg_neg_score":   round(avg_neg, 4),
            "_hub_counts":     hub_counts,
            "_fields_meta":    fields_meta,
        }


# ──────────────────────────────────────────────────────────────────────
# PRÉDICTION
# ──────────────────────────────────────────────────────────────────────

async def predict_relations(
    source_id:       UUID,
    model_info:      Dict[str, Any],
    db,
    profiles:        Optional[Dict[str, Dict]] = None,
    min_confidence:  float = 0.65,
    max_candidates:  int   = 500,
) -> List[MLPrediction]:
    """
    Parcourt les paires de colonnes non encore liées et prédit de nouvelles
    relations. Retourne une liste triée par confidence décroissante.
    """
    hub_counts  = model_info.get("_hub_counts", {})
    fields_meta = model_info.get("_fields_meta", {})
    model_obj   = model_info.get("_model_obj")        # XGBoost ou None
    model_type  = model_info.get("model_type", "heuristic")

    # Relations déjà connues → ne pas les re-prédire
    existing = await db.fetch("""
        SELECT source_entity, source_field, target_entity, target_field
        FROM entity_relations WHERE source_id = $1
    """, source_id)
    known_set = {(r["source_entity"], r["source_field"], r["target_entity"], r["target_field"])
                 for r in existing}

    # Colonnes potentielles FK (suffix_score > 0.5)
    fk_candidates: List[Tuple[str, str]] = []
    for entity, fmap in fields_meta.items():
        for field in fmap:
            if _suffix_score(field) >= 0.5:
                fk_candidates.append((entity, field))

    # Tables cibles (hubs en priorité)
    target_entities = sorted(
        fields_meta.keys(),
        key=lambda e: hub_counts.get(e, 0),
        reverse=True
    )

    predictions: List[MLPrediction] = []

    try:
        import xgboost as xgb
        import numpy as np
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    # Batch build features
    batch_feats = []
    batch_meta  = []

    for src_e, src_f in fk_candidates:
        src_fm = fields_meta.get(src_e, {}).get(src_f, {})
        sp     = (profiles or {}).get(src_e)

        for tgt_e in target_entities:
            if tgt_e == src_e:
                continue
            # Trouver le meilleur champ cible (PK ou champ similaire)
            tgt_fields = list(fields_meta.get(tgt_e, {}).keys())
            # Prioriser : champ nommé ID/Code, ou similaire au champ source
            tgt_fields_sorted = sorted(
                tgt_fields,
                key=lambda f: (
                    -_suffix_score(f) * 0.5 +
                    -_name_similarity(src_f, f) * 0.5
                )
            )
            # Prendre uniquement le top-2 champs cibles par table
            for tgt_f in tgt_fields_sorted[:2]:
                if (src_e, src_f, tgt_e, tgt_f) in known_set:
                    continue
                tgt_fm = fields_meta.get(tgt_e, {}).get(tgt_f, {})
                tp     = (profiles or {}).get(tgt_e)
                feats  = _build_features(src_e, src_f, src_fm, tgt_e, tgt_f, tgt_fm, sp, tp, hub_counts)
                batch_feats.append(feats)
                batch_meta.append((src_e, src_f, tgt_e, tgt_f))

                if len(batch_feats) >= max_candidates * 5:
                    break
            if len(batch_feats) >= max_candidates * 5:
                break
        if len(batch_feats) >= max_candidates * 5:
            break

    if not batch_feats:
        return []

    # Scoring
    if HAS_XGB and model_obj:
        import numpy as np
        dtest  = xgb.DMatrix(
            np.array([_to_vector(f) for f in batch_feats]),
            feature_names=FEATURE_ORDER
        )
        scores = model_obj.predict(dtest).tolist()
    else:
        scores = [_heuristic_score(f) for f in batch_feats]

    # Filtrer et trier
    results = []
    for (src_e, src_f, tgt_e, tgt_f), feats, score in zip(batch_meta, batch_feats, scores):
        if score >= min_confidence:
            results.append(MLPrediction(
                source_entity=src_e, source_field=src_f,
                target_entity=tgt_e, target_field=tgt_f,
                confidence=round(float(score), 4),
                detection_method=f"ml_{model_type}",
                features={**feats, "model_type": model_type},
            ))

    results.sort(key=lambda p: p.confidence, reverse=True)
    return results[:max_candidates]


# ──────────────────────────────────────────────────────────────────────
# SAUVEGARDER LES PRÉDICTIONS EN DB
# ──────────────────────────────────────────────────────────────────────

async def save_ml_predictions(
    source_id:   UUID,
    predictions: List[MLPrediction],
    db,
) -> Dict[str, int]:
    inserted = 0
    skipped  = 0
    for p in predictions:
        try:
            await db.execute("""
                INSERT INTO entity_relations
                  (source_id, source_entity, source_field,
                   target_entity, target_field,
                   relation_type, confidence, detection_method,
                   is_confirmed, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,FALSE,NOW())
                ON CONFLICT (source_id, source_entity, source_field, target_entity, target_field)
                DO NOTHING
            """,
                source_id,
                p.source_entity, p.source_field,
                p.target_entity, p.target_field,
                p.relation_type, p.confidence, p.detection_method,
            )
            inserted += 1
        except Exception as exc:
            logger.debug(f"[ML] Skip {p.source_entity}.{p.source_field}→{p.target_entity}.{p.target_field}: {exc}")
            skipped += 1
    return {"inserted": inserted, "skipped": skipped}