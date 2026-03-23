# ======================================================================
# ml_detector.py — OnePilot ML Relation Detector
# Extrait du notebook ml_relation_detector_v5
# ======================================================================
# Fonctions exposées :
#   train_ml_model(source_id, db_pool)   → entraîne + sauvegarde .pkl
#   predict_ml_relations(source_id, db_pool) → prédit + sauvegarde en DB
#   get_ml_status(source_id)             → statut du modèle
# ======================================================================

import re
import pickle
import logging
import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from uuid import UUID

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logger = logging.getLogger(__name__)

# ── Répertoire de sauvegarde des modèles ──
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Features utilisées ──
FEATURE_COLS = [
    'name_sim', 'norm_sim', 'entity_in_field', 'type_compat',
    'fk_pattern_a', 'fk_pattern_b', 'pk_fk_pair', 'common_parts',
    'name_contains_entity_a', 'name_contains_entity_b',
    'len_diff', 'prefix_match', 'suffix_match'
]

MAX_PREDICTIONS = 500


# ======================================================================
# FEATURE ENGINEERING
# ======================================================================

FK_PATTERNS = ['_id', '_fk', 'id_', 'fk_', '_code', '_num', '_no', '_key', '_ref']
TYPE_GROUPS = {
    'int':  ['int', 'integer', 'bigint', 'smallint', 'tinyint', 'numeric', 'decimal', 'number'],
    'str':  ['varchar', 'nvarchar', 'char', 'nchar', 'text', 'ntext', 'string'],
    'date': ['date', 'datetime', 'datetime2', 'timestamp'],
}


def _normalize(name: str) -> str:
    n = name.lower()
    for p in ['fk_', 'pk_', 'id_', 'num_', 'cod_', 'f_', 'c_']:
        if n.startswith(p):
            n = n[len(p):]
            break
    for s in ['_id', '_fk', '_pk', '_key', '_code', '_num', '_no', '_ref']:
        if n.endswith(s):
            n = n[:-len(s)]
            break
    return n


def _type_group(dtype: str) -> str:
    d = dtype.lower()
    return next((g for g, ts in TYPE_GROUPS.items() if any(t in d for t in ts)), 'other')


def _fk_pat(name: str) -> float:
    n = name.lower()
    return float(any(n.startswith(p) or n.endswith(p) for p in FK_PATTERNS))


def _sim(a: str, b: str) -> float:
    """Similarité Jaccard sur les bigrams — O(n), rapide."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = set(a[i:i + 2] for i in range(len(a) - 1))
    sb = set(b[i:i + 2] for i in range(len(b) - 1))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _camel_parts(name: str) -> set:
    return set(p.lower() for p in re.sub(r'([A-Z])', r' \1', name).split() if len(p) > 1)


def _compute_features(ea, fa, dta, is_pk_a, is_fk_a,
                       eb, fb, dtb, is_pk_b, is_fk_b) -> dict:
    na, nb = _normalize(fa), _normalize(fb)
    ea_n, eb_n = _normalize(ea), _normalize(eb)
    parts_a = _camel_parts(fa)
    parts_b = _camel_parts(fb)
    common = len(parts_a & parts_b) / max(len(parts_a | parts_b), 1)

    return {
        'name_sim':               _sim(fa, fb),
        'norm_sim':               _sim(na, nb),
        'entity_in_field':        float(ea_n in nb or eb_n in na
                                        or _sim(ea_n, nb) > 0.7
                                        or _sim(eb_n, na) > 0.7),
        'type_compat':            float(_type_group(dta) == _type_group(dtb)),
        'fk_pattern_a':           _fk_pat(fa),
        'fk_pattern_b':           _fk_pat(fb),
        'pk_fk_pair':             float((is_pk_a and is_fk_b) or (is_pk_b and is_fk_a)),
        'common_parts':           common,
        'name_contains_entity_a': float(ea_n in fa.lower()),
        'name_contains_entity_b': float(eb_n in fb.lower()),
        'len_diff':               abs(len(fa) - len(fb)) / max(len(fa), len(fb), 1),
        'prefix_match':           float(fa[:3].lower() == fb[:3].lower()),
        'suffix_match':           float(fa[-3:].lower() == fb[-3:].lower()),
    }


# ======================================================================
# CHARGEMENT DES DONNÉES
# ======================================================================

async def _load_data(source_id: UUID, pool) -> tuple:
    """Charge les champs et relations depuis la DB."""
    async with pool.acquire() as conn:
        fields_rows = await conn.fetch("""
            SELECT se.name AS entity_name,
                   ef.name AS field_name, ef.data_type,
                   ef.is_primary_key, ef.is_foreign_key
            FROM source_entities se
            JOIN entity_fields ef ON ef.entity_id = se.id
            WHERE se.source_id = $1
            ORDER BY se.name, ef.position
        """, source_id)

        relations_rows = await conn.fetch("""
            SELECT source_entity, source_field, target_entity, target_field,
                   detection_method,
                   CASE
                       WHEN detection_method = 'explicit_fk'  THEN 1.0
                       WHEN detection_method = 'view_join'    THEN 1.0
                       WHEN detection_method = 'name_pascal'  THEN 0.8
                       WHEN detection_method = 'name_m2m'     THEN 0.7
                       WHEN detection_method = 'fuzzy_match'  THEN 0.5
                       ELSE 0.6
                   END as sample_weight
            FROM entity_relations
            WHERE source_id = $1
              AND detection_method != 'ml_predicted'
            ORDER BY sample_weight DESC
        """, source_id)

    df_fields = pd.DataFrame([dict(r) for r in fields_rows])
    df_relations = pd.DataFrame([dict(r) for r in relations_rows])
    return df_fields, df_relations


# ======================================================================
# CONSTRUCTION DU DATASET
# ======================================================================

def _build_dataset(df_fields: pd.DataFrame, df_relations: pd.DataFrame) -> pd.DataFrame:
    """Construit le dataset positifs + négatifs."""
    entity_map = defaultdict(list)
    for _, r in df_fields.iterrows():
        entity_map[r['entity_name']].append(r)

    pk_map = {}
    for _, r in df_fields.iterrows():
        if r['is_primary_key']:
            pk_map.setdefault(r['entity_name'], []).append(r)

    positive_set = set(zip(df_relations['source_entity'], df_relations['target_entity']))

    # ── POSITIFS ──
    pos_samples = []
    for _, rel in df_relations.iterrows():
        src_fields = entity_map.get(rel['source_entity'], [])
        tgt_pks = pk_map.get(rel['target_entity'], [])
        if not src_fields:
            continue

        fk_field = next((f for f in src_fields if f['is_foreign_key']), None)
        if fk_field is None:
            fk_field = next((f for f in src_fields if _fk_pat(f['field_name']) > 0), None)
        if fk_field is None:
            fk_field = src_fields[0]

        tgt_field = tgt_pks[0] if tgt_pks else (
            entity_map[rel['target_entity']][0] if entity_map.get(rel['target_entity']) else None
        )
        if tgt_field is None:
            continue

        feat = _compute_features(
            fk_field['entity_name'], fk_field['field_name'], fk_field['data_type'],
            fk_field['is_primary_key'], fk_field['is_foreign_key'],
            tgt_field['entity_name'], tgt_field['field_name'], tgt_field['data_type'],
            tgt_field['is_primary_key'], tgt_field['is_foreign_key']
        )
        feat.update({
            'source_entity': rel['source_entity'],
            'target_entity': rel['target_entity'],
            'label': 1,
            'sample_weight': float(rel['sample_weight'])
        })
        pos_samples.append(feat)

    # ── NÉGATIFS : ratio 1:1 ──
    all_entities = list(entity_map.keys())
    target_neg = min(len(pos_samples), 5000)  # Cap pour éviter explosion sur grandes sources
    target_random = int(target_neg * 0.70)
    target_hard = int(target_neg * 0.30)
    neg_samples = []
    np.random.seed(42)

    attempts = 0
    while len(neg_samples) < target_random and attempts < target_random * 20:
        attempts += 1
        ea = all_entities[np.random.randint(len(all_entities))]
        eb = all_entities[np.random.randint(len(all_entities))]
        if ea == eb:
            continue
        if (ea, eb) in positive_set or (eb, ea) in positive_set:
            continue
        fa = entity_map[ea][np.random.randint(len(entity_map[ea]))]
        fb = entity_map[eb][np.random.randint(len(entity_map[eb]))]
        feat = _compute_features(
            fa['entity_name'], fa['field_name'], fa['data_type'],
            fa['is_primary_key'], fa['is_foreign_key'],
            fb['entity_name'], fb['field_name'], fb['data_type'],
            fb['is_primary_key'], fb['is_foreign_key']
        )
        feat.update({'source_entity': ea, 'target_entity': eb,
                     'label': 0, 'sample_weight': 1.0})
        neg_samples.append(feat)

    # Hard negatives
    fk_fields_all = [r for _, r in df_fields.iterrows() if _fk_pat(r['field_name']) > 0]
    pk_fields_all = [r for _, r in df_fields.iterrows() if r['is_primary_key']]
    np.random.shuffle(fk_fields_all)
    hard_count = 0

    for fa in fk_fields_all:
        if hard_count >= target_hard:
            break
        for fb in pk_fields_all:
            if hard_count >= target_hard:
                break
            if fa['entity_name'] == fb['entity_name']:
                continue
            if (fa['entity_name'], fb['entity_name']) in positive_set:
                continue
            if _sim(fa['field_name'], fb['field_name']) > 0.5:
                feat = _compute_features(
                    fa['entity_name'], fa['field_name'], fa['data_type'],
                    fa['is_primary_key'], fa['is_foreign_key'],
                    fb['entity_name'], fb['field_name'], fb['data_type'],
                    fb['is_primary_key'], fb['is_foreign_key']
                )
                feat.update({'source_entity': fa['entity_name'],
                             'target_entity': fb['entity_name'],
                             'label': 0, 'sample_weight': 1.0})
                neg_samples.append(feat)
                hard_count += 1

    df = pd.DataFrame(pos_samples + neg_samples).reset_index(drop=True)
    logger.info(f"Dataset: {len(df)} échantillons "
                f"({df['label'].sum()} positifs, {(df['label']==0).sum()} négatifs)")
    return df, entity_map, pk_map, positive_set


# ======================================================================
# ENTRAÎNEMENT
# ======================================================================

async def train_ml_model(source_id: UUID, pool) -> dict:
    """
    Entraîne un modèle ML pour la source donnée.
    Sauvegarde le modèle dans MODEL_DIR/{source_id}.pkl
    Retourne les métriques.
    """
    logger.info(f"[ML Train] Démarrage pour source {source_id}")

    # Chargement données
    df_fields, df_relations = await _load_data(source_id, pool)
    if df_relations.empty:
        raise ValueError("Pas assez de relations pour entraîner le modèle")

    # Dataset
    df_dataset, entity_map, pk_map, positive_set = _build_dataset(df_fields, df_relations)

    X = df_dataset[FEATURE_COLS].values
    y = df_dataset['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entraînement des modèles
    models = {}

    rf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                     learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    models['GradientBoosting'] = gb

    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model

    # Sélection meilleur modèle
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'model':   model,
            'f1':      f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'prec':    precision_score(y_test, y_pred),
            'rec':     recall_score(y_test, y_pred),
            'y_proba': y_proba,
        }

    best_name = max(results, key=lambda n: results[n]['f1'])
    best = results[best_name]

    # Seuil optimal via PR curve
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, best['y_proba'])
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    threshold = float(thresholds[best_idx])

    # Sauvegarde modèle
    model_path = MODEL_DIR / f"{source_id}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model':        best['model'],
            'model_name':   best_name,
            'feature_cols': FEATURE_COLS,
            'source_id':    str(source_id),
            'threshold':    threshold,
            'trained_at':   datetime.utcnow().isoformat(),
            'metrics': {
                'f1':      best['f1'],
                'roc_auc': best['roc_auc'],
                'precision': best['prec'],
                'recall':    best['rec'],
            }
        }, f)

    logger.info(f"[ML Train] ✅ {best_name} F1={best['f1']:.4f} sauvegardé → {model_path}")

    # Feature importance (RandomForest / XGBoost)
    feature_importance = {}
    if hasattr(best['model'], 'feature_importances_'):
        fi = best['model'].feature_importances_
        feature_importance = {
            col: round(float(fi[i]), 4)
            for i, col in enumerate(FEATURE_COLS)
        }

    return {
        'model_type':         best_name,
        'model_name':         best_name,
        'positive_count':     int(df_dataset['label'].sum()),
        'n_positifs':         int(df_dataset['label'].sum()),
        'n_negatifs':         int((df_dataset['label'] == 0).sum()),
        'val_auc':            round(best['roc_auc'], 4),
        'f1':                 round(best['f1'], 4),
        'roc_auc':            round(best['roc_auc'], 4),
        'precision':          round(best['prec'], 4),
        'recall':             round(best['rec'], 4),
        'threshold':          round(threshold, 4),
        'trained_at':         datetime.utcnow().isoformat(),
        'feature_importance': feature_importance,
    }


# ======================================================================
# PRÉDICTION
# ======================================================================

async def predict_ml_relations(source_id: UUID, pool) -> dict:
    """
    Prédit les nouvelles relations pour la source donnée.
    Supprime les anciennes prédictions ML et insère les nouvelles.
    """
    model_path = MODEL_DIR / f"{source_id}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé pour {source_id}. Entraînez d'abord.")

    # Charger le modèle
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    model = saved['model']
    threshold = saved['threshold']

    logger.info(f"[ML Predict] Modèle {saved['model_name']} chargé (seuil={threshold:.3f})")

    # Charger les données
    df_fields, df_relations = await _load_data(source_id, pool)
    _, entity_map, pk_map, positive_set = _build_dataset(df_fields, df_relations)

    # Générer les paires candidates
    fk_cands = [r for _, r in df_fields.iterrows()
                if r['is_foreign_key'] or _fk_pat(r['field_name']) > 0]
    pk_cands = list(df_fields[df_fields['is_primary_key'] == True].iterrows())

    logger.info(f"[ML Predict] {len(fk_cands)} FK × {len(pk_cands)} PK candidates")

    unknown_pairs = []
    for fa in fk_cands:
        for _, fb in pk_cands:
            if fa['entity_name'] == fb['entity_name']:
                continue
            if (fa['entity_name'], fb['entity_name']) in positive_set:
                continue
            feat = _compute_features(
                fa['entity_name'], fa['field_name'], fa['data_type'],
                fa['is_primary_key'], fa['is_foreign_key'],
                fb['entity_name'], fb['field_name'], fb['data_type'],
                fb['is_primary_key'], fb['is_foreign_key']
            )
            feat.update({
                'source_entity': fa['entity_name'], 'source_field': fa['field_name'],
                'target_entity': fb['entity_name'], 'target_field': fb['field_name']
            })
            unknown_pairs.append(feat)

    if not unknown_pairs:
        return {'inserted': 0, 'message': 'Aucune paire candidate trouvée'}

    df_unknown = pd.DataFrame(unknown_pairs)
    X_unknown = df_unknown[FEATURE_COLS].values
    proba = model.predict_proba(X_unknown)[:, 1]
    df_unknown['confidence'] = proba

    # Filtrer + dédupliquer
    df_preds = (
        df_unknown[df_unknown['confidence'] >= threshold]
        .sort_values('confidence', ascending=False)
        .groupby(['source_entity', 'source_field'])
        .first()
        .reset_index()
        .sort_values('confidence', ascending=False)
        .head(MAX_PREDICTIONS)
    )

    logger.info(f"[ML Predict] {len(df_preds)} relations prédites (seuil={threshold:.3f})")

    # Sauvegarder en DB
    async with pool.acquire() as conn:
        # Supprimer anciennes prédictions
        await conn.execute("""
            DELETE FROM entity_relations
            WHERE source_id = $1 AND detection_method = 'ml_predicted'
        """, source_id)

        # Insérer nouvelles
        inserted = errors = 0
        for _, row in df_preds.iterrows():
            try:
                await conn.execute("""
                    INSERT INTO entity_relations
                        (source_id, source_entity, source_field,
                         target_entity, target_field,
                         detection_method, confidence)
                    VALUES ($1, $2, $3, $4, $5, 'ml_predicted', $6)
                """,
                    source_id,
                    row['source_entity'], row['source_field'],
                    row['target_entity'], row['target_field'],
                    float(row['confidence'])
                )
                inserted += 1
            except Exception as e:
                errors += 1
                logger.warning(f"[ML Predict] Insert error: {e}")

    return {
        'inserted':           inserted,
        'errors':             errors,
        'threshold':          round(threshold, 4),
        'avg_confidence':     round(float(df_preds['confidence'].mean()), 4),
        'model_name':         saved['model_name'],
        'predicted_at':       datetime.utcnow().isoformat(),
    }


# ======================================================================
# STATUT
# ======================================================================

def get_ml_status(source_id: UUID) -> dict:
    """Retourne le statut du modèle ML pour une source."""
    model_path = MODEL_DIR / f"{source_id}.pkl"

    if not model_path.exists():
        return {
            'available':  False,
            'message':    'Modèle non entraîné. Lancez POST /sources/{id}/ml/train',
        }

    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    return {
        'available':   True,
        'model_name':  saved.get('model_name', 'unknown'),
        'trained_at':  saved.get('trained_at'),
        'threshold':   saved.get('threshold'),
        'metrics':     saved.get('metrics', {}),
        'feature_cols': saved.get('feature_cols', []),
    }