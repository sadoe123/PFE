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
_HERE = Path(__file__).parent

def _get_model_dir() -> Path:
    """Retourne un dossier accessible en écriture pour les modèles."""
    # Priorité : variable d'env → /app/models → dossier du fichier → /tmp/models
    candidates = [
        Path(os.getenv("MODEL_DIR", "")),
        Path("/app/models"),
        _HERE / "models",
        Path("/tmp/models"),
    ]
    for p in candidates:
        if not str(p): continue
        try:
            p.mkdir(parents=True, exist_ok=True)
            # Tester l'écriture
            test = p / ".write_test"
            test.touch()
            test.unlink()
            return p
        except (PermissionError, OSError):
            continue
    # Dernier recours : /tmp
    fallback = Path("/tmp/onepilot_models")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

MODEL_DIR = _get_model_dir()
logger.info(f"[ML] Répertoire modèles : {MODEL_DIR}")

# ── Features utilisées ──
FEATURE_COLS = [
    'name_sim', 'norm_sim', 'entity_in_field', 'type_compat',
    'fk_pattern_a', 'fk_pattern_b', 'pk_fk_pair', 'common_parts',
    'name_contains_entity_a', 'name_contains_entity_b',
    'len_diff', 'prefix_match', 'suffix_match',
    # ── 3 profiling features (notebook v6) — neutres si entity_profiles absent ──
    'value_overlap', 'cardinality_ratio', 'null_rate_compat',
    # ── co-occurrence feature (requêtes utilisateurs) ──
    'co_occurrence',
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
                       eb, fb, dtb, is_pk_b, is_fk_b,
                       profile_index: dict = None) -> dict:
    na, nb = _normalize(fa), _normalize(fb)
    ea_n, eb_n = _normalize(ea), _normalize(eb)
    parts_a = _camel_parts(fa)
    parts_b = _camel_parts(fb)
    common = len(parts_a & parts_b) / max(len(parts_a | parts_b), 1)

    # ── 3 profiling features (notebook v6) ──
    pi = profile_index or {}
    ka = (ea.upper(), fa.upper())
    kb = (eb.upper(), fb.upper())
    pa = pi.get(ka, {})
    pb = pi.get(kb, {})
    top_a = pa.get('top_values', set())
    top_b = pb.get('top_values', set())
    if top_a and top_b:
        inter = len(top_a & top_b)
        union = len(top_a | top_b)
        value_overlap = inter / union if union > 0 else 0.0
    else:
        value_overlap = 0.0
    uc_a = pa.get('unique_count', 0)
    uc_b = pb.get('unique_count', 0)
    cardinality_ratio = min(uc_a, uc_b) / max(uc_a, uc_b) if uc_a > 0 and uc_b > 0 else 0.5
    nr_a = pa.get('null_rate', 0.5)
    nr_b = pb.get('null_rate', 0.5)
    null_rate_compat = 1.0 - abs(nr_a - nr_b)

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
        # ── profiling ──
        'value_overlap':          value_overlap,
        'cardinality_ratio':      cardinality_ratio,
        'null_rate_compat':       null_rate_compat,
        # co_occurrence injecté ultérieurement (0.0 par défaut)
        'co_occurrence':          0.0,
    }


# ======================================================================
# CO-OCCURRENCE — Calcul depuis nlu_query_log
# ======================================================================

async def compute_co_occurrence(source_id: UUID, pool) -> dict:
    """
    Calcule un index de co-occurrence entre tables depuis nlu_query_log.
    Retourne un dict {(table_a, table_b): score} normalisé [0, 1].

    Exemple:
        {('Orders', 'Customers'): 0.42, ('Orders', 'Products'): 0.31}

    Plus deux tables apparaissent ensemble dans les requêtes utilisateurs,
    plus leur score de co-occurrence est élevé → feature ML.
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT tables_detected
                FROM nlu_query_log
                WHERE source_id = $1
                  AND tables_detected IS NOT NULL
                  AND array_length(tables_detected, 1) >= 2
                ORDER BY created_at DESC
                LIMIT 5000
            """, source_id)
    except Exception as e:
        logger.warning(f"[CoOccurrence] Query error: {e}")
        return {}

    if not rows:
        return {}

    from collections import Counter
    pair_counts: Counter = Counter()
    total_queries = len(rows)

    for row in rows:
        tables = row["tables_detected"] or []
        tables_upper = [t.upper() for t in tables]
        # Génère toutes les paires (non-ordonnées)
        for i in range(len(tables_upper)):
            for j in range(i + 1, len(tables_upper)):
                pair = tuple(sorted([tables_upper[i], tables_upper[j]]))
                pair_counts[pair] += 1

    # Normalise par le nombre total de requêtes
    co_index = {}
    for pair, count in pair_counts.items():
        score = min(count / max(total_queries, 1), 1.0)
        co_index[pair] = round(score, 4)

    logger.info(f"[CoOccurrence] {len(co_index)} paires calculées depuis {total_queries} requêtes")
    return co_index


def inject_co_occurrence(df_dataset, co_index: dict) -> "pd.DataFrame":
    """
    Injecte la feature co_occurrence dans le dataset de features ML.
    Remplace la valeur par défaut 0.0 par le score réel pour chaque paire.
    """
    if not co_index or df_dataset.empty:
        return df_dataset

    def _get_score(row):
        ea = str(row.get('entity_a', '')).upper()
        eb = str(row.get('entity_b', '')).upper()
        pair = tuple(sorted([ea, eb]))
        return co_index.get(pair, 0.0)

    if 'entity_a' in df_dataset.columns and 'entity_b' in df_dataset.columns:
        df_dataset = df_dataset.copy()
        df_dataset['co_occurrence'] = df_dataset.apply(_get_score, axis=1)
        n_nonzero = (df_dataset['co_occurrence'] > 0).sum()
        logger.info(f"[CoOccurrence] {n_nonzero}/{len(df_dataset)} paires avec score > 0")

    return df_dataset


# ======================================================================
# CHARGEMENT DES DONNÉES
# ======================================================================

async def _load_data(source_id: UUID, pool) -> tuple:
    """Charge les champs, relations et profils depuis la DB."""
    import json as _json
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
                       WHEN detection_method IN ('explicit_fk','view_join')        THEN 1.0
                       WHEN detection_method LIKE '%name_pascal%'                  THEN 0.8
                       WHEN detection_method LIKE '%name_m2m%'                     THEN 0.7
                       WHEN detection_method LIKE '%fuzzy%'                        THEN 0.5
                       WHEN detection_method LIKE 'heuristic_%'                    THEN 0.6
                       WHEN is_confirmed = TRUE                                    THEN 0.9
                       ELSE 0.6
                   END as sample_weight
            FROM entity_relations
            WHERE source_id = $1
              AND detection_method NOT LIKE '%ml_predicted%'
              AND (is_confirmed IS NULL OR is_confirmed = TRUE)
            ORDER BY sample_weight DESC
        """, source_id)

        # Charger entity_profiles pour les 3 features profiling (notebook v6)
        try:
            prof_rows = await conn.fetch("""
                SELECT entity_name, profile_data
                FROM entity_profiles
                WHERE source_id = $1
                  AND profile_data->>'error' IS NULL
            """, source_id)
        except Exception:
            prof_rows = []

    df_fields = pd.DataFrame([dict(r) for r in fields_rows])
    df_relations = pd.DataFrame([dict(r) for r in relations_rows])

    # Construire profile_index : {(TABLE, CHAMP) -> {unique_count, null_rate, top_values}}
    profile_index = {}
    for row in prof_rows:
        try:
            table = row['entity_name'].upper()
            data = row['profile_data']
            if isinstance(data, str):
                data = _json.loads(data)
            for col in data.get('columns', []):
                key = (table, col['name'].upper())
                profile_index[key] = {
                    'top_values':   set(str(v['value']) for v in col.get('top_values', [])),
                    'unique_count': col.get('unique_count', 0),
                    'null_rate':    col.get('null_rate', 0.5),
                }
        except Exception:
            pass
    logger.info(f"[ML] {len(profile_index)} profils chargés")

    return df_fields, df_relations, profile_index


# ======================================================================
# CONSTRUCTION DU DATASET
# ======================================================================

def _build_dataset(df_fields: pd.DataFrame, df_relations: pd.DataFrame, profile_index: dict = None) -> pd.DataFrame:
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
            tgt_field['is_primary_key'], tgt_field['is_foreign_key'],
            profile_index=profile_index
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
            fb['is_primary_key'], fb['is_foreign_key'],
            profile_index=profile_index
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
    df_fields, df_relations, profile_index = await _load_data(source_id, pool)
    if df_relations.empty:
        raise ValueError("Pas assez de relations pour entraîner le modèle")

    # Co-occurrence depuis nlu_query_log
    co_index = await compute_co_occurrence(source_id, pool)
    logger.info(f"[ML Train] Co-occurrence: {len(co_index)} paires")

    # Dataset
    df_dataset, entity_map, pk_map, positive_set = _build_dataset(df_fields, df_relations, profile_index)
    df_dataset = inject_co_occurrence(df_dataset, co_index)

    # Vérifier que toutes les features sont présentes
    missing = [c for c in FEATURE_COLS if c not in df_dataset.columns]
    if missing:
        logger.warning(f"[ML Train] Features manquantes: {missing} — remplissage à 0")
        for c in missing:
            df_dataset[c] = 0.0

    X = df_dataset[FEATURE_COLS].values
    y = df_dataset['label'].values
    w = df_dataset['sample_weight'].values

    if len(df_dataset) < 10:
        raise ValueError(f"Dataset trop petit ({len(df_dataset)} exemples). Lancez d'abord ↻ Relancer pour détecter les relations.")

    # stratify=y nécessite au moins 2 classes avec assez d'exemples
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min() if len(counts) > 1 else 0
    use_stratify = len(unique) >= 2 and min_count >= 2
    try:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, w, test_size=0.2, random_state=42,
            stratify=y if use_stratify else None
        )
    except ValueError:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, w, test_size=0.2, random_state=42
        )

    # Entraînement des modèles — hyperparamètres alignés avec notebook v7
    models = {}

    # ── RandomForest (hyperparams notebook v7) ──
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=1,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train, sample_weight=w_train)
    models['RandomForest'] = rf

    # ── GradientBoosting (hyperparams notebook v7) ──
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=2, random_state=42
    )
    gb.fit(X_train, y_train, sample_weight=w_train)
    models['GradientBoosting'] = gb

    if HAS_XGB:
        spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.5, reg_alpha=0.1,
            scale_pos_weight=spw,
            eval_metric='logloss', random_state=42, n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model

    # Sélection meilleur modèle
    # Tie-break : RandomForest préféré si les scores sont très proches (< 0.005)
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

    # Tie-break : si RandomForest est dans les 1.5% du meilleur → préférer RandomForest
    # Raison : cohérence avec notebook v7, modèle plus stable et plus rapide en prod
    rf_f1   = results.get('RandomForest', {}).get('f1', 0)
    best_f1 = best['f1']
    if best_name != 'RandomForest' and rf_f1 > 0 and (best_f1 - rf_f1) < 0.015:
        best_name = 'RandomForest'
        best = results['RandomForest']
        logger.info(f"[ML Train] Tie-break → RandomForest sélectionné (Δ F1={best_f1-rf_f1:.4f} < 0.015)")

    # Seuil optimal via PR curve — minimum 0.80 pour éviter les faux positifs
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, best['y_proba'])
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    threshold = float(thresholds[best_idx])
    # Enforce minimum threshold 0.80 pour réduire les faux positifs
    # (évite le sur-fitting sur petits datasets)
    threshold = max(threshold, 0.80)
    logger.info(f"[ML Train] Seuil final: {threshold:.3f} (min=0.80)")

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

    # Synchroniser best_model_randomforest.pkl dans MODEL_DIR (writable)
    # → predict_ml_relations chargera ce fichier en priorité
    if best_name == 'RandomForest':
        rf_path = MODEL_DIR / "best_model_randomforest.pkl"
        try:
            with open(rf_path, 'wb') as f:
                pickle.dump({
                    'model':        best['model'],
                    'model_name':   best_name,
                    'feature_cols': FEATURE_COLS,
                    'source_id':    str(source_id),
                    'threshold':    threshold,
                    'trained_at':   datetime.utcnow().isoformat(),
                    'metrics': {
                        'f1':        best['f1'],
                        'roc_auc':   best['roc_auc'],
                        'precision': best['prec'],
                        'recall':    best['rec'],
                    }
                }, f)
            logger.info(f"[ML Train] ✅ best_model_randomforest.pkl → {rf_path}")
        except Exception as e:
            logger.warning(f"[ML Train] Impossible de sauvegarder best_model_randomforest.pkl : {e}")

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
    # Chercher le modèle dans plusieurs endroits (notebook + API)
    # ── 1. Modèle spécifique à cette source (entraîné sur ses données) ──
    specific_candidates = [
        _HERE / f"best_model_xgboost_{source_id}.pkl",
        _HERE / f"best_model_randomforest_{source_id}.pkl",
        MODEL_DIR / f"{source_id}.pkl",
    ]
    model_path = next((p for p in specific_candidates if p.exists()), None)

    # ── 2. Fallback : modèle générique si aucun spécifique disponible ──
    if model_path is None:
        logger.warning(
            f"[ML] Aucun modèle spécifique pour {source_id} — "
            "utilisation du modèle générique. "
            "Lancez le notebook ml_relation_detector_v8.ipynb pour entraîner un modèle dédié."
        )
        generic_candidates = [
            _HERE / "best_model_xgboost.pkl",
            _HERE / "best_model_randomforest.pkl",
            _HERE / "best_model_xgboost_v2.pkl",
            MODEL_DIR / "best_model_xgboost.pkl",
            MODEL_DIR / "best_model_xgboost_v2.pkl",
            MODEL_DIR / "best_model_randomforest.pkl",
        ]
        model_path = next((p for p in generic_candidates if p.exists()), None)

    if model_path is None:
        raise FileNotFoundError(
            f"Aucun modèle ML disponible pour {source_id}. "
            "Lancez le notebook ml_relation_detector_v8.ipynb."
        )
    logger.info(f"[ML Predict] Chargement modèle depuis {model_path} "
                f"({'spécifique source' if str(source_id) in str(model_path) else 'générique'})")

    # Charger le modèle
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    model = saved['model']
    threshold = saved['threshold']

    logger.info(f"[ML Predict] Modèle {saved['model_name']} chargé (seuil={threshold:.3f})")

    # Charger les données
    df_fields, df_relations, profile_index = await _load_data(source_id, pool)
    _, entity_map, pk_map, positive_set = _build_dataset(df_fields, df_relations, profile_index)

    # Générer les paires candidates
    # Priorité : FK explicites > patterns forts > patterns faibles
    # Cap pour éviter explosion combinatoire (ex: 4051 × 2070 = 8.4M)
    fk_explicit = [r for _, r in df_fields.iterrows() if r['is_foreign_key']]
    fk_pattern  = [r for _, r in df_fields.iterrows()
                   if not r['is_foreign_key'] and _fk_pat(r['field_name']) > 0]
    pk_cands    = list(df_fields[df_fields['is_primary_key'] == True].iterrows())

    # Cap : max 500 FK candidates (explicites en priorité)
    FK_CAP = 500
    fk_cands = fk_explicit[:FK_CAP]
    remaining = FK_CAP - len(fk_cands)
    if remaining > 0:
        fk_cands += fk_pattern[:remaining]

    # Cap PK candidates aussi
    pk_cands = pk_cands[:300]

    logger.info(f"[ML Predict] {len(fk_cands)} FK × {len(pk_cands)} PK candidates "
                f"(plafonné depuis {len(fk_explicit)+len(fk_pattern)} FK × {len(df_fields[df_fields['is_primary_key']==True])} PK)")

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
                fb['is_primary_key'], fb['is_foreign_key'],
                profile_index=profile_index
            )
            feat.update({
                'source_entity': fa['entity_name'], 'source_field': fa['field_name'],
                'target_entity': fb['entity_name'], 'target_field': fb['field_name']
            })
            unknown_pairs.append(feat)

    if not unknown_pairs:
        return {'inserted': 0, 'message': 'Aucune paire candidate trouvée'}

    df_unknown = pd.DataFrame(unknown_pairs)
    # v8 : utiliser les features du modèle sauvegardé (16 ou 21 selon la version)
    model_feature_cols = saved.get('active_cols', saved.get('feature_cols', FEATURE_COLS))
    # Ajouter les colonnes manquantes avec 0.0 (rétrocompatibilité v7 → v8)
    for col in model_feature_cols:
        if col not in df_unknown.columns:
            df_unknown[col] = 0.0
    X_unknown = df_unknown[model_feature_cols].values
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
                    ON CONFLICT (source_id, source_entity, source_field, target_entity)
                    DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        detection_method = EXCLUDED.detection_method
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
    # ── 1. Modèle spécifique à cette source ──
    specific_candidates = [
        _HERE / f"best_model_xgboost_{source_id}.pkl",
        _HERE / f"best_model_randomforest_{source_id}.pkl",
        MODEL_DIR / f"{source_id}.pkl",
    ]
    model_path = next((p for p in specific_candidates if p.exists()), None)

    # ── 2. Fallback générique ──
    if model_path is None:
        generic_candidates = [
            _HERE / "best_model_xgboost.pkl",
            _HERE / "best_model_randomforest.pkl",
            _HERE / "best_model_xgboost_v2.pkl",
            MODEL_DIR / "best_model_xgboost.pkl",
            MODEL_DIR / "best_model_xgboost_v2.pkl",
        ]
        model_path = next((p for p in generic_candidates if p.exists()), None)

    if not model_path:
        return {
            'available':  False,
            'message':    'Modèle non trouvé. Copiez best_model_xgboost_v2.pkl dans /api/ ou lancez /ml/train',
        }

    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    active_cols = saved.get('active_cols', saved.get('feature_cols', FEATURE_COLS))
    return {
        'available':       True,
        'model_name':      saved.get('model_name', 'unknown'),
        'trained_at':      saved.get('trained_at'),
        'threshold':       saved.get('threshold'),
        'metrics':         saved.get('metrics', {}),
        'feature_cols':    saved.get('feature_cols', []),
        'active_cols':     active_cols,
        'feature_count':   len(active_cols),
        'use_v2':          saved.get('use_v2', False),
        'embed_available': saved.get('embed_available', False),
        'stem_available':  saved.get('stem_available', False),
        'model_file':      str(model_path),
        'source_specific': str(source_id) in str(model_path),
    }

# ══════════════════════════════════════════════════════════════════════
# PATCH P10 — Embeddings colonnes (sentence-transformers / fallback TF-IDF)
# ══════════════════════════════════════════════════════════════════════

class EmbeddingCache:
    """
    Cache singleton pour les embeddings de noms de colonnes/entités.
    Utilise sentence-transformers all-MiniLM-L6-v2 si disponible,
    sinon fallback sur vecteur caractères (Jaccard-like).
    """
    _instance = None
    _model    = None
    _cache: dict = {}
    _model_name  = "all-MiniLM-L6-v2"

    @classmethod
    def get_instance(cls) -> "EmbeddingCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"[Embeddings] {self._model_name} chargé")
            return True
        except ImportError:
            logger.warning(
                "[Embeddings] sentence-transformers non installé — fallback char-vector. "
                "pip install sentence-transformers pour activer les embeddings sémantiques."
            )
            return False

    def encode(self, text: str):
        import numpy as np
        if text in self._cache:
            return self._cache[text]
        if self._load_model():
            try:
                vec = self._model.encode(text, show_progress_bar=False)
                self._cache[text] = vec
                return vec
            except Exception as e:
                logger.debug(f"[Embeddings] encode error: {e}")
        # Fallback : vecteur binaire sur l'alphabet
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789_"
        chars = set(text.lower())
        vec = np.array([float(c in chars) for c in alphabet], dtype=np.float32)
        norm = np.linalg.norm(vec)
        result = vec / norm if norm > 0 else vec
        self._cache[text] = result
        return result

    def similarity(self, a: str, b: str) -> float:
        import numpy as np
        va, vb = self.encode(a), self.encode(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def precompute_batch(self, texts: list) -> None:
        uncached = [t for t in texts if t not in self._cache]
        if not uncached or not self._load_model():
            return
        try:
            import numpy as np
            vecs = self._model.encode(uncached, batch_size=64, show_progress_bar=False)
            for text, vec in zip(uncached, vecs):
                self._cache[text] = vec
        except Exception as e:
            logger.warning(f"[Embeddings] batch encode error: {e}")


def _compute_embedding_features(field_a: str, entity_a: str,
                                  field_b: str, entity_b: str) -> dict:
    """PATCH P10 — Similarité cosinus sur les embeddings des noms."""
    emb = EmbeddingCache.get_instance()
    def _h(s): return s.replace("_", " ").replace("-", " ").lower()
    return {
        "embed_sim":        round(emb.similarity(_h(field_a),  _h(field_b)),  4),
        "embed_entity_sim": round(emb.similarity(_h(entity_a), _h(entity_b)), 4),
    }


def enrich_features_with_embeddings(df: "pd.DataFrame") -> "pd.DataFrame":
    """Ajoute embed_sim et embed_entity_sim au DataFrame de features."""
    emb = EmbeddingCache.get_instance()
    all_texts = set()
    for col in ["source_field", "target_field", "source_entity", "target_entity"]:
        if col in df.columns:
            all_texts.update(df[col].str.replace("_", " ").str.lower().tolist())
    emb.precompute_batch(list(all_texts))

    def _row(row):
        f = _compute_embedding_features(
            row.get("source_field",  ""), row.get("source_entity", ""),
            row.get("target_field",  ""), row.get("target_entity", ""),
        )
        return pd.Series(f)

    embed_cols = df.apply(_row, axis=1)
    for col in ["embed_sim", "embed_entity_sim"]:
        df[col] = embed_cols[col] if col in embed_cols.columns else 0.0
    return df


# ══════════════════════════════════════════════════════════════════════
# PATCH P11 — Distance topologique dans le graphe de schéma
# ══════════════════════════════════════════════════════════════════════

def compute_topological_distances(relations_df: "pd.DataFrame") -> dict:
    """
    PATCH P11 — Calcule les distances BFS entre toutes les paires d'entités.
    score = 1/(1+distance) : 1.0 = voisins directs, 0.0 = très éloignés.
    """
    from collections import deque
    graph: dict = {}
    if relations_df.empty:
        return {}
    for _, row in relations_df.iterrows():
        src = row.get("source_entity", "")
        tgt = row.get("target_entity", "")
        if src and tgt:
            graph.setdefault(src, set()).add(tgt)
            graph.setdefault(tgt, set()).add(src)
    distances: dict = {}
    for start in list(graph.keys()):
        visited = {start: 0}
        q = deque([start])
        while q:
            node = q.popleft()
            for nb in graph.get(node, set()):
                if nb not in visited:
                    visited[nb] = visited[node] + 1
                    q.append(nb)
        for end, dist in visited.items():
            if end != start:
                score = round(1.0 / (1.0 + dist), 4)
                distances[(start, end)] = score
                distances[(end, start)] = score
    return distances


def add_topological_feature(df: "pd.DataFrame", topo: dict) -> "pd.DataFrame":
    """PATCH P11 — Ajoute la colonne topo_distance au DataFrame."""
    df["topo_distance"] = df.apply(
        lambda r: topo.get((r.get("source_entity",""), r.get("target_entity","")), 0.0),
        axis=1
    )
    return df


# ══════════════════════════════════════════════════════════════════════
# PATCH P12 — Re-training incrémental basé sur feedback expert
# ══════════════════════════════════════════════════════════════════════

async def retrain_with_feedback(source_id: UUID, pool) -> dict:
    """
    PATCH P12 — Re-entraîne le modèle ML en incorporant le feedback expert.
    Feedbacks confirmés → poids ×2 | Feedbacks rejetés → ajoutés comme négatifs.
    """
    async with pool.acquire() as conn:
        feedback_rows = await conn.fetch("""
            SELECT source_entity, source_field, target_entity, target_field, feedback
            FROM relation_feedback
            WHERE source_id=$1 AND feedback IN ('confirmed','rejected')
        """, source_id)

    if not feedback_rows:
        return {"error": "Aucun feedback disponible", "retrained": False}

    confirmed = {(r["source_entity"], r["source_field"],
                  r["target_entity"], r["target_field"])
                 for r in feedback_rows if r["feedback"] == "confirmed"}
    rejected  = {(r["source_entity"], r["source_field"],
                  r["target_entity"], r["target_field"])
                 for r in feedback_rows if r["feedback"] == "rejected"}

    logger.info(f"[ML Retrain] {len(confirmed)} confirmés, {len(rejected)} rejetés")

    df_fields, df_relations, profile_index = await _load_data(source_id, pool)
    result_tuple = _build_dataset(df_fields, df_relations, profile_index)
    df_dataset   = result_tuple[0] if isinstance(result_tuple, tuple) else result_tuple

    if df_dataset.empty:
        return {"error": "Dataset vide", "retrained": False}

    # Ajuster les poids
    weights = []
    for _, row in df_dataset.iterrows():
        key = (row.get("source_entity",""), row.get("source_field",""),
               row.get("target_entity",""), row.get("target_field",""))
        if key in confirmed and row.get("label", 0) == 1:
            weights.append(2.0)
        elif key in rejected and row.get("label", 0) == 1:
            weights.append(0.1)
        else:
            weights.append(1.0)

    feature_cols = [c for c in FEATURE_COLS if c in df_dataset.columns]
    X = df_dataset[feature_cols].fillna(0).values
    y = df_dataset["label"].values
    w = np.array(weights[:len(y)])

    if len(set(y)) < 2:
        return {"error": "Dataset déséquilibré", "retrained": False}

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score as _f1

    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X, y, w, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr, sample_weight=w_tr)

    proba_val = model.predict_proba(X_val)[:, 1]
    auc       = roc_auc_score(y_val, proba_val)
    best_f1, best_thr = 0.0, 0.5
    for t in np.arange(0.3, 0.8, 0.05):
        f = _f1(y_val, (proba_val >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, float(t)

    model_data = {
        "model":        model,
        "model_name":   "RandomForest_feedback",
        "threshold":    best_thr,
        "feature_cols": feature_cols,
        "metrics":      {"roc_auc": round(auc, 4), "f1": round(best_f1, 4)},
        "trained_at":   datetime.utcnow().isoformat(),
        "feedback_count": len(confirmed) + len(rejected),
        "retrained_from_feedback": True,
    }
    model_path = MODEL_DIR / f"{source_id}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(model_data, fh)

    logger.info(f"[ML Retrain] AUC={auc:.4f} F1={best_f1:.4f} seuil={best_thr:.3f}")
    return {
        "retrained":      True,
        "model_path":     str(model_path),
        "roc_auc":        round(auc, 4),
        "f1":             round(best_f1, 4),
        "threshold":      round(best_thr, 3),
        "feedback_count": len(confirmed) + len(rejected),
        "trained_at":     datetime.utcnow().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════
# PATCH P13 — Transfer learning depuis schémas publics annotés
# ══════════════════════════════════════════════════════════════════════

PUBLIC_SCHEMAS = {
    "northwind": [
        ("Orders","CustomerID","Customers","CustomerID"),
        ("Orders","EmployeeID","Employees","EmployeeID"),
        ("Orders","ShipVia","Shippers","ShipperID"),
        ("Order Details","OrderID","Orders","OrderID"),
        ("Order Details","ProductID","Products","ProductID"),
        ("Products","CategoryID","Categories","CategoryID"),
        ("Products","SupplierID","Suppliers","SupplierID"),
        ("Employees","ReportsTo","Employees","EmployeeID"),
        ("EmployeeTerritories","EmployeeID","Employees","EmployeeID"),
        ("EmployeeTerritories","TerritoryID","Territories","TerritoryID"),
        ("Territories","RegionID","Region","RegionID"),
    ],
    "adventureworks": [
        ("SalesOrderHeader","CustomerID","Customer","CustomerID"),
        ("SalesOrderHeader","SalesPersonID","SalesPerson","BusinessEntityID"),
        ("SalesOrderDetail","SalesOrderID","SalesOrderHeader","SalesOrderID"),
        ("SalesOrderDetail","ProductID","Product","ProductID"),
        ("Product","ProductSubcategoryID","ProductSubcategory","ProductSubcategoryID"),
        ("ProductSubcategory","ProductCategoryID","ProductCategory","ProductCategoryID"),
        ("Employee","BusinessEntityID","Person","BusinessEntityID"),
        ("PurchaseOrderHeader","VendorID","Vendor","BusinessEntityID"),
        ("PurchaseOrderDetail","ProductID","Product","ProductID"),
        ("Address","StateProvinceID","StateProvince","StateProvinceID"),
    ],
}


def build_transfer_dataset(feature_fn) -> "pd.DataFrame":
    """
    PATCH P13 — Dataset depuis Northwind + AdventureWorks pour transfer learning.
    feature_fn : _compute_features de ml_detector.py
    """
    rows = []
    for schema_name, relations in PUBLIC_SCHEMAS.items():
        for (src_e, src_f, tgt_e, tgt_f) in relations:
            try:
                feat = feature_fn(
                    src_e, src_f, "int", False, src_f.lower().endswith("id"),
                    tgt_e, tgt_f, "int", True,  False, {},
                )
                feat.update({
                    "source_entity": src_e, "source_field": src_f,
                    "target_entity": tgt_e, "target_field": tgt_f,
                    "label":         1,
                    "sample_weight": 0.5,
                    "schema_origin": schema_name,
                })
                rows.append(feat)
            except Exception as e:
                logger.debug(f"[transfer] {src_e}.{src_f}: {e}")
    df = pd.DataFrame(rows)
    logger.info(f"[transfer] {len(df)} exemples positifs depuis schémas publics")
    return df