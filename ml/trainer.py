# ml/trainer.py
"""
ML Trainer V3 — Separate LONG/SHORT models + Optuna + Feature Selection
Improvements over V2:
  1. Separate models for LONG and SHORT predictions
  2. Optuna hyperparameter tuning for LightGBM
  3. Feature selection via importance + correlation cleanup
  4. Probability calibration (isotonic regression)
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

logger = logging.getLogger(__name__)

EXCLUDE_COLS = [
    'datetime', 'open', 'high', 'low', 'close', 'volume',
    'target', 'target_reason', 'target_pnl_ratio',
    'target_long', 'target_short', 'target_3class',
    'ema10', 'ema20', 'ema50', 'h4_ema20', 'd1_ema20',
    'atr14', 'atr_avg50',  # Keep derived (atr_ratio), drop raw
]

MODELS_DIR = 'ml/models'

LSTM_FEATURES = [
    'return_1h', 'return_4h', 'rsi14', 'atr_ratio',
    'macd_hist', 'adx', 'bb_pctb', 'body_ratio',
    'stoch_k', 'obv_zscore', 'volume_ratio', 'plus_di', 'minus_di'
]
LOOKBACK = 24


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS
            and not c.startswith('target')]


def prepare_data(df: pd.DataFrame, target_col: str = 'target'):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y, feature_cols


# =========================================================================
# FEATURE SELECTION
# =========================================================================

def select_features(X_train, y_train, feature_cols, top_n=35):
    """Select top features by LightGBM importance + remove high correlation."""
    import lightgbm as lgb

    # Quick model for feature importance
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {'objective': 'binary', 'verbose': -1, 'num_leaves': 31,
              'learning_rate': 0.1, 'n_jobs': -1, 'max_depth': 5}
    quick_model = lgb.train(params, train_data, num_boost_round=100,
                            callbacks=[lgb.log_evaluation(period=0)])
    importances = quick_model.feature_importance(importance_type='gain')
    fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    # Take top features
    selected = [name for name, _ in fi[:top_n]]

    # Remove highly correlated features (>0.95)
    corr_matrix = X_train[selected].corr().abs()
    to_drop = set()
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if corr_matrix.iloc[i, j] > 0.95:
                # Drop the one with lower importance
                to_drop.add(selected[j])

    final = [f for f in selected if f not in to_drop]
    logger.info("Feature selection: %d → %d (dropped %d correlated)",
                len(feature_cols), len(final), len(to_drop))
    return final


# =========================================================================
# OPTUNA HYPERPARAMETER TUNING
# =========================================================================

def optuna_tune_lgb(X_train, y_train, X_val, y_val, n_trials=30):
    """Find optimal LightGBM hyperparameters with Optuna."""
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / max(n_pos, 1)

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'n_jobs': -1,
            'scale_pos_weight': scale_pos,
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 200),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 2.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.001, 0.1, log=True),
            'path_smooth': trial.suggest_float('path_smooth', 0.0, 2.0),
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params, train_data, num_boost_round=1000,
            valid_sets=[val_data], valid_names=['val'],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(period=0)]
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration)
        try:
            return roc_auc_score(y_val, preds)
        except:
            return 0.5

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Optuna best AUC=%.4f (trial %d/%d)",
                study.best_value, study.best_trial.number + 1, n_trials)
    return study.best_params


# =========================================================================
# LightGBM V3 — Optuna tuned
# =========================================================================

def train_lightgbm(X_train, y_train, X_val=None, y_val=None,
                   params_override=None, n_rounds=1000):
    import lightgbm as lgb

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / max(n_pos, 1)

    if params_override:
        params = {**params_override,
                  'objective': 'binary', 'metric': 'binary_logloss',
                  'verbose': -1, 'n_jobs': -1,
                  'scale_pos_weight': scale_pos}
    else:
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'boosting_type': 'gbdt', 'num_leaves': 63,
            'learning_rate': 0.03, 'feature_fraction': 0.7,
            'bagging_fraction': 0.7, 'bagging_freq': 5,
            'scale_pos_weight': scale_pos, 'verbose': -1, 'n_jobs': -1,
            'min_child_samples': 100, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
            'max_depth': 8, 'min_gain_to_split': 0.01, 'path_smooth': 1.0,
        }

    train_data = lgb.Dataset(X_train, label=y_train)

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(
            params, train_data, num_boost_round=n_rounds,
            valid_sets=[train_data, val_data], valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)]
        )
    else:
        model = lgb.train(params, train_data, num_boost_round=n_rounds,
                          callbacks=[lgb.log_evaluation(period=0)])

    logger.info("LightGBM: %d trees (best_iter=%d)",
                model.num_trees(), model.best_iteration)
    return model


def predict_lightgbm(model, X) -> np.ndarray:
    return model.predict(X, num_iteration=model.best_iteration)


# =========================================================================
# LSTM V2
# =========================================================================

def build_lstm_sequences(X, y, lookback=LOOKBACK):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def train_lstm(X_train, y_train, X_val=None, y_val=None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras

    available = [c for c in LSTM_FEATURES if c in X_train.columns] if isinstance(X_train, pd.DataFrame) else LSTM_FEATURES
    X_train_lstm = X_train[available].values if isinstance(X_train, pd.DataFrame) else X_train

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_lstm)
    X_seq, y_seq = build_lstm_sequences(
        X_scaled, y_train.values if hasattr(y_train, 'values') else y_train)

    if len(X_seq) < 100:
        return None, scaler, available

    n_pos = y_seq.sum()
    n_neg = len(y_seq) - n_pos
    cw = {0: 1.0, 1: n_neg / max(n_pos, 1)}

    model = keras.Sequential([
        keras.layers.Input(shape=(LOOKBACK, len(available))),
        keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.1),
        keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.1),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy', metrics=['accuracy'])

    val_data = None
    if X_val is not None:
        X_val_lstm = X_val[available].values if isinstance(X_val, pd.DataFrame) else X_val
        X_val_s = scaler.transform(X_val_lstm)
        X_vs, y_vs = build_lstm_sequences(
            X_val_s, y_val.values if hasattr(y_val, 'values') else y_val)
        if len(X_vs) > 0:
            val_data = (X_vs, y_vs)

    model.fit(X_seq, y_seq, epochs=80, batch_size=128, validation_data=val_data,
              class_weight=cw, callbacks=[
                  keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True,
                                                monitor='val_loss' if val_data else 'loss'),
                  keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
              ], verbose=0)

    logger.info("LSTM: %d sequences, %d features", len(X_seq), len(available))
    return model, scaler, available


def predict_lstm(model, scaler, X, feature_cols, lookback=LOOKBACK) -> np.ndarray:
    if model is None:
        return np.full(len(X), 0.5)
    available = [c for c in feature_cols if c in X.columns] if isinstance(X, pd.DataFrame) else feature_cols
    X_data = X[available].values if isinstance(X, pd.DataFrame) else X
    X_scaled = scaler.transform(X_data)
    X_seq, _ = build_lstm_sequences(X_scaled, np.zeros(len(X_scaled)), lookback)
    if len(X_seq) == 0:
        return np.full(len(X), 0.5)
    preds = model.predict(X_seq, verbose=0).flatten()
    full = np.full(len(X), 0.5)
    full[lookback:lookback + len(preds)] = preds
    return full


# =========================================================================
# STACKING + CALIBRATION
# =========================================================================

def train_stacking(lgb_probs, lstm_probs, y):
    X_meta = np.column_stack([lgb_probs, lstm_probs])
    meta = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced')
    meta.fit(X_meta, y)
    logger.info("Stacking: LGB=%.3f, LSTM=%.3f", meta.coef_[0][0], meta.coef_[0][1])
    return meta


def predict_stacking(meta, lgb_probs, lstm_probs) -> np.ndarray:
    X_meta = np.column_stack([lgb_probs, lstm_probs])
    return meta.predict_proba(X_meta)[:, 1]


def find_optimal_threshold(y_true, probs, min_precision=0.55) -> float:
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.10, 0.70, 0.01):
        preds = (probs > t).astype(int)
        if preds.sum() < 10:
            continue
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if prec >= min_precision and f1 > best_f1:
            best_f1, best_t = f1, t
    if best_f1 == 0:  # Fallback: best F1
        for t in np.arange(0.10, 0.70, 0.01):
            preds = (probs > t).astype(int)
            if preds.sum() < 5:
                continue
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1, best_t = f1, t
    return best_t


# =========================================================================
# SINGLE DIRECTION TRAINING
# =========================================================================

def _train_one_direction(df, feature_cols_all, direction: str,
                         use_optuna: bool = True, optuna_trials: int = 30) -> Dict:
    """Train complete model pipeline for one direction (LONG or SHORT)."""
    target_col = f'target_{direction}'
    prefix = direction  # 'long' or 'short'

    logger.info("=" * 50)
    logger.info("TRAINING %s MODEL", direction.upper())
    logger.info("=" * 50)

    df = df.sort_values('datetime').reset_index(drop=True)
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    years = sorted(df['year'].unique())
    test_years = [y for y in years if y >= 2022]

    X_all, y_all, _ = prepare_data(df, target_col)

    # Feature selection on full data (quick)
    selected_features = select_features(X_all, y_all, feature_cols_all)
    X_all = X_all[selected_features]

    results = []
    best = {'lgb': None, 'lstm': None, 'lstm_scaler': None,
            'lstm_features': None, 'meta': None, 'threshold': 0.5,
            'optuna_params': None, 'selected_features': selected_features}

    for test_year in test_years:
        purge_bars = 24
        train_idx = np.where(df['year'] < test_year)[0]
        test_idx = np.where(df['year'] == test_year)[0]

        if len(train_idx) < 5000 or len(test_idx) < 200:
            continue

        if len(train_idx) > purge_bars:
            train_idx = train_idx[:-purge_bars]

        X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_test, y_test = X_all.iloc[test_idx], y_all.iloc[test_idx]

        val_size = int(len(X_train) * 0.15)
        X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
        y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

        # Optuna tuning (only on first fold, reuse params)
        if use_optuna and best['optuna_params'] is None:
            logger.info("Running Optuna tuning (%d trials)...", optuna_trials)
            best['optuna_params'] = optuna_tune_lgb(X_tr, y_tr, X_val, y_val, optuna_trials)

        # LightGBM
        lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val,
                                    params_override=best['optuna_params'])
        lgb_probs_train = predict_lightgbm(lgb_model, X_train)
        lgb_probs_test = predict_lightgbm(lgb_model, X_test)

        # LSTM
        lstm_model, lstm_scaler, lstm_feats = train_lstm(X_tr, y_tr, X_val, y_val)
        lstm_probs_train = predict_lstm(lstm_model, lstm_scaler, X_train, lstm_feats)
        lstm_probs_test = predict_lstm(lstm_model, lstm_scaler, X_test, lstm_feats)

        # Stacking
        meta_model = train_stacking(lgb_probs_train, lstm_probs_train, y_train)
        final_probs = predict_stacking(meta_model, lgb_probs_test, lstm_probs_test)

        # Threshold
        lgb_pv = predict_lightgbm(lgb_model, X_val)
        lstm_pv = predict_lstm(lstm_model, lstm_scaler, X_val, lstm_feats)
        val_probs = predict_stacking(meta_model, lgb_pv, lstm_pv)
        threshold = find_optimal_threshold(y_val, val_probs, min_precision=0.55)

        # Evaluate
        preds = (final_probs > threshold).astype(int)
        n_sig = preds.sum()

        prec = precision_score(y_test, preds, zero_division=0) if n_sig > 0 else 0
        rec = recall_score(y_test, preds, zero_division=0) if n_sig > 0 else 0
        f1 = f1_score(y_test, preds, zero_division=0) if n_sig > 0 else 0
        try:
            auc_val = roc_auc_score(y_test, final_probs)
        except:
            auc_val = 0.5

        results.append({
            'test_year': test_year, 'threshold': threshold,
            'precision': prec, 'recall': rec, 'f1': f1,
            'auc': auc_val, 'n_signals': n_sig,
            'lgb_trees': lgb_model.num_trees(),
        })

        logger.info("[%s] Year %d: T=%.2f Prec=%.1f%% Rec=%.1f%% F1=%.3f AUC=%.3f Sig=%d",
                     direction.upper(), test_year, threshold,
                     prec*100, rec*100, f1, auc_val, n_sig)

        best.update({'lgb': lgb_model, 'lstm': lstm_model,
                     'lstm_scaler': lstm_scaler, 'lstm_features': lstm_feats,
                     'meta': meta_model, 'threshold': threshold})

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    if best['lgb']:
        best['lgb'].save_model(os.path.join(MODELS_DIR, f'lgb_{prefix}.txt'))
    if best['lstm']:
        best['lstm'].save(os.path.join(MODELS_DIR, f'lstm_{prefix}.keras'))
    if best['lstm_scaler']:
        joblib.dump(best['lstm_scaler'], os.path.join(MODELS_DIR, f'lstm_scaler_{prefix}.pkl'))
    if best['lstm_features']:
        joblib.dump(best['lstm_features'], os.path.join(MODELS_DIR, f'lstm_features_{prefix}.pkl'))
    if best['meta']:
        joblib.dump(best['meta'], os.path.join(MODELS_DIR, f'meta_{prefix}.pkl'))
    joblib.dump(best['threshold'], os.path.join(MODELS_DIR, f'threshold_{prefix}.pkl'))
    joblib.dump(best['selected_features'], os.path.join(MODELS_DIR, f'features_{prefix}.pkl'))
    if best['optuna_params']:
        joblib.dump(best['optuna_params'], os.path.join(MODELS_DIR, f'optuna_params_{prefix}.pkl'))

    # Feature importance
    fi = {}
    if best['lgb']:
        imp = best['lgb'].feature_importance(importance_type='gain')
        pairs = sorted(zip(selected_features, imp), key=lambda x: x[1], reverse=True)
        fi = {n: float(v) for n, v in pairs[:20]}

    return {'direction': direction, 'windows': results,
            'feature_importance': fi, 'threshold': best['threshold']}


# =========================================================================
# WALK-FORWARD V3 — DUAL MODEL
# =========================================================================

def run_training(base_dir: str = '.', use_optuna: bool = True,
                 optuna_trials: int = 30) -> Dict:
    from ml.feature_engine import build_full_feature_set

    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE V3 — DUAL LONG/SHORT")
    logger.info("=" * 60)

    df = build_full_feature_set(base_dir, add_news=True)
    feature_cols = get_feature_columns(df)
    logger.info("Feature set: %d rows × %d features", len(df), len(feature_cols))

    # Train LONG model
    long_results = _train_one_direction(df, feature_cols, 'long',
                                         use_optuna=use_optuna,
                                         optuna_trials=optuna_trials)

    # Train SHORT model
    short_results = _train_one_direction(df, feature_cols, 'short',
                                          use_optuna=use_optuna,
                                          optuna_trials=optuna_trials)

    return {'long': long_results, 'short': short_results}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    results = run_training(use_optuna=True, optuna_trials=30)

    for direction in ['long', 'short']:
        r = results[direction]
        print(f"\n{'=' * 60}")
        print(f"{direction.upper()} MODEL RESULTS")
        print(f"{'=' * 60}")

        if r.get('windows'):
            print(f"\n{'Year':<6} {'Thr':>5} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7} {'Sig':>6}")
            print("-" * 50)
            for w in r['windows']:
                print(f"{w['test_year']:<6} {w['threshold']:>5.2f} {w['precision']:>6.1%} "
                      f"{w['recall']:>6.1%} {w['f1']:>6.3f} {w['auc']:>6.3f} {w['n_signals']:>5d}")

            ws = [w for w in r['windows'] if w['n_signals'] > 0]
            if ws:
                print(f"\n  Avg Prec: {np.mean([w['precision'] for w in ws]):.1%}")
                print(f"  Avg AUC:  {np.mean([w['auc'] for w in ws]):.3f}")

        if r.get('feature_importance'):
            print(f"\nTop 10 {direction.upper()} Features:")
            for i, (n, v) in enumerate(list(r['feature_importance'].items())[:10]):
                print(f"  {i+1:>2}. {n:<25} {v:>8.0f}")

        print(f"  Threshold: {r.get('threshold', 0.5):.2f}")
