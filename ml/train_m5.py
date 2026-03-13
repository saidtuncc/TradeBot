#!/usr/bin/env python
"""
Train M5 Scalp Model — Same pipeline as H1 (V3 trainer)
Usage: python ml/train_m5.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = 'ml/models'
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    from ml.feature_engine import build_full_m5_set
    from ml.trainer import (prepare_data, select_features, EXCLUDE_COLS,
                            get_feature_columns)

    # 1. Build M5 feature set
    logger.info("=" * 60)
    logger.info("M5 SCALP MODEL TRAINING")
    logger.info("=" * 60)

    df = build_full_m5_set('.')
    logger.info("Dataset: %d rows, %d columns", len(df), len(df.shape))

    # 2. Time-based split (no look-ahead)
    df = df.sort_values('datetime').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    logger.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))
    logger.info("Train period: %s to %s", train_df['datetime'].min(), train_df['datetime'].max())
    logger.info("Test period:  %s to %s", test_df['datetime'].min(), test_df['datetime'].max())

    # 3. Train LONG model
    logger.info("\n" + "=" * 40)
    logger.info("Training M5 LONG model...")
    logger.info("=" * 40)
    _train_and_save(train_df, test_df, 'target_long', 'm5_long')

    # 4. Train SHORT model
    logger.info("\n" + "=" * 40)
    logger.info("Training M5 SHORT model...")
    logger.info("=" * 40)
    _train_and_save(train_df, test_df, 'target_short', 'm5_short')

    logger.info("\n✅ M5 model training complete!")


def _train_and_save(train_df, test_df, target_col, model_name):
    """Train, evaluate, and save one model."""
    import lightgbm as lgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, classification_report)

    EXCLUDE = [
        'datetime', 'open', 'high', 'low', 'close', 'volume',
        'target', 'target_reason', 'target_pnl_ratio',
        'target_long', 'target_short', 'target_3class',
        'ema10', 'ema20', 'ema50', 'atr14', 'atr_avg50',
    ]

    feature_cols = [c for c in train_df.columns if c not in EXCLUDE and not c.startswith('target')]

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test_df[target_col]

    logger.info("Features: %d | Target: %s", len(feature_cols), target_col)
    logger.info("Train target rate: %.1f%% | Test target rate: %.1f%%",
                y_train.mean() * 100, y_test.mean() * 100)

    # Feature selection
    logger.info("Selecting top features...")
    selected = _select_top_features(X_train, y_train, feature_cols, top_n=45)
    logger.info("Selected %d features: %s", len(selected), selected[:10])

    X_train_sel = X_train[selected]
    X_test_sel = X_test[selected]

    # Optuna hyperparameter tuning
    logger.info("Optuna tuning (80 trials)...")
    best_params = _optuna_tune(X_train_sel, y_train, n_trials=80)
    logger.info("Best params: %s", best_params)

    # Train final model
    train_data = lgb.Dataset(X_train_sel, label=y_train)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'n_jobs': -1,
        **best_params
    }

    model = lgb.train(
        params, train_data,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(period=0)]
    )

    # Calibration
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    raw_probs = model.predict(X_train_sel)
    scaler = StandardScaler()
    cal_input = scaler.fit_transform(raw_probs.reshape(-1, 1))
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(cal_input, y_train)

    # Evaluate
    test_raw = model.predict(X_test_sel)
    test_cal = calibrator.predict_proba(scaler.transform(test_raw.reshape(-1, 1)))[:, 1]
    test_pred = (test_cal > 0.5).astype(int)

    acc = accuracy_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_cal)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)

    logger.info("\n%s TEST RESULTS:", model_name.upper())
    logger.info("  Accuracy:  %.4f", acc)
    logger.info("  AUC:       %.4f", auc)
    logger.info("  Precision: %.4f", prec)
    logger.info("  Recall:    %.4f", rec)
    logger.info("  F1:        %.4f", f1)
    logger.info("\n%s", classification_report(y_test, test_pred))

    # Save
    bundle = {
        'model': model,
        'calibrator': calibrator,
        'scaler': scaler,
        'features': selected,
        'metrics': {'acc': acc, 'auc': auc, 'prec': prec, 'rec': rec, 'f1': f1},
    }
    path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    joblib.dump(bundle, path, compress=3)
    logger.info("Saved: %s (features=%d, AUC=%.4f)", path, len(selected), auc)


def _select_top_features(X, y, feature_cols, top_n=35):
    """Select top features by LightGBM importance."""
    import lightgbm as lgb

    train_data = lgb.Dataset(X, label=y)
    params = {'objective': 'binary', 'verbose': -1, 'num_leaves': 31,
              'learning_rate': 0.1, 'n_jobs': -1, 'max_depth': 5}
    quick = lgb.train(params, train_data, num_boost_round=100,
                      callbacks=[lgb.log_evaluation(period=0)])
    importances = quick.feature_importance(importance_type='gain')
    fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in fi[:top_n]]

    # Remove highly correlated
    corr = X[selected].corr().abs()
    to_drop = set()
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if corr.iloc[i, j] > 0.95:
                to_drop.add(selected[j])
    return [f for f in selected if f not in to_drop]


def _optuna_tune(X, y, n_trials=20):
    """Optuna hyperparameter search for LightGBM."""
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        }

        model = lgb.LGBMClassifier(
            n_estimators=200, verbose=-1, n_jobs=-1, **params
        )
        scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best AUC: %.4f", study.best_value)
    return study.best_params


if __name__ == '__main__':
    main()
