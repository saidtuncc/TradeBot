#!/usr/bin/env python
"""
Train M15 Tactical Model — Full pipeline with thorough optimization
Usage: python ml/train_m15.py
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
    from ml.feature_engine import build_full_m15_set

    logger.info("=" * 60)
    logger.info("M15 TACTICAL MODEL TRAINING")
    logger.info("=" * 60)

    # 1. Build M15 feature set
    df = build_full_m15_set('.')
    logger.info("Dataset: %d rows, %d columns", len(df), len(df.columns))

    # 2. Time-based split (80/20, no look-ahead)
    df = df.sort_values('datetime').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info("Train: %d rows (%s to %s)",
                len(train_df), train_df['datetime'].min(), train_df['datetime'].max())
    logger.info("Test:  %d rows (%s to %s)",
                len(test_df), test_df['datetime'].min(), test_df['datetime'].max())

    # 3. Train LONG
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING M15 LONG MODEL")
    logger.info("=" * 50)
    _train_and_save(train_df, test_df, 'target_long', 'm15_long')

    # 4. Train SHORT
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING M15 SHORT MODEL")
    logger.info("=" * 50)
    _train_and_save(train_df, test_df, 'target_short', 'm15_short')

    logger.info("\n✅ M15 model training complete!")


EXCLUDE = [
    'datetime', 'open', 'high', 'low', 'close', 'volume',
    'target', 'target_reason', 'target_pnl_ratio',
    'target_long', 'target_short', 'target_3class',
    'ema10', 'ema20', 'ema50', 'ema100', 'atr14', 'atr_avg50',
]


def _train_and_save(train_df, test_df, target_col, model_name):
    """Full training pipeline: feature selection → Optuna → train → calibrate → evaluate."""
    import lightgbm as lgb
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, classification_report)

    feature_cols = [c for c in train_df.columns if c not in EXCLUDE and not c.startswith('target')]

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test_df[target_col]

    logger.info("Features: %d | Target: %s", len(feature_cols), target_col)
    logger.info("Train: %d (%.1f%% positive) | Test: %d (%.1f%% positive)",
                len(y_train), y_train.mean() * 100, len(y_test), y_test.mean() * 100)

    # --- Step 1: Feature selection (top 45) ---
    logger.info("Step 1: Feature selection...")
    selected = _select_top_features(X_train, y_train, feature_cols, top_n=45)
    logger.info("Selected %d features", len(selected))
    logger.info("Top 15: %s", selected[:15])

    X_train_sel = X_train[selected]
    X_test_sel = X_test[selected]

    # --- Step 2: Optuna tuning (100 trials) ---
    logger.info("Step 2: Optuna hyperparameter search (100 trials)...")
    best_params = _optuna_tune(X_train_sel, y_train, n_trials=100)
    logger.info("Best params: %s", best_params)

    # --- Step 3: Train final model ---
    logger.info("Step 3: Training final model (500 rounds)...")
    train_data = lgb.Dataset(X_train_sel, label=y_train)
    val_data = lgb.Dataset(X_test_sel, label=y_test, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'n_jobs': -1,
        **best_params
    }

    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ]
    )
    logger.info("Best iteration: %d", model.best_iteration)

    # --- Step 4: Probability calibration ---
    logger.info("Step 4: Calibrating probabilities...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    raw_probs = model.predict(X_train_sel)
    scaler = StandardScaler()
    cal_input = scaler.fit_transform(raw_probs.reshape(-1, 1))
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(cal_input, y_train)

    # --- Step 5: Evaluate ---
    logger.info("Step 5: Evaluating on test set...")
    test_raw = model.predict(X_test_sel)
    test_cal = calibrator.predict_proba(scaler.transform(test_raw.reshape(-1, 1)))[:, 1]
    test_pred = (test_cal > 0.5).astype(int)

    acc = accuracy_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_cal)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)

    logger.info("\n" + "=" * 40)
    logger.info("%s RESULTS:", model_name.upper())
    logger.info("=" * 40)
    logger.info("  Accuracy:  %.4f", acc)
    logger.info("  AUC:       %.4f", auc)
    logger.info("  Precision: %.4f", prec)
    logger.info("  Recall:    %.4f", rec)
    logger.info("  F1:        %.4f", f1)
    logger.info("\n%s", classification_report(y_test, test_pred))

    # Feature importance
    importances = model.feature_importance(importance_type='gain')
    fi = sorted(zip(selected, importances), key=lambda x: x[1], reverse=True)
    logger.info("Top 10 features:")
    for name, imp in fi[:10]:
        logger.info("  %-25s %.0f", name, imp)

    # --- Step 6: Save ---
    bundle = {
        'model': model,
        'calibrator': calibrator,
        'scaler': scaler,
        'features': selected,
        'metrics': {'acc': acc, 'auc': auc, 'prec': prec, 'rec': rec, 'f1': f1},
        'best_iteration': model.best_iteration,
    }
    path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    joblib.dump(bundle, path, compress=3)
    logger.info("Saved: %s (features=%d, AUC=%.4f)", path, len(selected), auc)


def _select_top_features(X, y, feature_cols, top_n=45):
    """Select top features by LightGBM importance + correlation cleanup."""
    import lightgbm as lgb

    train_data = lgb.Dataset(X, label=y)
    params = {'objective': 'binary', 'verbose': -1, 'num_leaves': 31,
              'learning_rate': 0.1, 'n_jobs': -1, 'max_depth': 5}
    quick = lgb.train(params, train_data, num_boost_round=150,
                      callbacks=[lgb.log_evaluation(period=0)])
    importances = quick.feature_importance(importance_type='gain')
    fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in fi[:top_n]]

    # Remove highly correlated (>0.95)
    corr = X[selected].corr().abs()
    to_drop = set()
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if corr.iloc[i, j] > 0.95:
                to_drop.add(selected[j])  # Drop the less important one

    final = [f for f in selected if f not in to_drop]
    logger.info("Removed %d correlated features", len(to_drop))
    return final


def _optuna_tune(X, y, n_trials=100):
    """Optuna hyperparameter search for LightGBM with time-series CV."""
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
        }

        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
            y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_t, label=y_t)
            val_data = lgb.Dataset(X_v, label=y_v, reference=train_data)

            model = lgb.train(
                {**params, 'objective': 'binary', 'metric': 'auc',
                 'verbose': -1, 'n_jobs': -1},
                train_data,
                num_boost_round=300,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(period=0)]
            )

            from sklearn.metrics import roc_auc_score
            preds = model.predict(X_v)
            scores.append(roc_auc_score(y_v, preds))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best CV AUC: %.4f (trial %d)", study.best_value, study.best_trial.number)
    return study.best_params


if __name__ == '__main__':
    main()
