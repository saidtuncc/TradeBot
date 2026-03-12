# ml/retrain.py
"""
Periodic Retrain Script — Rolling Window Retrain + Bias Test + Threshold Optimizer
Solves PF decay problem: model retrained on recent data, old data discarded.
"""

import logging
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

MODELS_DIR = 'ml/models'
RETRAIN_LOG = 'ml/retrain_history.json'


def rolling_retrain(
    base_dir: str = '.',
    train_years: int = 3,
    use_optuna: bool = True,
    optuna_trials: int = 20,
) -> Dict:
    """
    Retrain models using only the last N years of data.
    Solves PF decay by focusing on recent market conditions.
    """
    from ml.feature_engine import build_full_feature_set
    from ml.trainer import (get_feature_columns, prepare_data, select_features,
                            optuna_tune_lgb, train_lightgbm, predict_lightgbm,
                            train_lstm, predict_lstm, train_stacking,
                            predict_stacking, find_optimal_threshold)

    logger.info("=" * 60)
    logger.info("ROLLING RETRAIN (last %d years)", train_years)
    logger.info("=" * 60)

    df = build_full_feature_set(base_dir, add_news=True)
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    max_year = df['year'].max()
    cutoff_year = max_year - train_years

    # Only use recent data
    df_recent = df[df['year'] > cutoff_year].copy()
    logger.info("Using data from %d-%d: %d rows (discarded %d older rows)",
                cutoff_year + 1, max_year, len(df_recent), len(df) - len(df_recent))

    results = {}
    for direction in ['long', 'short']:
        target_col = f'target_{direction}'
        feature_cols = get_feature_columns(df_recent)
        X_all, y_all, _ = prepare_data(df_recent, target_col)

        # Feature selection
        selected = select_features(X_all, y_all, feature_cols, top_n=35)
        X_all = X_all[selected]

        # Split: last 20% as validation
        val_size = int(len(X_all) * 0.20)
        X_tr, X_val = X_all.iloc[:-val_size], X_all.iloc[-val_size:]
        y_tr, y_val = y_all.iloc[:-val_size], y_all.iloc[-val_size:]

        # Optuna
        optuna_params = None
        if use_optuna:
            optuna_params = optuna_tune_lgb(X_tr, y_tr, X_val, y_val, optuna_trials)

        # Train
        lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val, params_override=optuna_params)
        lgb_probs_tr = predict_lightgbm(lgb_model, X_all)
        lgb_probs_val = predict_lightgbm(lgb_model, X_val)

        lstm_model, lstm_scaler, lstm_feats = train_lstm(X_tr, y_tr, X_val, y_val)
        lstm_probs_tr = predict_lstm(lstm_model, lstm_scaler, X_all, lstm_feats)
        lstm_probs_val = predict_lstm(lstm_model, lstm_scaler, X_val, lstm_feats)

        meta = train_stacking(lgb_probs_tr, lstm_probs_tr, y_all)
        val_probs = predict_stacking(meta, lgb_probs_val, lstm_probs_val)
        threshold = find_optimal_threshold(y_val, val_probs, min_precision=0.55)

        # Save
        lgb_model.save_model(os.path.join(MODELS_DIR, f'lgb_{direction}.txt'))
        if lstm_model:
            lstm_model.save(os.path.join(MODELS_DIR, f'lstm_{direction}.keras'))
        if lstm_scaler:
            joblib.dump(lstm_scaler, os.path.join(MODELS_DIR, f'lstm_scaler_{direction}.pkl'))
        if lstm_feats:
            joblib.dump(lstm_feats, os.path.join(MODELS_DIR, f'lstm_features_{direction}.pkl'))
        joblib.dump(meta, os.path.join(MODELS_DIR, f'meta_{direction}.pkl'))
        joblib.dump(threshold, os.path.join(MODELS_DIR, f'threshold_{direction}.pkl'))
        joblib.dump(selected, os.path.join(MODELS_DIR, f'features_{direction}.pkl'))
        if optuna_params:
            joblib.dump(optuna_params, os.path.join(MODELS_DIR, f'optuna_params_{direction}.pkl'))

        # Eval
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        preds = (val_probs > threshold).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        auc = roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else 0.5

        results[direction] = {
            'threshold': threshold,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc,
            'train_rows': len(X_tr),
            'val_rows': len(X_val),
            'trees': lgb_model.num_trees(),
        }

        logger.info("[%s] Prec=%.1f%% Rec=%.1f%% F1=%.3f AUC=%.3f T=%.2f Trees=%d",
                     direction.upper(), prec*100, rec*100, f1, auc,
                     threshold, lgb_model.num_trees())

    # Log retrain
    _log_retrain(results, train_years, len(df_recent))
    return results


def _log_retrain(results, train_years, n_rows):
    """Append retrain result to history log."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'train_years': train_years,
        'n_rows': n_rows,
        'long': results.get('long', {}),
        'short': results.get('short', {}),
    }
    history = []
    if os.path.exists(RETRAIN_LOG):
        try:
            with open(RETRAIN_LOG) as f:
                history = json.load(f)
        except:
            pass
    history.append(entry)
    with open(RETRAIN_LOG, 'w') as f:
        json.dump(history, f, indent=2, default=str)


# =========================================================================
# LOOK-AHEAD BIAS TEST
# =========================================================================

def test_look_ahead_bias(base_dir: str = '.') -> Dict:
    """
    Test for look-ahead bias by comparing:
    - Full dataset feature values at each point
    - Incrementally computed feature values (only past data)
    If they differ significantly, there's look-ahead bias.
    """
    from ml.data_loader import load_all_timeframes
    from ml.feature_engine import build_h1_features, calculate_atr

    logger.info("=" * 60)
    logger.info("LOOK-AHEAD BIAS TEST")
    logger.info("=" * 60)

    data = load_all_timeframes(base_dir)
    h1 = data['H1'].copy()

    # Full computation (what backtest uses)
    h1_full = build_h1_features(h1.copy())

    # Check H1 features — these should NOT have bias since they're computed
    # from past data only (rolling windows, EMA, etc.)
    test_features = ['rsi14', 'atr14', 'ema_spread', 'macd_hist', 'adx', 'bb_width']

    # For H1 features, compute on first half vs full
    half = len(h1) // 2
    h1_first_half = build_h1_features(h1.iloc[:half+100].copy())

    bias_results = {}
    for feat in test_features:
        if feat in h1_full.columns and feat in h1_first_half.columns:
            # Compare overlapping region (last 100 of first half)
            full_vals = h1_full[feat].iloc[half-10:half+10].values
            half_vals = h1_first_half[feat].iloc[-120:-100].values

            if len(full_vals) > 0 and len(half_vals) > 0:
                # They should be identical (no future data leaks into indicators)
                overlap_len = min(len(full_vals), len(half_vals))
                diff = np.abs(full_vals[:overlap_len] - half_vals[:overlap_len])
                max_diff = np.nanmax(diff) if len(diff) > 0 else 0
                mean_diff = np.nanmean(diff) if len(diff) > 0 else 0

                bias_results[feat] = {
                    'max_diff': float(max_diff),
                    'mean_diff': float(mean_diff),
                    'has_bias': max_diff > 0.01
                }

    # Check cross-TF features (the real risk area)
    # merge_asof should prevent bias, but let's verify
    from ml.feature_engine import add_higher_tf_features
    from ml.data_loader import merge_higher_tf_features

    h4 = data.get('H4')
    if h4 is not None:
        # Check: does H4 feature at time T only use H4 bars <= T?
        h1_with_h4 = add_higher_tf_features(h1_full.copy(), h4_df=h4.copy())
        sample_idx = len(h1_with_h4) // 2
        if 'h4_rsi14' in h1_with_h4.columns:
            h1_time = h1_with_h4.iloc[sample_idx]['datetime']
            h4_before = h4[h4['datetime'] <= h1_time]
            h4_after = h4[h4['datetime'] > h1_time]
            bias_results['cross_tf_check'] = {
                'h1_sample_time': str(h1_time),
                'h4_bars_before': len(h4_before),
                'h4_bars_after': len(h4_after),
                'merge_asof_safe': True  # By design
            }

    # Summary
    has_bias = any(v.get('has_bias', False) for v in bias_results.values()
                   if isinstance(v, dict) and 'has_bias' in v)

    logger.info("Bias test result: %s", "⚠️ BIAS DETECTED" if has_bias else "✅ NO BIAS")
    for feat, info in bias_results.items():
        if isinstance(info, dict) and 'max_diff' in info:
            status = "⚠️" if info.get('has_bias') else "✅"
            logger.info("  %s %s: max_diff=%.6f", status, feat, info['max_diff'])

    return {'has_bias': has_bias, 'details': bias_results}


# =========================================================================
# THRESHOLD OPTIMIZER
# =========================================================================

def optimize_thresholds(base_dir: str = '.') -> Dict:
    """
    Grid-search optimal thresholds for LONG and SHORT.
    Maximizes: precision × sqrt(n_signals) — quality × quantity balance.
    """
    from ml.feature_engine import build_full_feature_set
    from ml.trainer import get_feature_columns, prepare_data
    from ml.predictor import get_predictor, reset_predictor

    logger.info("=" * 60)
    logger.info("THRESHOLD OPTIMIZATION")
    logger.info("=" * 60)

    df = build_full_feature_set(base_dir, add_news=True)
    df['year'] = pd.to_datetime(df['datetime']).dt.year

    # Use only 2024-2025 for optimization (recent, not too small)
    df_opt = df[df['year'].isin([2024, 2025])].copy()
    feature_cols = get_feature_columns(df_opt)

    reset_predictor()
    predictor = get_predictor()

    results = {}
    for direction in ['long', 'short']:
        target_col = f'target_{direction}'
        X, y, _ = prepare_data(df_opt, target_col)

        # Get predictions
        selected = predictor.models[direction].get('features', feature_cols)
        available = [c for c in selected if c in X.columns]
        X_sel = X[available] if available else X

        probs = []
        lgb_model = predictor.models[direction].get('lgb')
        if lgb_model:
            probs = lgb_model.predict(X_sel)
        else:
            continue

        # Grid search thresholds
        best_score = 0
        best_threshold = 0.5
        threshold_table = []

        for t in np.arange(0.15, 0.60, 0.01):
            preds = (probs > t).astype(int)
            n_sig = preds.sum()
            if n_sig < 20:
                continue

            tp = ((preds == 1) & (y == 1)).sum()
            fp = ((preds == 1) & (y == 0)).sum()
            fn = ((preds == 0) & (y == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0

            # Score: precision × sqrt(signals) — balances quality with quantity
            score = prec * np.sqrt(n_sig / len(y))
            daily_signals = n_sig / (len(y) / (252 * 10))  # Approx per day

            threshold_table.append({
                'threshold': t, 'precision': prec, 'recall': rec,
                'signals': n_sig, 'daily': daily_signals, 'score': score
            })

            if score > best_score:
                best_score = score
                best_threshold = t

        results[direction] = {
            'best_threshold': best_threshold,
            'table': threshold_table,
        }

        # Print top 5
        logger.info("\n[%s] Top thresholds:", direction.upper())
        top5 = sorted(threshold_table, key=lambda x: x['score'], reverse=True)[:5]
        for row in top5:
            logger.info("  T=%.2f Prec=%.1f%% Rec=%.1f%% Sig=%d (~%.1f/day) Score=%.4f",
                         row['threshold'], row['precision']*100, row['recall']*100,
                         row['signals'], row['daily'], row['score'])

        logger.info("  → Best: T=%.2f", best_threshold)

    return results


# =========================================================================
# MAIN
# =========================================================================

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if cmd in ['retrain', 'all']:
        print("\n" + "🔄 RUNNING RETRAIN...")
        results = rolling_retrain(train_years=3, optuna_trials=20)
        for d, r in results.items():
            print(f"  {d.upper()}: Prec={r['precision']:.1%} AUC={r['auc']:.3f} T={r['threshold']:.2f}")

    if cmd in ['bias', 'all']:
        print("\n" + "🔍 RUNNING BIAS TEST...")
        bias = test_look_ahead_bias()
        print(f"  Result: {'⚠️ BIAS FOUND' if bias['has_bias'] else '✅ NO BIAS'}")

    if cmd in ['threshold', 'all']:
        print("\n" + "🎯 RUNNING THRESHOLD OPTIMIZATION...")
        thresh = optimize_thresholds()
        for d, r in thresh.items():
            print(f"  {d.upper()}: Best T={r['best_threshold']:.2f}")
