# ml/predictor.py
"""
ML Predictor V3: Dual LONG/SHORT models with Kelly-based position sizing.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MODELS_DIR = 'ml/models'


class MLPredictor:
    """Dual LONG/SHORT predictor with per-direction probability output."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.models = {'long': {}, 'short': {}}
        self.loaded = False
        self._load_models()

    def _load_models(self):
        for direction in ['long', 'short']:
            try:
                import lightgbm as lgb
                # Prefer binary (.bin) over text (.txt) — binary survives Git
                bin_path = os.path.join(self.models_dir, f'lgb_{direction}.bin')
                txt_path = os.path.join(self.models_dir, f'lgb_{direction}.txt')
                lgb_path = bin_path if os.path.exists(bin_path) else txt_path
                if os.path.exists(lgb_path):
                    self.models[direction]['lgb'] = lgb.Booster(model_file=lgb_path)

                keras_path = os.path.join(self.models_dir, f'lstm_{direction}.keras')
                if os.path.exists(keras_path):
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                    import tensorflow as tf
                    tf.get_logger().setLevel('ERROR')
                    self.models[direction]['lstm'] = tf.keras.models.load_model(keras_path)

                for key in ['lstm_scaler', 'lstm_features', 'meta', 'threshold', 'features']:
                    path = os.path.join(self.models_dir, f'{key}_{direction}.pkl')
                    if os.path.exists(path):
                        self.models[direction][key] = joblib.load(path)

            except Exception as e:
                logger.warning("Failed to load %s models: %s", direction, e)

        self.loaded = bool(self.models['long'].get('lgb'))
        if self.loaded:
            logger.info("MLPredictor V3 ready: LONG=%s, SHORT=%s",
                        "✓" if self.models['long'].get('lgb') else "✗",
                        "✓" if self.models['short'].get('lgb') else "✗")

    def predict_direction(self, features: pd.DataFrame) -> Tuple[Optional[str], float]:
        """
        Returns (direction, probability):
          ('long', 0.65) — LONG signal with 65% confidence
          ('short', 0.58) — SHORT signal
          (None, 0.0) — no signal
        """
        import config

        long_prob = self._predict_one('long', features)
        short_prob = self._predict_one('short', features)

        long_threshold = getattr(config, 'ML_LONG_THRESHOLD', 0.22)
        short_threshold = getattr(config, 'ML_SHORT_THRESHOLD', 0.35)
        min_conf = getattr(config, 'ML_MIN_CONFIDENCE', 0.15)

        long_signal = long_prob > long_threshold
        short_signal = short_prob > short_threshold

        if long_signal and short_signal:
            # Both signal — take the stronger one
            if long_prob > short_prob:
                return 'long', float(long_prob)
            else:
                return 'short', float(short_prob)
        elif long_signal and long_prob > min_conf:
            return 'long', float(long_prob)
        elif short_signal and short_prob > min_conf:
            return 'short', float(short_prob)

        return None, 0.0

    def _predict_one(self, direction: str, features: pd.DataFrame) -> float:
        m = self.models.get(direction, {})
        lgb_model = m.get('lgb')
        if lgb_model is None:
            return 0.5

        try:
            selected = m.get('features', features.columns.tolist())
            available = [c for c in selected if c in features.columns]
            X = features[available] if available else features

            lgb_prob = lgb_model.predict(X)[0]

            lstm_prob = 0.5
            lstm_model = m.get('lstm')
            if lstm_model and m.get('lstm_scaler'):
                from ml.trainer import predict_lstm
                lstm_probs = predict_lstm(lstm_model, m['lstm_scaler'],
                                          X, m.get('lstm_features', []))
                lstm_prob = lstm_probs[-1] if len(lstm_probs) > 0 else 0.5

            meta = m.get('meta')
            if meta:
                from ml.trainer import predict_stacking
                final = predict_stacking(meta, np.array([lgb_prob]), np.array([lstm_prob]))[0]
            else:
                final = lgb_prob

            return float(np.clip(final, 0, 1))
        except Exception as e:
            logger.warning("Predict %s error: %s", direction, e)
            return 0.5

    def kelly_size(self, probability: float, win_loss_ratio: float = 1.5) -> float:
        """
        Quarter-Kelly position sizing.
        f* = (p*b - q) / b, then × kelly_fraction
        """
        import config
        kelly_frac = getattr(config, 'ML_KELLY_FRACTION', 0.25)

        p = probability
        q = 1 - p
        b = win_loss_ratio  # TP/SL ratio (3:2 = 1.5)

        kelly_full = (p * b - q) / b
        if kelly_full <= 0:
            return 0.0

        return kelly_full * kelly_frac


# Singleton
_predictor: Optional[MLPredictor] = None


def get_predictor() -> MLPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor


def reset_predictor():
    global _predictor
    _predictor = None
