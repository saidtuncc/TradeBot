# ml/predictor.py
"""
ML Predictor V4: Multi-Timeframe Ensemble (H1 + M15 + M5)
Weighted voting across 3 models for robust signal generation.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

MODELS_DIR = 'ml/models'

# Ensemble weights: H1=strategic, M15=tactical, M5=scalp
WEIGHTS = {
    'h1':  0.50,   # Strategic direction — highest weight
    'm15': 0.30,   # Tactical confirmation
    'm5':  0.20,   # Scalp timing
}


class MLPredictor:
    """
    Multi-TF Ensemble Predictor.

    Loads H1, M15, M5 models and combines predictions via weighted average.
    Falls back gracefully: if only H1 available, works like V3.
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.timeframes = {}  # {'h1': {'long': bundle, 'short': bundle}, ...}
        self.loaded = False
        self._load_all_models()

    def _load_all_models(self):
        """Load all available model bundles (H1, M15, M5)."""

        # New-style .pkl bundles (M5, M15)
        for tf in ['m5', 'm15']:
            for direction in ['long', 'short']:
                path = os.path.join(self.models_dir, f'{tf}_{direction}.pkl')
                if os.path.exists(path):
                    try:
                        bundle = joblib.load(path)
                        if tf not in self.timeframes:
                            self.timeframes[tf] = {}
                        self.timeframes[tf][direction] = bundle
                    except Exception as e:
                        logger.warning("Failed to load %s_%s: %s", tf, direction, e)

        # H1: try new .pkl first, then old lgb_*.bin
        for direction in ['long', 'short']:
            pkl_path = os.path.join(self.models_dir, f'h1_{direction}.pkl')
            if os.path.exists(pkl_path):
                try:
                    bundle = joblib.load(pkl_path)
                    if 'h1' not in self.timeframes:
                        self.timeframes['h1'] = {}
                    self.timeframes['h1'][direction] = bundle
                except Exception as e:
                    logger.warning("Failed to load h1_%s.pkl: %s", direction, e)
            else:
                # Fallback: old-style lgb_*.bin
                self._load_legacy_h1(direction)

        # Summary
        available = []
        for tf in ['h1', 'm15', 'm5']:
            if tf in self.timeframes:
                dirs = list(self.timeframes[tf].keys())
                available.append(f"{tf.upper()}({'/'.join(dirs)})")

        self.loaded = len(self.timeframes) > 0
        if self.loaded:
            logger.info("Ensemble Predictor V4: %s", ' + '.join(available))
        else:
            logger.warning("No models found!")

    def _load_legacy_h1(self, direction: str):
        """Load old H1 model format (lgb_*.bin + features_*.json)."""
        try:
            import lightgbm as lgb

            bin_path = os.path.join(self.models_dir, f'lgb_{direction}.bin')
            txt_path = os.path.join(self.models_dir, f'lgb_{direction}.txt')
            lgb_path = bin_path if os.path.exists(bin_path) else txt_path

            if not os.path.exists(lgb_path):
                return

            model = lgb.Booster(model_file=lgb_path)

            # Load features list
            features = None
            json_path = os.path.join(self.models_dir, f'features_{direction}.json')
            if os.path.exists(json_path):
                import json
                with open(json_path) as f:
                    features = json.load(f)

            bundle = {
                'model': model,
                'features': features,
                'calibrator': None,
                'scaler': None,
                'legacy': True,
            }

            if 'h1' not in self.timeframes:
                self.timeframes['h1'] = {}
            self.timeframes['h1'][direction] = bundle
            logger.info("Loaded legacy H1 %s model", direction)

        except Exception as e:
            logger.warning("Failed to load legacy H1 %s: %s", direction, e)

    # ─── PREDICTION ─────────────────────────────────────────────────

    def predict_single(self, timeframe: str, direction: str,
                       features: pd.DataFrame) -> float:
        """Predict probability for one timeframe + direction."""
        tf_models = self.timeframes.get(timeframe, {})
        bundle = tf_models.get(direction)
        if bundle is None:
            return 0.5

        try:
            model = bundle['model']
            selected = bundle.get('features')

            if selected:
                # Align features
                X = pd.DataFrame(0.0, index=features.index, columns=selected)
                for col in selected:
                    if col in features.columns:
                        X[col] = features[col].values
            else:
                X = features

            X = X.replace([np.inf, -np.inf], 0).fillna(0)

            # Predict
            raw = model.predict(X)
            if hasattr(raw, '__len__') and len(raw) > 0:
                raw_prob = float(raw[0]) if len(raw.shape) == 1 else float(raw[0])
            else:
                raw_prob = float(raw)

            # Calibrate if available
            calibrator = bundle.get('calibrator')
            scaler = bundle.get('scaler')
            if calibrator and scaler:
                cal_input = scaler.transform(np.array([[raw_prob]]))
                prob = float(calibrator.predict_proba(cal_input)[0][1])
            else:
                prob = raw_prob

            return float(np.clip(prob, 0, 1))

        except Exception as e:
            logger.warning("Predict %s/%s error: %s", timeframe, direction, e)
            return 0.5

    def predict_ensemble(self, features_dict: Dict[str, pd.DataFrame]
                         ) -> Tuple[Optional[str], float, Dict]:
        """
        Ensemble prediction across all available timeframes.

        Args:
            features_dict: {'h1': df, 'm15': df, 'm5': df}
                Each df should have the appropriate features for that TF.

        Returns:
            (direction, probability, details)
            details = {'h1_long': 0.65, 'h1_short': 0.35, ...}
        """
        import config

        long_threshold = getattr(config, 'ML_LONG_THRESHOLD', 0.22)
        short_threshold = getattr(config, 'ML_SHORT_THRESHOLD', 0.35)

        details = {}
        long_weighted = 0.0
        short_weighted = 0.0
        total_weight = 0.0

        for tf in ['h1', 'm15', 'm5']:
            if tf not in self.timeframes or tf not in features_dict:
                continue

            feats = features_dict[tf]
            w = WEIGHTS.get(tf, 0.0)

            long_prob = self.predict_single(tf, 'long', feats)
            short_prob = self.predict_single(tf, 'short', feats)

            details[f'{tf}_long'] = round(long_prob, 4)
            details[f'{tf}_short'] = round(short_prob, 4)

            long_weighted += long_prob * w
            short_weighted += short_prob * w
            total_weight += w

        if total_weight == 0:
            return None, 0.0, details

        # Normalize
        long_final = long_weighted / total_weight
        short_final = short_weighted / total_weight

        details['ensemble_long'] = round(long_final, 4)
        details['ensemble_short'] = round(short_final, 4)

        # Signal decision
        long_signal = long_final > long_threshold
        short_signal = short_final > short_threshold

        if long_signal and short_signal:
            if long_final > short_final:
                return 'long', float(long_final), details
            else:
                return 'short', float(short_final), details
        elif long_signal:
            return 'long', float(long_final), details
        elif short_signal:
            return 'short', float(short_final), details

        return None, 0.0, details

    def predict_direction(self, features: pd.DataFrame) -> Tuple[Optional[str], float]:
        """
        Backward-compatible single-TF predict (H1 only).
        Used by existing _get_ml_signal().
        """
        import config

        long_prob = self.predict_single('h1', 'long', features)
        short_prob = self.predict_single('h1', 'short', features)

        long_threshold = getattr(config, 'ML_LONG_THRESHOLD', 0.22)
        short_threshold = getattr(config, 'ML_SHORT_THRESHOLD', 0.35)

        long_signal = long_prob > long_threshold
        short_signal = short_prob > short_threshold

        if long_signal and short_signal:
            if long_prob > short_prob:
                return 'long', float(long_prob)
            else:
                return 'short', float(short_prob)
        elif long_signal:
            return 'long', float(long_prob)
        elif short_signal:
            return 'short', float(short_prob)

        return None, 0.0

    # ─── AGREEMENT ─────────────────────────────────────────────────

    def get_agreement(self, features_dict: Dict[str, pd.DataFrame],
                      direction: str) -> Tuple[int, int, float]:
        """
        How many timeframes agree on a direction?
        Returns: (agree_count, total_count, max_prob)
        """
        agree = 0
        total = 0
        max_prob = 0.0

        for tf in ['h1', 'm15', 'm5']:
            if tf not in self.timeframes or tf not in features_dict:
                continue
            total += 1
            prob = self.predict_single(tf, direction, features_dict[tf])
            if prob > 0.5:
                agree += 1
                max_prob = max(max_prob, prob)

        return agree, total, max_prob

    # ─── KELLY SIZING ──────────────────────────────────────────────

    def kelly_size(self, probability: float, win_loss_ratio: float = 1.5) -> float:
        """Quarter-Kelly position sizing with agreement bonus."""
        import config
        kelly_frac = getattr(config, 'ML_KELLY_FRACTION', 0.25)

        p = probability
        q = 1 - p
        b = win_loss_ratio

        kelly_full = (p * b - q) / b
        if kelly_full <= 0:
            return 0.0

        return kelly_full * kelly_frac


# ─── SINGLETON ──────────────────────────────────────────────────

_predictor: Optional[MLPredictor] = None


def get_predictor() -> MLPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor


def reset_predictor():
    global _predictor
    _predictor = None
