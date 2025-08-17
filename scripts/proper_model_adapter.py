#!/usr/bin/env python3
"""
Proper Model Adapter
Converts ADP player data into historical features for our leak-free AI model
Now uses shared feature builder for real historical features (no leakage),
with a safe fallback to mock features if needed.
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import json

# Ensure src/ is importable for shared feature builder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from models.feature_builder import build_live_features_for_players, REQUIRED_FEATURES
except Exception as _e:
    build_live_features_for_players = None
    try:
        # Minimal fallback list if import fails
        REQUIRED_FEATURES = [
            'hist_games_played',
            'hist_avg_passing_yards', 'hist_avg_passing_tds', 'hist_avg_interceptions',
            'hist_avg_rushing_yards', 'hist_avg_rushing_tds',
            'hist_avg_receiving_yards', 'hist_avg_receiving_tds',
            'hist_avg_receptions', 'hist_avg_targets', 'hist_avg_carries',
            'hist_std_fantasy_points', 'hist_max_fantasy_points', 'hist_min_fantasy_points',
            'recent_avg_fantasy_points', 'recent_vs_season_trend',
            'recent5_avg_fantasy_points', 'recent5_std_fantasy_points',
            'weighted_avg_fantasy_points',
            'eff_yards_per_target', 'eff_yards_per_carry',
            'rate_receiving_td', 'rate_rushing_td', 'rate_pass_td_to_int',
            'is_qb', 'is_rb', 'is_wr', 'is_te',
            'season_2022', 'season_2023', 'season_2024',
            'week', 'early_season', 'mid_season', 'late_season',
        ]
    except Exception:
        REQUIRED_FEATURES = []

class ProperModelAdapter:
    """Adapter to use proper AI model with ADP data"""
    
    def __init__(self):
        self.model = None
        # Default to shared REQUIRED_FEATURES; can be overridden by model metadata
        self.model_features = list(REQUIRED_FEATURES) if REQUIRED_FEATURES else []
        # Optional quantile models
        self.quantile_models = {"q05": None, "q50": None, "q95": None}
        self.load_model()
        # Cache weekly data for live features
        self._weekly_df = None
        # Try to load quantile models after base model to keep feature order ready
        self._load_quantile_models()
    
    def _load_latest_metadata_features(self, model_dir: str):
        """Load feature order from the latest metadata JSON if available."""
        try:
            meta_files = sorted(glob.glob(os.path.join(model_dir, 'proper_model_metadata_*.json')))
            if not meta_files:
                return False
            meta_path = meta_files[-1]
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            feats = meta.get('features')
            if isinstance(feats, list) and len(feats) > 0:
                self.model_features = list(feats)
                print(f"🔧 Loaded feature order from metadata: {meta_path} ({len(self.model_features)} features)")
                return True
            return False
        except Exception as e:
            print(f"⚠️ Could not load metadata features: {e}")
            return False
    
    def load_model(self):
        """Load the proper CatBoost model"""
        try:
            # Import joblib lazily to avoid import-time failures when the package isn't installed
            try:
                import joblib  # type: ignore
            except ImportError:
                print("❌ joblib not installed; AI model disabled. Ensure joblib is in your environment (pip install -r requirements.txt).")
                self.model = None
                return False
            # Prefer explicit env override if provided
            env_model_path = os.getenv('PROPER_MODEL_PATH')
            if env_model_path and os.path.exists(env_model_path):
                self.model = joblib.load(env_model_path)
                print(f"✅ Loaded proper AI model from env PROPER_MODEL_PATH: {env_model_path}")
                meta_path = os.getenv('PROPER_MODEL_METADATA')
                if meta_path and os.path.exists(meta_path):
                    print(f"ℹ️ Using metadata file: {meta_path}")
                # Sync feature order from model if available
                try:
                    model_feature_names = getattr(self.model, 'feature_names_', None)
                    if model_feature_names:
                        self.model_features = list(model_feature_names)
                        print(f"🔧 Synced feature order from model: {len(self.model_features)} features")
                except Exception:
                    pass
                return True
            elif env_model_path:
                print(f"⚠️ PROPER_MODEL_PATH set but not found: {env_model_path}")
            
            candidate_paths = [
                'models/proper_fantasy_model.pkl',
                '../models/proper_fantasy_model.pkl',
                'functions/models/proper_fantasy_model.pkl',
                'deploy_bundle/models/proper_fantasy_model.pkl',
                'deploy_bundle/functions/models/proper_fantasy_model.pkl'
            ]
            chosen_path = None
            for path in candidate_paths:
                if os.path.exists(path):
                    chosen_path = path
                    break
            if chosen_path is None:
                print(f"❌ Proper model not found. Tried: {candidate_paths}")
                self.model = None
                return False
            self.model = joblib.load(chosen_path)
            print(f"✅ Loaded proper AI model from: {chosen_path}")
            model_dir = os.path.dirname(chosen_path)
            # Load feature order from latest metadata if possible
            self._load_latest_metadata_features(model_dir)
            # Sync feature order from model if available
            try:
                model_feature_names = getattr(self.model, 'feature_names_', None)
                if model_feature_names:
                    self.model_features = list(model_feature_names)
                    print(f"🔧 Synced feature order from model: {len(self.model_features)} features")
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"❌ Error loading proper model: {e}")
            self.model = None
            return False

    def _load_quantile_models(self):
        """Attempt to load optional quantile models (q05/q50/q95) from common locations."""
        # Import joblib lazily
        try:
            import joblib  # type: ignore
        except Exception:
            print("⚠️ joblib not available; skipping quantile models load")
            return False

        tags = ["q05", "q50", "q95"]
        candidates_per_tag = {
            tag: [
                f"models/proper_fantasy_model_{tag}.pkl",
                f"../models/proper_fantasy_model_{tag}.pkl",
                f"functions/models/proper_fantasy_model_{tag}.pkl",
                f"deploy_bundle/models/proper_fantasy_model_{tag}.pkl",
                f"deploy_bundle/functions/models/proper_fantasy_model_{tag}.pkl",
            ]
            for tag in tags
        }

        loaded_any = False
        for tag, paths in candidates_per_tag.items():
            model_loaded = False
            # Allow environment variable override, e.g., PROPER_Q50_MODEL_PATH
            env_key = f"PROPER_{tag.upper()}_MODEL_PATH"
            env_path = os.getenv(env_key)
            try_paths = [env_path] + paths if env_path else paths
            for p in try_paths:
                if p and os.path.exists(p):
                    try:
                        self.quantile_models[tag] = joblib.load(p)
                        print(f"✅ Loaded quantile model {tag} from: {p}")
                        model_loaded = True
                        loaded_any = True
                        break
                    except Exception as e:
                        print(f"⚠️ Failed to load quantile model {tag} from {p}: {e}")
            if not model_loaded:
                self.quantile_models[tag] = None
        return loaded_any
    
    def create_mock_historical_features(self, players_df):
        """
        Create mock historical features from ADP data
        Since we don't have real historical data for 2025 ADP players,
        we'll create reasonable estimates based on position and ADP ranking
        """
        features_df = players_df.copy()
        
        # Position encoding
        features_df['is_qb'] = (features_df['position'] == 'QB').astype(int)
        features_df['is_rb'] = (features_df['position'] == 'RB').astype(int)
        features_df['is_wr'] = (features_df['position'] == 'WR').astype(int)
        features_df['is_te'] = (features_df['position'] == 'TE').astype(int)
        
        # Season encoding (assume current season 2024)
        features_df['season_2022'] = 0
        features_df['season_2023'] = 0
        features_df['season_2024'] = 1
        
        # Week features (assume mid-season for draft)
        features_df['week'] = 8
        features_df['early_season'] = 0
        features_df['mid_season'] = 1
        features_df['late_season'] = 0
        
        # Mock historical games played (veteran vs rookie estimates)
        features_df['hist_games_played'] = features_df.apply(
            lambda row: 12 if 'rookie' not in str(row.get('name', '')).lower() else 0, axis=1
        )
        
        # Create position-based historical stat estimates from ADP ranking
        for _, row in features_df.iterrows():
            adp = row.get('adp_rank', 100)
            position = row['position']
            
            # Better ADP = higher estimated historical performance
            # Scale inversely with ADP (lower ADP rank = higher stats)
            performance_factor = max(0.1, (200 - adp) / 200)
            
            if position == 'QB':
                features_df.loc[_, 'hist_avg_passing_yards'] = 220 * performance_factor
                features_df.loc[_, 'hist_avg_passing_tds'] = 1.8 * performance_factor
                features_df.loc[_, 'hist_avg_interceptions'] = 0.8 * (1 - performance_factor * 0.5)
                features_df.loc[_, 'hist_avg_rushing_yards'] = 15 * performance_factor
                features_df.loc[_, 'hist_avg_rushing_tds'] = 0.3 * performance_factor
                features_df.loc[_, 'recent_avg_fantasy_points'] = 18 * performance_factor
                
            elif position == 'RB':
                features_df.loc[_, 'hist_avg_rushing_yards'] = 75 * performance_factor
                features_df.loc[_, 'hist_avg_rushing_tds'] = 0.6 * performance_factor
                features_df.loc[_, 'hist_avg_carries'] = 14 * performance_factor
                features_df.loc[_, 'hist_avg_receptions'] = 3 * performance_factor
                features_df.loc[_, 'hist_avg_receiving_yards'] = 25 * performance_factor
                features_df.loc[_, 'hist_avg_targets'] = 4 * performance_factor
                features_df.loc[_, 'recent_avg_fantasy_points'] = 12 * performance_factor
                
            elif position == 'WR':
                features_df.loc[_, 'hist_avg_receptions'] = 4.5 * performance_factor
                features_df.loc[_, 'hist_avg_receiving_yards'] = 65 * performance_factor
                features_df.loc[_, 'hist_avg_receiving_tds'] = 0.5 * performance_factor
                features_df.loc[_, 'hist_avg_targets'] = 7 * performance_factor
                features_df.loc[_, 'recent_avg_fantasy_points'] = 11 * performance_factor
                
            elif position == 'TE':
                features_df.loc[_, 'hist_avg_receptions'] = 3.2 * performance_factor
                features_df.loc[_, 'hist_avg_receiving_yards'] = 40 * performance_factor
                features_df.loc[_, 'hist_avg_receiving_tds'] = 0.4 * performance_factor
                features_df.loc[_, 'hist_avg_targets'] = 5 * performance_factor
                features_df.loc[_, 'recent_avg_fantasy_points'] = 8 * performance_factor
        
        # Fill in remaining features with defaults
        stat_columns = [
            'hist_avg_passing_yards', 'hist_avg_passing_tds', 'hist_avg_interceptions',
            'hist_avg_rushing_yards', 'hist_avg_rushing_tds',
            'hist_avg_receiving_yards', 'hist_avg_receiving_tds', 
            'hist_avg_receptions', 'hist_avg_targets', 'hist_avg_carries'
        ]
        
        for col in stat_columns:
            if col not in features_df.columns:
                features_df[col] = 0
            features_df[col] = features_df[col].fillna(0)
        
        # Historical consistency features (mock based on ADP)
        features_df['hist_std_fantasy_points'] = features_df['recent_avg_fantasy_points'] * 0.4
        features_df['hist_max_fantasy_points'] = features_df['recent_avg_fantasy_points'] * 1.8
        features_df['hist_min_fantasy_points'] = features_df['recent_avg_fantasy_points'] * 0.2
        features_df['recent_vs_season_trend'] = 0  # No trend data available
        
        return features_df
    
    def load_weekly_data(self):
        """Load combined weekly data for live feature building."""
        if self._weekly_df is not None:
            return self._weekly_df
        candidates = [
            'data/nflverse/combined_weekly_2022_2024.csv',
            os.path.join(ROOT_DIR, 'data', 'nflverse', 'combined_weekly_2022_2024.csv'),
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    self._weekly_df = pd.read_csv(p)
                    print(f"📚 Loaded weekly data for live features: {p} ({len(self._weekly_df)} rows)")
                    return self._weekly_df
                except Exception as e:
                    print(f"⚠️ Failed to read weekly data {p}: {e}")
        print("⚠️ Weekly data not found; will fallback to mock features")
        self._weekly_df = pd.DataFrame()
        return self._weekly_df

    def get_ai_predictions(self, players_df):
        """Get AI predictions for a list of players"""
        if self.model is None and self.quantile_models.get("q50") is None:
            print("❌ No model available (neither point-estimate nor quantile p50). Cannot make predictions")
            return None
        
        try:
            # Prefer real historical features if builder and weekly data are available
            features_df = None
            if build_live_features_for_players is not None:
                weekly_df = self.load_weekly_data()
                if weekly_df is not None and not weekly_df.empty:
                    try:
                        features_df = build_live_features_for_players(players_df, weekly_df)
                        print(f"✅ Built real live features for {len(features_df)} players")
                    except Exception as e:
                        print(f"⚠️ Live feature build failed, will fallback to mock: {e}")
                        features_df = None
            if features_df is None:
                features_df = self.create_mock_historical_features(players_df)
                print(f"ℹ️ Using mock features for {len(features_df)} players")

            # Ensure all required features are present and ordered
            feature_order = self.model_features if self.model_features else list(REQUIRED_FEATURES)
            for feature in feature_order:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            X = features_df[feature_order]
            # Point estimate from base model if available; else fallback to p50 quantile
            if self.model is not None:
                predictions = self.model.predict(X)
            else:
                predictions = self.quantile_models["q50"].predict(X) if self.quantile_models.get("q50") is not None else np.zeros(len(X))
            
            # Add predictions to the dataframe
            result_df = players_df.copy()
            result_df['ai_prediction'] = predictions
            
            # Optional quantile predictions
            try:
                q05_model = self.quantile_models.get("q05")
                q50_model = self.quantile_models.get("q50")
                q95_model = self.quantile_models.get("q95")
                if q05_model is not None:
                    result_df['ai_p5'] = q05_model.predict(X)
                if q50_model is not None:
                    result_df['ai_p50'] = q50_model.predict(X)
                if q95_model is not None:
                    result_df['ai_p95'] = q95_model.predict(X)
                # Convenience bounds if both are present
                if 'ai_p5' in result_df.columns and 'ai_p95' in result_df.columns:
                    result_df['ai_lower'] = result_df['ai_p5']
                    result_df['ai_upper'] = result_df['ai_p95']
            except Exception as _qe:
                # Do not fail predictions if quantiles error out
                print(f"⚠️ Quantile prediction step failed: {_qe}")
            
            print(f"✅ Generated AI predictions for {len(result_df)} players")
            return result_df
            
        except Exception as e:
            print(f"❌ Error generating AI predictions: {e}")
            return None
    
    def is_available(self):
        """Check if the model is available for predictions"""
        # Available if either base model or p50 quantile model exists
        return (self.model is not None) or (self.quantile_models.get("q50") is not None)

# Global instance
_adapter = None

def get_adapter():
    """Get the global adapter instance"""
    global _adapter
    if _adapter is None:
        _adapter = ProperModelAdapter()
    return _adapter

def predict_players(players_df):
    """Simple function to get predictions for players"""
    adapter = get_adapter()
    return adapter.get_ai_predictions(players_df)

def is_model_available():
    """Check if the AI model is available"""
    adapter = get_adapter()
    return adapter.is_available() 