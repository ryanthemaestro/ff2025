#!/usr/bin/env python3
"""
Rebuild Model Architecture
Implements position-specific models with ensemble approach for better real-world performance
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class PositionSpecificModelBuilder:
    def __init__(self):
        self.models = {}
        self.ensemble_weights = {
            'catboost': 0.70,
            'adp_consensus': 0.15,
            'volume_projection': 0.10,
            'sos_adjustment': 0.05
        }
        
    def load_expanded_data(self):
        """Load the expanded training dataset"""
        print("🏈 LOADING EXPANDED TRAINING DATA")
        print("=" * 40)
        
        try:
            # Load the expanded dataset we created
            expanded_df = pd.read_csv('expanded_training_data_20250803_203851.csv')
            print(f"✅ Loaded expanded dataset: {len(expanded_df)} players")
            
            # Verify we have the target variable
            if 'fantasy_points_target' in expanded_df.columns:
                print(f"✅ Target variable found: fantasy_points_target")
                non_zero_targets = len(expanded_df[expanded_df['fantasy_points_target'] > 0])
                print(f"📊 Players with valid targets: {non_zero_targets} ({non_zero_targets/len(expanded_df)*100:.1f}%)")
            else:
                print("❌ Target variable missing - using fantasy_points_ppr")
                expanded_df['fantasy_points_target'] = expanded_df.get('fantasy_points_ppr', 0)
            
            return expanded_df
            
        except FileNotFoundError:
            print("❌ Expanded dataset not found. Creating basic dataset...")
            return self.create_basic_dataset()
    
    def create_basic_dataset(self):
        """Fallback: Create basic dataset if expanded one not available"""
        try:
            actual_2024 = pd.read_csv('data/fantasy_metrics_2024.csv')
            print(f"✅ Using 2024 actual data: {len(actual_2024)} players")
            
            # Create basic features for fallback
            basic_df = actual_2024.copy()
            basic_df['fantasy_points_target'] = basic_df['fantasy_points_ppr']
            basic_df['weighted_historical_avg'] = basic_df['fantasy_points_ppr']
            basic_df['consistency_score'] = 0.5  # Neutral
            basic_df['performance_trend'] = 0.0  # Neutral
            basic_df['seasons_played'] = 1
            basic_df['total_career_points'] = basic_df['fantasy_points_ppr']
            basic_df['games_played'] = basic_df.get('games', 16)
            
            return basic_df
            
        except Exception as e:
            print(f"❌ Error creating basic dataset: {e}")
            return None
    
    def prepare_position_features(self, df, position):
        """Create position-specific features"""
        print(f"\n🎯 PREPARING {position} FEATURES")
        print("-" * 30)
        
        pos_df = df[df['position'] == position].copy()
        print(f"Players: {len(pos_df)}")
        
        if len(pos_df) < 10:
            print(f"⚠️ Too few {position} players for reliable model")
            return None, None
        
        # Base features for all positions
        feature_cols = [
            'weighted_historical_avg', 'consistency_score', 'performance_trend',
            'seasons_played', 'total_career_points', 'games_played'
        ]
        
        # Position-specific features
        if position == 'QB':
            # QB-specific features
            pos_specific = ['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds']
            feature_cols.extend([col for col in pos_specific if col in pos_df.columns])
            
        elif position == 'RB':
            # RB-specific features  
            pos_specific = ['carries', 'rushing_yards', 'rushing_tds', 'targets', 'receiving_yards']
            feature_cols.extend([col for col in pos_specific if col in pos_df.columns])
            
        elif position == 'WR':
            # WR-specific features
            pos_specific = ['targets', 'receiving_yards', 'receiving_tds', 'receptions']
            feature_cols.extend([col for col in pos_specific if col in pos_df.columns])
            
        elif position == 'TE':
            # TE-specific features
            pos_specific = ['targets', 'receiving_yards', 'receiving_tds', 'receptions']
            feature_cols.extend([col for col in pos_specific if col in pos_df.columns])
        
        # Filter to available features
        available_features = [col for col in feature_cols if col in pos_df.columns]
        print(f"Available features: {len(available_features)}")
        
        # Prepare X and y
        X = pos_df[available_features].fillna(0)
        y = pos_df['fantasy_points_target'].fillna(0)
        
        # Remove players with 0 target (insufficient data)
        valid_mask = y > 0
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Valid training samples: {len(X)}")
        
        if len(X) < 5:
            print(f"⚠️ Insufficient valid samples for {position}")
            return None, None
            
        return X, y
    
    def train_position_model(self, position, X, y):
        """Train CatBoost model for specific position"""
        print(f"\n🤖 TRAINING {position} MODEL")
        print("-" * 25)
        
        if X is None or y is None:
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train CatBoost model with position-specific parameters
        model_params = {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 6,
            'random_seed': 42,
            'verbose': False,
            'loss_function': 'RMSE'
        }
        
        # Adjust parameters by position
        if position == 'QB':
            model_params['depth'] = 8  # More complex for QB
        elif position in ['RB', 'WR']:
            model_params['depth'] = 7  # Medium complexity
        elif position == 'TE':
            model_params['depth'] = 5  # Simpler for smaller dataset
        
        model = CatBoostRegressor(**model_params)
        model.fit(X_train, y_train)
        
        # Validate model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        print(f"Test MAE: {test_mae:.2f}")
        
        # Performance assessment
        if test_r2 > 0.3:
            print(f"✅ Good {position} model performance")
        elif test_r2 > 0.1:
            print(f"⚠️ Modest {position} model performance")
        else:
            print(f"❌ Poor {position} model performance")
        
        # Store model with metadata
        model_data = {
            'model': model,
            'features': list(X.columns),
            'position': position,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'training_samples': len(X_train)
        }
        
        return model_data
    
    def build_all_position_models(self):
        """Build models for all positions"""
        print("\n🏗️ BUILDING POSITION-SPECIFIC MODELS")
        print("=" * 45)
        
        # Load data
        expanded_df = self.load_expanded_data()
        if expanded_df is None:
            print("❌ Cannot build models without data")
            return False
        
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            print(f"\n{'='*50}")
            
            # Prepare features
            X, y = self.prepare_position_features(expanded_df, position)
            
            # Train model
            model_data = self.train_position_model(position, X, y)
            
            if model_data:
                self.models[position] = model_data
                print(f"✅ {position} model ready")
            else:
                print(f"❌ Failed to create {position} model")
        
        return len(self.models) > 0
    
    def save_position_models(self):
        """Save all position models"""
        print("\n💾 SAVING POSITION-SPECIFIC MODELS")
        print("-" * 35)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for position, model_data in self.models.items():
            # Save model
            model_filename = f"models/position_{position.lower()}_model_{timestamp}.pkl"
            joblib.dump(model_data, model_filename)
            print(f"✅ Saved {position} model: {model_filename}")
            
            # Save metadata
            metadata = {
                'position': position,
                'features': model_data['features'],
                'train_r2': float(model_data['train_r2']),
                'test_r2': float(model_data['test_r2']),
                'test_mae': float(model_data['test_mae']),
                'training_samples': int(model_data['training_samples']),
                'timestamp': timestamp
            }
            
            metadata_filename = f"models/position_{position.lower()}_metadata_{timestamp}.json"
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save ensemble configuration
        ensemble_config = {
            'positions': list(self.models.keys()),
            'ensemble_weights': self.ensemble_weights,
            'timestamp': timestamp,
            'total_models': len(self.models)
        }
        
        ensemble_filename = f"models/ensemble_config_{timestamp}.json"
        with open(ensemble_filename, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        print(f"✅ Saved ensemble config: {ensemble_filename}")
        return timestamp
    
    def create_ensemble_predictor(self):
        """Create ensemble prediction system"""
        print("\n🔄 CREATING ENSEMBLE PREDICTION SYSTEM")
        print("-" * 40)
        
        ensemble_code = '''
def ensemble_predict(player_data, position, models, adp_rank=None):
    """
    Ensemble prediction combining multiple approaches
    """
    predictions = {}
    
    # 1. Position-specific CatBoost prediction (70% weight)
    if position in models and models[position]:
        try:
            model_data = models[position]
            model = model_data['model']
            features = model_data['features']
            
            # Prepare features for this player
            player_features = []
            for feature in features:
                player_features.append(player_data.get(feature, 0))
            
            catboost_pred = model.predict([player_features])[0]
            predictions['catboost'] = max(0, catboost_pred)  # No negative predictions
            
        except Exception as e:
            print(f"Warning: CatBoost prediction failed for {position}: {e}")
            predictions['catboost'] = 0
    else:
        predictions['catboost'] = 0
    
    # 2. ADP consensus (15% weight)
    if adp_rank and adp_rank > 0:
        # Convert ADP rank to point expectation (rough heuristic)
        if adp_rank <= 12:  # Round 1
            adp_points = 250 - (adp_rank * 15)
        elif adp_rank <= 24:  # Round 2
            adp_points = 200 - ((adp_rank - 12) * 10)
        elif adp_rank <= 36:  # Round 3
            adp_points = 150 - ((adp_rank - 24) * 8)
        else:
            adp_points = max(50, 120 - (adp_rank - 36) * 2)
        
        predictions['adp_consensus'] = max(0, adp_points)
    else:
        predictions['adp_consensus'] = 100  # Default neutral
    
    # 3. Volume projection (10% weight)
    volume_score = 0
    if position == 'QB':
        volume_score = player_data.get('passing_yards', 0) * 0.04 + player_data.get('passing_tds', 0) * 4
    elif position == 'RB':
        volume_score = player_data.get('carries', 0) * 0.1 + player_data.get('targets', 0) * 0.5
    elif position in ['WR', 'TE']:
        volume_score = player_data.get('targets', 0) * 0.8 + player_data.get('receiving_yards', 0) * 0.1
    
    predictions['volume_projection'] = max(0, volume_score)
    
    # 4. SOS adjustment (5% weight)
    sos_score = player_data.get('sos_score', 3.0)
    sos_adjustment = 100 * (sos_score / 3.0)  # Neutral is 3.0
    predictions['sos_adjustment'] = sos_adjustment
    
    # Combine with ensemble weights
    ensemble_weights = {
        'catboost': 0.70,
        'adp_consensus': 0.15,
        'volume_projection': 0.10,
        'sos_adjustment': 0.05
    }
    
    final_prediction = 0
    for component, weight in ensemble_weights.items():
        if component in predictions:
            final_prediction += predictions[component] * weight
    
    return max(0, final_prediction), predictions
'''
        
        # Save ensemble predictor code
        with open('scripts/ensemble_predictor.py', 'w') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('"""\nEnsemble Prediction System\n"""\n\n')
            f.write(ensemble_code)
        
        print("✅ Created ensemble prediction system")
        return True

def main():
    """Main execution function"""
    print("🏈 REBUILDING FANTASY FOOTBALL MODEL ARCHITECTURE")
    print("=" * 55)
    
    builder = PositionSpecificModelBuilder()
    
    # Build position-specific models
    if not builder.build_all_position_models():
        print("❌ Failed to build models")
        return False
    
    # Save models
    timestamp = builder.save_position_models()
    
    # Create ensemble system
    builder.create_ensemble_predictor()
    
    # Summary
    print(f"\n🎉 MODEL REBUILD COMPLETE!")
    print("=" * 30)
    print(f"✅ Built {len(builder.models)} position-specific models")
    print(f"✅ Models saved with timestamp: {timestamp}")
    print(f"✅ Ensemble prediction system created")
    
    # Performance summary
    print(f"\n📊 MODEL PERFORMANCE SUMMARY:")
    for position, model_data in builder.models.items():
        test_r2 = model_data['test_r2']
        test_mae = model_data['test_mae']
        samples = model_data['training_samples']
        print(f"  {position}: R²={test_r2:.3f}, MAE={test_mae:.1f}, Samples={samples}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Update draft_optimizer.py to use position-specific models")
    print(f"2. Run validation to confirm improvement from D to B+ grade") 
    print(f"3. Test ensemble predictions in live system")
    
    return True

if __name__ == "__main__":
    main() 