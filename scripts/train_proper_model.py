#!/usr/bin/env python3
"""
Train CatBoost Model on Proper Data (NO LEAKAGE)
Uses historical performance to predict future fantasy points
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime

class ProperCatBoostTrainer:
    """Train CatBoost model on proper leak-free data"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        
    def load_proper_training_data(self):
        """Load the proper training data with no leakage"""
        print("📊 LOADING PROPER TRAINING DATA (NO LEAKAGE)...")
        
        # Find the most recent proper training data file
        data_files = [
            f for f in os.listdir('data/')
            if f.startswith('proper_training_data_') and 'backup' not in f.lower()
        ]
        if not data_files:
            print("❌ No proper training data found! Run create_proper_training_data.py first")
            return None
        
        # Use the most recent file
        latest_file = max(data_files)
        filepath = f'data/{latest_file}'
        
        training_data = pd.read_csv(filepath)
        print(f"✅ Loaded {len(training_data)} training samples from {filepath}")
        
        # Show data quality
        print(f"📈 Data breakdown:")
        print(f"   - Positions: {training_data['position'].value_counts().to_dict()}")
        print(f"   - Seasons: {training_data['season'].value_counts().to_dict()}")
        print(f"   - Target mean: {training_data['target_fantasy_points'].mean():.2f}")
        print(f"   - Target std: {training_data['target_fantasy_points'].std():.2f}")
        
        return training_data
    
    def prepare_features(self, data):
        """Prepare features for training"""
        print("🔧 PREPARING FEATURES...")
        
        # Define feature columns (historical averages and trends)
        feature_cols = [
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
            'rate_receiving_td', 'rate_rushing_td', 'rate_pass_td_to_int'
        ]
        
        # Add position encoding
        data['is_qb'] = (data['position'] == 'QB').astype(int)
        data['is_rb'] = (data['position'] == 'RB').astype(int)
        data['is_wr'] = (data['position'] == 'WR').astype(int)
        data['is_te'] = (data['position'] == 'TE').astype(int)
        
        # Add season encoding
        data['season_2022'] = (data['season'] == 2022).astype(int)
        data['season_2023'] = (data['season'] == 2023).astype(int)
        data['season_2024'] = (data['season'] == 2024).astype(int)
        
        # Add week features
        data['week'] = data['week'].fillna(1)  # Handle any missing weeks
        data['early_season'] = (data['week'] <= 6).astype(int)
        data['mid_season'] = ((data['week'] > 6) & (data['week'] <= 12)).astype(int)
        data['late_season'] = (data['week'] > 12).astype(int)
        
        # Final feature list
        position_features = ['is_qb', 'is_rb', 'is_wr', 'is_te']
        season_features = ['season_2022', 'season_2023', 'season_2024']
        week_features = ['week', 'early_season', 'mid_season', 'late_season']
        
        self.feature_names = feature_cols + position_features + season_features + week_features
        
        # Fill any missing values
        for col in self.feature_names:
            if col in data.columns:
                data[col] = data[col].fillna(0)
        
        print(f"✅ Created {len(self.feature_names)} features")
        print(f"📋 Feature categories:")
        print(f"   - Historical stats: {len(feature_cols)} features")
        print(f"   - Position indicators: {len(position_features)} features") 
        print(f"   - Season indicators: {len(season_features)} features")
        print(f"   - Week features: {len(week_features)} features")
        
        return data
    
    def train_model(self, training_data):
        """Train the CatBoost model"""
        print("🤖 TRAINING CATBOOST MODEL ON PROPER DATA...")
        
        # Prepare features and target
        X = training_data[self.feature_names]
        y = training_data['target_fantasy_points']
        
        # Remove any rows with missing target values
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        print(f"📊 Training samples: {len(X)}")
        print(f"🎯 Target range: {y.min():.1f} to {y.max():.1f} fantasy points")
        print(f"📈 Target mean: {y.mean():.1f} fantasy points")
        
        # Split data (no stratification for regression)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"🔄 Train set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Train CatBoost model with proper hyperparameters
        self.model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=10,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            rsm=0.8,
            random_strength=1.0,
            od_type='Iter',
            od_wait=100,
            random_seed=42,
            verbose=False
        )
        
        print(f"🔄 Training CatBoost model...")
        self.model.fit(X_train, y_train)
        print(f"✅ Model training complete!")
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"📊 Performance metrics:")
        print(f"   - Train MAE: {train_mae:.2f} fantasy points")
        print(f"   - Test MAE: {test_mae:.2f} fantasy points")
        print(f"   - Train R²: {train_r2:.3f}")
        print(f"   - Test R²: {test_r2:.3f}")
        
        # Feature importance
        feature_importance = self.model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"   {row.name + 1}. {row['feature']}: {row['importance']:.1f}")
        
        metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': importance_df.to_dict('records')
        }
        
        return metrics
    
    def save_model(self, metrics):
        """Save the trained model"""
        print("💾 SAVING TRAINED MODEL...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = 'models/proper_fantasy_model.pkl'
        metadata_path = f'models/proper_model_metadata_{timestamp}.json'
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model_type': 'CatBoost',
            'training_samples': len(self.feature_names),
            'features': self.feature_names,
            'metrics': metrics,
            'target': 'future_fantasy_points_from_historical_avg',
            'no_data_leakage': True,
            'description': 'Predicts future fantasy performance from historical averages'
        }
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved model to {model_path}")
        print(f"✅ Saved metadata to {metadata_path}")
        
        return model_path, metadata_path
    
    def train_complete_model(self):
        """Complete model training pipeline"""
        print("🚀 PROPER CATBOOST MODEL TRAINING (NO LEAKAGE)")
        print("=" * 60)
        
        # Step 1: Load proper training data
        training_data = self.load_proper_training_data()
        if training_data is None:
            return False
        
        # Step 2: Prepare features
        training_data = self.prepare_features(training_data)
        
        # Step 3: Train model
        metrics = self.train_model(training_data)
        
        # Step 4: Save model
        model_path, metadata_path = self.save_model(metrics)
        
        print(f"\n�� PROPER MODEL TRAINING COMPLETE!")
        print(f"✅ Model trained on leak-free data")
        print(f"✅ Test accuracy: {metrics['test_r2']:.3f} R² score")
        print(f"✅ Test error: {metrics['test_mae']:.2f} fantasy points MAE")
        print(f"📁 Model saved: {model_path}")
        
        # Interpret results
        if metrics['test_r2'] > 0.6:
            print(f"🎯 Excellent prediction accuracy!")
        elif metrics['test_r2'] > 0.3:
            print(f"🎯 Good prediction accuracy for sports data!")
        elif metrics['test_r2'] > 0.1:
            print(f"🎯 Moderate prediction accuracy - typical for fantasy sports!")
        else:
            print(f"⚠️  Low prediction accuracy - may need more features or data")
        
        return True

if __name__ == "__main__":
    trainer = ProperCatBoostTrainer()
    success = trainer.train_complete_model()
    
    if success:
        print(f"\n🚀 Ready to use leak-free model in fantasy draft system!")
    else:
        print(f"\n💥 Model training failed, check errors above") 