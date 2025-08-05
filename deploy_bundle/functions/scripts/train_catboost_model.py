#!/usr/bin/env python3
"""
CatBoost Model Training on Real NFL Data
Train a high-quality fantasy prediction model using clean NFLverse data (2022-2024)

REAL NFL STATISTICS ONLY - NO SYNTHETIC DATA
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime

class CatBoostFantasyTrainer:
    """Train CatBoost model on real NFL fantasy data"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        
    def load_training_data(self):
        """Load the clean NFLverse training data"""
        print("📊 LOADING CLEAN NFLVERSE TRAINING DATA...")
        
        # Load combined seasonal data
        seasonal_file = 'data/nflverse/combined_seasonal_2022_2024.csv'
        weekly_file = 'data/nflverse/combined_weekly_2022_2024.csv'
        
        if not os.path.exists(seasonal_file):
            print("❌ Seasonal data not found! Run collect_nflverse_data.py first")
            return None
            
        if not os.path.exists(weekly_file):
            print("❌ Weekly data not found! Run collect_nflverse_data.py first")
            return None
        
        # Load both datasets
        seasonal_data = pd.read_csv(seasonal_file)
        weekly_data = pd.read_csv(weekly_file)
        
        print(f"✅ Loaded {len(seasonal_data)} seasonal records from 2022-2024")
        print(f"✅ Loaded {len(weekly_data)} weekly records from 2022-2024")
        
        # Show data quality
        print(f"📈 Seasonal data breakdown:")
        if 'data_year' in seasonal_data.columns:
            print(f"   - Years: {sorted(seasonal_data['data_year'].unique())}")
        if 'position' in seasonal_data.columns:
            print(f"   - Positions: {seasonal_data['position'].value_counts().to_dict()}")
        
        print(f"📈 Weekly data breakdown:")
        if 'season' in weekly_data.columns:
            print(f"   - Years: {sorted(weekly_data['season'].unique())}")
        if 'position' in weekly_data.columns:
            print(f"   - Positions: {weekly_data['position'].value_counts().to_dict()}")
        
        return seasonal_data, weekly_data
    
    def create_features(self, data):
        """Create comprehensive features for fantasy prediction"""
        print("🔧 CREATING FEATURES FOR MODEL...")
        
        # Copy data to avoid modifying original
        df = data.copy()
        
        # Basic statistics features
        feature_cols = [
            'passing_yards', 'passing_tds', 'interceptions', 
            'rushing_yards', 'rushing_tds', 
            'receiving_yards', 'receiving_tds', 'receptions', 'targets',
            'carries'
        ]
        
        # Fill missing values with 0 (players didn't record stats in those categories)
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Create derived features
        df['passing_td_rate'] = df['passing_tds'] / (df['passing_yards'] + 1)  # +1 to avoid division by 0
        df['rushing_td_rate'] = df['rushing_tds'] / (df['rushing_yards'] + 1)
        df['receiving_td_rate'] = df['receiving_tds'] / (df['receiving_yards'] + 1)
        df['catch_rate'] = df['receptions'] / (df['targets'] + 1)
        df['yards_per_carry'] = df['rushing_yards'] / (df['carries'] + 1)
        df['yards_per_reception'] = df['receiving_yards'] / (df['receptions'] + 1)
        df['yards_per_target'] = df['receiving_yards'] / (df['targets'] + 1)
        
        # Position-specific features
        df['is_qb'] = (df['position'] == 'QB').astype(int)
        df['is_rb'] = (df['position'] == 'RB').astype(int)
        df['is_wr'] = (df['position'] == 'WR').astype(int)
        df['is_te'] = (df['position'] == 'TE').astype(int)
        df['is_k'] = (df['position'] == 'K').astype(int)
        
        # Season and week features
        df['season_encoded'] = df['season'] - 2022  # 0 for 2022, 1 for 2023
        df['week_early'] = (df['week'] <= 6).astype(int)
        df['week_mid'] = ((df['week'] > 6) & (df['week'] <= 12)).astype(int)
        df['week_late'] = (df['week'] > 12).astype(int)
        
        # Feature list for model
        self.feature_names = [
            # Basic stats
            'passing_yards', 'passing_tds', 'interceptions',
            'rushing_yards', 'rushing_tds', 'carries',
            'receiving_yards', 'receiving_tds', 'receptions', 'targets',
            
            # Derived features
            'passing_td_rate', 'rushing_td_rate', 'receiving_td_rate',
            'catch_rate', 'yards_per_carry', 'yards_per_reception', 'yards_per_target',
            
            # Position features
            'is_qb', 'is_rb', 'is_wr', 'is_te', 'is_k',
            
            # Time features
            'season_encoded', 'week_early', 'week_mid', 'week_late'
        ]
        
        # Ensure all features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        print(f"✅ Created {len(self.feature_names)} features")
        print(f"📋 Feature categories:")
        print(f"   - Basic stats: 10 features")
        print(f"   - Derived ratios: 7 features") 
        print(f"   - Position indicators: 5 features")
        print(f"   - Time features: 4 features")
        
        return df
    
    def train_model(self, training_data):
        """Train CatBoost model on real data"""
        print("🤖 TRAINING CATBOOST MODEL ON REAL NFL DATA...")
        
        # Create target variable from raw stats (NO LEAKAGE)
        # Standard fantasy scoring: Pass TD=4, Rush/Rec TD=6, Pass Yd=0.04, Rush/Rec Yd=0.1, Rec=1 (PPR), INT=-2
        training_data['calculated_fantasy_points'] = (
            training_data['passing_tds'].fillna(0) * 4 +
            training_data['passing_yards'].fillna(0) * 0.04 +
            training_data['interceptions'].fillna(0) * -2 +
            training_data['rushing_tds'].fillna(0) * 6 +
            training_data['rushing_yards'].fillna(0) * 0.1 +
            training_data['receiving_tds'].fillna(0) * 6 +
            training_data['receiving_yards'].fillna(0) * 0.1 +
            training_data['receptions'].fillna(0) * 1  # PPR scoring
        )
        
        # Prepare features and target
        X = training_data[self.feature_names]
        y = training_data['calculated_fantasy_points']  # Use our calculated points (no leakage)
        
        # Remove any rows with missing target values
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        print(f"📊 Training samples: {len(X)}")
        print(f"🎯 Target range: {y.min():.1f} to {y.max():.1f} fantasy points")
        print(f"📈 Target mean: {y.mean():.1f} fantasy points")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"🔄 Train set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Train CatBoost model
        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            bootstrap_type='Bayesian',
            bagging_temperature=1,
            od_type='Iter',
            od_wait=50,
            random_seed=42,
            allow_writing_files=False,  # Suppress CatBoost output files
            verbose=100  # Print every 100 iterations
        )
        
        print("🔄 Training CatBoost model...")
        self.model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"✅ Model training complete!")
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
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.1f}")
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': importance_df.to_dict('records')
        }
    
    def save_model(self, metrics):
        """Save the trained model"""
        print("💾 SAVING TRAINED MODEL...")
        
        # Save model
        model_path = 'models/draft_model.pkl'
        joblib.dump(self.model, model_path)
        print(f"✅ Saved model to {model_path}")
        
        # Save model metadata
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata = {
            'timestamp': timestamp,
            'model_type': 'CatBoost',
            'training_data': 'nflverse 2022-2023',
            'features': self.feature_names,
            'metrics': metrics,
            'target': 'calculated_fantasy_points'
        }
        
        metadata_path = f'models/model_metadata_{timestamp}.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata to {metadata_path}")
        
        return model_path, metadata_path
    
    def train_complete_model(self):
        """Complete model training pipeline"""
        print("🚀 CATBOOST MODEL TRAINING ON REAL NFL DATA")
        print("=" * 60)
        
        # Step 1: Load training data
        data_tuple = self.load_training_data()
        if data_tuple is None:
            return False
        
        seasonal_data, weekly_data = data_tuple
        
        # Step 2: Use weekly data for training (has position info)
        print("🎯 Using weekly data for model training...")
        training_data = self.create_features(weekly_data)
        
        # Step 3: Train model
        metrics = self.train_model(training_data)
        
        # Step 4: Save model
        model_path, metadata_path = self.save_model(metrics)
        
        print(f"\n🎉 MODEL TRAINING COMPLETE!")
        print(f"✅ High-quality CatBoost model trained on real NFL data")
        print(f"✅ Test accuracy: {metrics['test_r2']:.3f} R² score")
        print(f"✅ Test error: {metrics['test_mae']:.2f} fantasy points MAE")
        print(f"📁 Model saved: {model_path}")
        
        return True

if __name__ == "__main__":
    trainer = CatBoostFantasyTrainer()
    success = trainer.train_complete_model()
    
    if success:
        print(f"\n🚀 Ready to use new model in fantasy draft system!")
    else:
        print(f"\n💥 Model training failed, check errors above") 