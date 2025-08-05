#!/usr/bin/env python3
"""
Retrain CatBoost Model with Fixed Comprehensive Data
Uses the new multi-year, properly matched dataset for enhanced predictions
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime
import warnings
import glob
warnings.filterwarnings('ignore')

def retrain_fixed_model():
    """Retrain model with the fixed comprehensive multi-year dataset"""
    
    print("🤖 RETRAINING CATBOOST MODEL WITH FIXED COMPREHENSIVE DATA")
    print("=" * 65)
    
    # 1. LOAD THE FIXED COMPREHENSIVE TRAINING DATA
    print("📊 Loading fixed comprehensive training dataset...")
    
    # Find the most recent fixed dataset
    fixed_files = glob.glob('comprehensive_training_data_FIXED_*.csv')
    if not fixed_files:
        print("❌ No fixed comprehensive training data found!")
        return
    
    latest_file = max(fixed_files)
    print(f"📂 Using: {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        print(f"✅ Loaded {len(df)} players with enhanced multi-year features")
        print(f"   Positions: {dict(df['position'].value_counts())}")
        print(f"   Seasons: {dict(df['season'].value_counts())}")
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return
    
    # 2. PREPARE FEATURES AND TARGET
    print("\n🔧 Preparing features and target variable...")
    
    # Define target variable
    target_col = 'fantasy_points_ppr'
    
    # Select feature columns (exclude non-predictive columns)
    exclude_cols = {
        'name', 'canonical_name', 'clean_name', 'player_key', 'team', 
        target_col, 'season'  # Keep season info but don't use as feature
    }
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Ensure we have the required features
    required_features = [
        'position', 'games', 'targets', 'carries', 'receiving_yards', 'rushing_yards',
        'receptions', 'receiving_tds', 'rushing_tds', 'avg_fantasy_points_3yr',
        'performance_trend_pct', 'consistency_score', 'seasons_played'
    ]
    
    available_features = [f for f in required_features if f in df.columns]
    additional_features = [f for f in feature_cols if f not in required_features and f in df.columns]
    
    final_features = available_features + additional_features
    
    print(f"🎯 Using {len(final_features)} features:")
    print(f"   Core features: {len(available_features)}")
    print(f"   Additional features: {len(additional_features)}")
    
    # 3. PREPARE DATA FOR TRAINING
    X = df[final_features].copy()
    y = df[target_col].copy()
    
    # Handle categorical variables
    categorical_features = ['position']
    for cat_feature in categorical_features:
        if cat_feature in X.columns:
            X[cat_feature] = X[cat_feature].astype('category')
    
    # Handle missing values
    X = X.fillna(0)
    
    # Remove any invalid target values
    valid_mask = (y >= 0) & (y.notna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"✅ Final training data: {len(X)} samples with {len(final_features)} features")
    print(f"   Target range: {y.min():.1f} to {y.max():.1f} fantasy points")
    
    # 4. SPLIT DATA
    print("\n📊 Splitting data for training and validation...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X['position']
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # 5. TRAIN THE MODEL
    print("\n🤖 Training CatBoost model...")
    
    # Enhanced CatBoost parameters for better performance
    model = CatBoostRegressor(
        iterations=500,  # More iterations for better learning
        depth=8,         # Deeper trees for complex patterns
        learning_rate=0.1,
        l2_leaf_reg=3,
        random_state=42,
        cat_features=categorical_features,
        verbose=100,     # Show progress
        eval_metric='RMSE',
        early_stopping_rounds=50
    )
    
    # Train with validation set for early stopping
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        plot=False,
        verbose_eval=100
    )
    
    # 6. EVALUATE THE MODEL
    print("\n📈 Evaluating model performance...")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"🎯 Training Performance:")
    print(f"   R² Score: {train_r2:.4f}")
    print(f"   RMSE: {train_rmse:.2f}")
    print(f"   MAE: {train_mae:.2f}")
    
    print(f"✅ Validation Performance:")
    print(f"   R² Score: {test_r2:.4f}")
    print(f"   RMSE: {test_rmse:.2f}")
    print(f"   MAE: {test_mae:.2f}")
    
    # 7. FEATURE IMPORTANCE
    print(f"\n🔍 Top 10 Most Important Features:")
    feature_importance = model.get_feature_importance()
    feature_names = X.columns
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<25} ({row['importance']:.1f})")
    
    # 8. SAVE THE MODEL
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'models/draft_model_FIXED_{timestamp}.pkl'
    metadata_filename = f'models/model_metadata_FIXED_{timestamp}.json'
    
    # Save model
    joblib.dump(model, model_filename)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'features': list(final_features),
        'categorical_features': categorical_features,
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'target_range': [float(y.min()), float(y.max())],
        'position_distribution': {k: int(v) for k, v in X['position'].value_counts().items()},
        'season_distribution': {k: int(v) for k, v in df['season'].value_counts().items()},
        'data_source': latest_file,
        'model_version': 'FIXED_COMPREHENSIVE'
    }
    
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Model saved:")
    print(f"   Model: {model_filename}")
    print(f"   Metadata: {metadata_filename}")
    
    # 9. SAMPLE PREDICTIONS
    print(f"\n🎯 Sample Predictions on Test Set:")
    print("=" * 50)
    
    # Show predictions for different positions
    test_with_pred = X_test.copy()
    test_with_pred['actual'] = y_test
    test_with_pred['predicted'] = y_pred_test
    test_with_pred['name'] = df.loc[y_test.index, 'canonical_name']
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_data = test_with_pred[test_with_pred['position'] == position].head(3)
        if len(pos_data) > 0:
            print(f"\n{position} Predictions:")
            for _, row in pos_data.iterrows():
                print(f"   {row['name']:<20} | Actual: {row['actual']:.1f} | Predicted: {row['predicted']:.1f}")
    
    return model_filename, metadata_filename

if __name__ == "__main__":
    retrain_fixed_model() 