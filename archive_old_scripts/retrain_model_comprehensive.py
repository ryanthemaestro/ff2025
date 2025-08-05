#!/usr/bin/env python3
"""
Retrain CatBoost Model with Comprehensive Data
Uses the new multi-year, feature-rich dataset for better predictions
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
warnings.filterwarnings('ignore')

def retrain_comprehensive_model():
    """Retrain model with comprehensive multi-year dataset"""
    
    print("🤖 RETRAINING CATBOOST MODEL WITH COMPREHENSIVE DATA")
    print("=" * 60)
    
    # 1. LOAD THE COMPREHENSIVE TRAINING DATA
    print("📊 Loading comprehensive training dataset...")
    
    # Find the most recent comprehensive training data file
    import glob
    training_files = glob.glob('comprehensive_training_data_*.csv')
    if not training_files:
        print("❌ No comprehensive training data found!")
        print("   Please run create_comprehensive_training_data.py first")
        return False
    
    latest_file = max(training_files)
    print(f"   Using: {latest_file}")
    
    df = pd.read_csv(latest_file)
    print(f"✅ Loaded {len(df)} players with {len(df.columns)} features")
    print(f"   Positions: {dict(df['position'].value_counts())}")
    
    # 2. PREPARE FEATURES FOR TRAINING
    print("\n🔧 Preparing features for training...")
    
    # Target variable
    target = 'fantasy_points_ppr'
    
    # Select meaningful features for training
    feature_columns = [
        # Basic stats
        'games', 'targets', 'carries', 'receptions',
        'receiving_yards', 'rushing_yards', 'receiving_tds', 'rushing_tds',
        'passing_yards', 'passing_tds', 'interceptions',
        
        # Multi-year features (the magic!)
        'avg_fantasy_points_3yr', 'avg_targets_3yr', 'avg_carries_3yr',
        'performance_trend_pct', 'consistency_score', 'seasons_played',
        'total_games', 'avg_opportunity',
        
        # Efficiency metrics
        'points_per_game', 'opportunity_share',
        
        # Position-specific features (if available)
        'catch_rate', 'yards_per_target', 'total_touches', 'yards_per_touch',
        'passing_efficiency'
    ]
    
    # Only use features that exist in the dataset
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"✅ Using {len(available_features)} features for training")
    print(f"   Features: {available_features}")
    
    # Prepare training data
    X = df[available_features].fillna(0)
    y = df[target]
    
    # Add position as categorical feature
    if 'position' in df.columns:
        X['position'] = df['position']
        categorical_features = ['position']
    else:
        categorical_features = []
    
    print(f"   Target range: {y.min():.1f} - {y.max():.1f} fantasy points")
    print(f"   Training examples: {len(X)}")
    
    # 3. TRAIN-TEST SPLIT
    print("\n📋 Splitting data for training and validation...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['position']
    )
    
    print(f"✅ Training set: {len(X_train)} players")
    print(f"✅ Test set: {len(X_test)} players")
    
    # 4. TRAIN CATBOOST MODEL
    print("\n🏋️ Training CatBoost model...")
    
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False,
        cat_features=categorical_features,
        loss_function='RMSE',
        eval_metric='RMSE'
    )
    
    # Train with validation
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=100,
        use_best_model=True,
        verbose=50
    )
    
    print(f"✅ Model trained successfully!")
    print(f"   Best iteration: {model.best_iteration_}")
    print(f"   Best score: {model.best_score_['validation']['RMSE']:.2f}")
    
    # 5. EVALUATE MODEL PERFORMANCE
    print("\n📈 Evaluating model performance...")
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_rmse = mean_squared_error(y_train, y_pred_train) ** 0.5
    test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"📊 TRAINING METRICS:")
    print(f"   RMSE: {train_rmse:.2f}")
    print(f"   MAE:  {train_mae:.2f}")
    print(f"   R²:   {train_r2:.3f}")
    
    print(f"\n📊 VALIDATION METRICS:")
    print(f"   RMSE: {test_rmse:.2f}")
    print(f"   MAE:  {test_mae:.2f}")
    print(f"   R²:   {test_r2:.3f}")
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<25} ({row['importance']:.1f})")
    
    # 6. SAVE THE TRAINED MODEL
    print("\n💾 Saving trained model...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = f'models/catboost_comprehensive_{timestamp}.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'CatBoost',
        'training_timestamp': timestamp,
        'training_data_file': latest_file,
        'num_training_examples': len(X_train),
        'num_test_examples': len(X_test),
        'features': list(feature_names),
        'categorical_features': categorical_features,
        'target_variable': target,
        'performance_metrics': {
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2)
        },
        'feature_importance': {
            feat: float(imp) for feat, imp in zip(feature_names, feature_importance)
        },
        'position_distribution': df['position'].value_counts().to_dict(),
        'fantasy_points_range': {
            'min': float(y.min()),
            'max': float(y.max()),
            'mean': float(y.mean()),
            'std': float(y.std())
        }
    }
    
    metadata_path = f'models/model_metadata_comprehensive_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {metadata_path}")
    
    # Update the default model files
    try:
        joblib.dump(model, 'models/draft_model.pkl')
        with open('models/model_metadata_nflverse.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Updated default model files")
    except Exception as e:
        print(f"⚠️ Could not update default files: {e}")
    
    # 7. PERFORMANCE ANALYSIS BY POSITION
    print(f"\n📊 PERFORMANCE BY POSITION:")
    print("-" * 40)
    
    # Add predictions to test set for analysis
    test_df = X_test.copy()
    test_df['actual'] = y_test
    test_df['predicted'] = y_pred_test
    test_df['error'] = abs(test_df['actual'] - test_df['predicted'])
    
    for pos in sorted(test_df['position'].unique()):
        pos_data = test_df[test_df['position'] == pos]
        if len(pos_data) > 0:
            pos_rmse = mean_squared_error(pos_data['actual'], pos_data['predicted']) ** 0.5
            pos_mae = mean_absolute_error(pos_data['actual'], pos_data['predicted'])
            print(f"{pos:>3}: {len(pos_data):2d} players | RMSE: {pos_rmse:5.1f} | MAE: {pos_mae:5.1f}")
    
    print(f"\n🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"   📈 This model has REAL multi-year features")
    print(f"   📈 Performance trends and consistency scores")
    print(f"   📈 Position-specific efficiency metrics")
    print(f"   📈 {len(available_features)} rich features vs basic stats")
    print(f"   📈 Ready to provide much better predictions!")
    
    return True, model_path, metadata_path

if __name__ == "__main__":
    try:
        success, model_path, metadata_path = retrain_comprehensive_model()
        if success:
            print(f"\n🚀 SUCCESS! Your fantasy draft AI just got MUCH smarter!")
            print(f"   Model: {model_path}")
            print(f"   Metadata: {metadata_path}")
            print(f"\n   Restart your draft UI to use the new model!")
        else:
            print(f"\n❌ Training failed")
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc() 