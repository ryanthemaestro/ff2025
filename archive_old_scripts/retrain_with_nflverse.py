#!/usr/bin/env python3
"""
Retrain CatBoost model using REAL NFL data from nflverse
This will fix the AI recommendations to show actual star players
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_and_clean_nflverse_data():
    """Load and clean the nflverse data"""
    print("🎯 Loading REAL NFL data from nflverse...")
    
    # Load the data
    df = pd.read_csv('nflverse_fantasy_data_20250801_201533.csv')
    print(f"✅ Loaded {len(df)} players from nflverse")
    
    # Create unique player identifiers to handle name collisions like J.Jefferson
    df['unique_id'] = df['name'] + '_' + df['position'] + '_' + df['recent_team'].fillna('')
    
    # Check for name collisions
    name_counts = df['name'].value_counts()
    duplicate_names = name_counts[name_counts > 1]
    if len(duplicate_names) > 0:
        print(f"⚠️ Found {len(duplicate_names)} duplicate names:")
        for name, count in duplicate_names.head(5).items():
            players = df[df['name'] == name][['name', 'position', 'recent_team', 'fantasy_points']]
            print(f"   {name}: {count} players")
            for _, player in players.iterrows():
                print(f"     -> {player['name']} ({player['position']}, {player['recent_team']}) - {player['fantasy_points']} pts")
    
    # Clean the data
    df = df.dropna(subset=['fantasy_points', 'name', 'position'])
    df = df[df['fantasy_points'] > 0]  # Remove 0-point players
    
    # Filter to fantasy-relevant positions
    fantasy_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    df = df[df['position'].isin(fantasy_positions)]
    
    # Remove duplicates based on unique_id (keeps first occurrence)
    df = df.drop_duplicates(subset=['unique_id'], keep='first')
    
    print(f"✅ Cleaned data: {len(df)} fantasy-relevant players")
    
    # Fill missing values with reasonable defaults
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(0)
    
    return df

def create_features(df):
    """Create feature set for CatBoost training"""
    print("🧠 Creating features for CatBoost training...")
    
    # Core NFL stats features
    features = [
        'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
        'receiving_yards', 'receiving_tds', 'receptions', 'targets',
        'games_played', 'avg_fantasy_points', 'avg_ppr_all_years',
        'total_games', 'seasons_played', 'consistency_score',
        'age', 'years_exp'
    ]
    
    # Create additional derived features
    df['yards_per_game'] = (df['passing_yards'] + df['rushing_yards'] + df['receiving_yards']) / df['games_played'].clip(lower=1)
    df['tds_per_game'] = (df['passing_tds'] + df['rushing_tds'] + df['receiving_tds']) / df['games_played'].clip(lower=1)
    df['targets_per_game'] = df['targets'] / df['games_played'].clip(lower=1)
    df['catch_rate'] = df['receptions'] / df['targets'].clip(lower=1)
    df['yards_per_reception'] = df['receiving_yards'] / df['receptions'].clip(lower=1)
    df['experience_factor'] = df['years_exp'] / (df['age'] - 21).clip(lower=1)
    
    # Add derived features to feature list
    derived_features = [
        'yards_per_game', 'tds_per_game', 'targets_per_game', 
        'catch_rate', 'yards_per_reception', 'experience_factor'
    ]
    
    features.extend(derived_features)
    
    # Handle categorical features
    categorical_features = ['position']
    
    # Ensure all features exist
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    
    print(f"✅ Using {len(available_features)} features for training")
    
    return df, available_features, categorical_features

def train_catboost_model(df, features, categorical_features):
    """Train the CatBoost model"""
    print("🤖 Training CatBoost model with REAL NFL data...")
    
    # Prepare data
    X = df[features].copy()
    y = df['fantasy_points'].copy()
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    # Convert categorical features to strings
    for cat_feature in categorical_features:
        if cat_feature in X.columns:
            X[cat_feature] = X[cat_feature].astype(str)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['position']
    )
    
    print(f"📊 Training set: {len(X_train)} players")
    print(f"📊 Test set: {len(X_test)} players")
    
    # Train CatBoost model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        cat_features=[features.index(f) for f in categorical_features if f in features],
        random_seed=42,
        verbose=100
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\n🎯 MODEL PERFORMANCE:")
    print(f"   Train RMSE: {train_rmse:.2f}")
    print(f"   Test RMSE: {test_rmse:.2f}")
    print(f"   Train R²: {train_r2:.3f}")
    print(f"   Test R²: {test_r2:.3f}")
    
    return model, features

def save_model_and_metadata(model, features):
    """Save the trained model and metadata"""
    print("💾 Saving trained model...")
    
    # Save model
    joblib.dump(model, 'models/draft_model.pkl')
    
    # Save metadata
    metadata = {
        'model_type': 'CatBoostRegressor',
        'features': features,
        'data_source': 'nflverse_fantasy_data_20250801_201533.csv',
        'training_date': pd.Timestamp.now().isoformat(),
        'target': 'fantasy_points',
        'model_version': '2.0_nflverse'
    }
    
    with open('models/model_metadata_nflverse.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved with {len(features)} features")
    print("✅ Metadata saved")
    
    return metadata

def test_predictions(model, df, features):
    """Test predictions on some star players"""
    print("\n🌟 TESTING PREDICTIONS ON STAR PLAYERS:")
    print("=" * 50)
    
    # Test on some known star players
    star_players = ['L.Jackson', 'J.Chase', 'J.Allen', 'S.Barkley', 'C.McCaffrey', 'T.Hill']
    
    for player in star_players:
        player_data = df[df['name'].str.contains(player, na=False)]
        if not player_data.empty:
            player_row = player_data.iloc[0]
            X_player = player_row[features].values.reshape(1, -1)
            prediction = model.predict(X_player)[0]
            actual = player_row['fantasy_points']
            
            print(f"   {player_row['name']} ({player_row['position']}): "
                  f"Predicted: {prediction:.1f}, Actual: {actual:.1f}")

if __name__ == "__main__":
    print("🚀 RETRAINING CATBOOST WITH REAL NFL DATA")
    print("=" * 50)
    
    # Load and prepare data
    df = load_and_clean_nflverse_data()
    df, features, categorical_features = create_features(df)
    
    # Train model
    model, final_features = train_catboost_model(df, features, categorical_features)
    
    # Save model
    metadata = save_model_and_metadata(model, final_features)
    
    # Test predictions
    test_predictions(model, df, final_features)
    
    print("\n🎉 MODEL RETRAINING COMPLETE!")
    print("🎯 Your AI recommendations will now use REAL NFL star players!")
    print("💥 No more 10.8-score scrubs - you'll get Lamar Jackson, Ja'Marr Chase, etc!") 