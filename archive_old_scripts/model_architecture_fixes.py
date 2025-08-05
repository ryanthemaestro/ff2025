#!/usr/bin/env python3
"""
Model Architecture Fixes
Address the fundamental issues identified by real-world validation
"""

import pandas as pd
import numpy as np
from datetime import datetime

def diagnose_model_issues():
    """Diagnose the core issues with our current model"""
    print("🔧 MODEL ARCHITECTURE DIAGNOSIS & FIXES")
    print("=" * 50)
    
    print("\n❌ IDENTIFIED PROBLEMS:")
    print("1. Tiny training dataset (100 players vs 1,081 actual)")
    print("2. Poor name matching between training and real data")
    print("3. Over-aggressive recency weighting creating noise")
    print("4. Training on wrong target (weighted points vs actual performance)")
    print("5. No position-specific modeling")
    
    print("\n✅ PROPOSED FIXES:")
    print("1. Expand training dataset to include ALL available players")
    print("2. Improve name normalization and matching")
    print("3. Use moderate recency weighting (2x vs 4x)")
    print("4. Train directly on 2024 actual fantasy points")
    print("5. Build position-specific models")
    print("6. Add ensemble modeling with multiple approaches")

def create_expanded_training_dataset():
    """Create a much larger, cleaner training dataset"""
    print("\n📊 CREATING EXPANDED TRAINING DATASET")
    print("-" * 40)
    
    try:
        # Load all available data sources
        actual_2024 = pd.read_csv('data/fantasy_metrics_2024.csv')
        print(f"✅ Actual 2024 data: {len(actual_2024)} players")
        
        try:
            historical_2022_2023 = pd.read_csv('data/historical/real_nfl_historical_2022_2023.csv')
            print(f"✅ Historical 2022-2023: {len(historical_2022_2023)} players")
        except:
            historical_2022_2023 = pd.DataFrame()
            print("⚠️ No historical data found")
        
        # Combine datasets with proper target variable
        if not historical_2022_2023.empty:
            all_data = pd.concat([actual_2024, historical_2022_2023], ignore_index=True)
        else:
            all_data = actual_2024.copy()
        
        print(f"📋 Combined dataset: {len(all_data)} total records")
        
        # Create proper features with moderate recency weighting
        expanded_data = create_proper_features(all_data)
        
        # Save expanded dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"expanded_training_data_{timestamp}.csv"
        expanded_data.to_csv(filename, index=False)
        
        print(f"💾 Saved expanded dataset: {filename}")
        print(f"📊 Final dataset size: {len(expanded_data)} players")
        
        return expanded_data
        
    except Exception as e:
        print(f"❌ Error creating expanded dataset: {e}")
        return None

def create_proper_features(df):
    """Create proper features with moderate recency weighting"""
    print("\n🔧 CREATING PROPER FEATURES")
    print("-" * 30)
    
    # Ensure we have required columns
    required_cols = ['name', 'position', 'fantasy_points_ppr', 'season']
    
    # Add season if missing (assume 2024 for recent data)
    if 'season' not in df.columns:
        df['season'] = 2024
    
    # Group by player and calculate multi-year stats with MODERATE recency weighting
    player_stats = []
    
    for name, group in df.groupby('name'):
        if len(group) == 0:
            continue
            
        # Sort by season
        group = group.sort_values('season')
        
        # Calculate weighted averages (2x for most recent, not 4x)
        weights = []
        fantasy_points = []
        
        for _, row in group.iterrows():
            season = row.get('season', 2024)
            points = row.get('fantasy_points_ppr', 0)
            
            if season == 2024:
                weight = 2.0  # Moderate 2x weighting for recent data
            elif season == 2023:
                weight = 1.5  # Mild boost for 2023
            else:
                weight = 1.0  # Baseline for older data
                
            weights.append(weight)
            fantasy_points.append(points)
        
        if weights and fantasy_points:
            # Calculate weighted average
            weighted_avg = np.average(fantasy_points, weights=weights)
            
            # Calculate other features
            consistency = 1 / (1 + np.std(fantasy_points)) if len(fantasy_points) > 1 else 0.5
            trend = (fantasy_points[-1] - fantasy_points[0]) / (fantasy_points[0] + 1) if len(fantasy_points) > 1 else 0
            
            player_stats.append({
                'name': name,
                'position': group.iloc[-1]['position'],  # Use most recent position
                'fantasy_points_target': group.iloc[-1].get('fantasy_points_ppr', 0),  # 2024 actual as target
                'weighted_historical_avg': weighted_avg,
                'consistency_score': consistency,
                'performance_trend': trend,
                'seasons_played': len(group),
                'total_career_points': sum(fantasy_points),
                'games_played': group.iloc[-1].get('games', 16),
                'targets': group.iloc[-1].get('targets', 0),
                'carries': group.iloc[-1].get('carries', 0)
            })
    
    result_df = pd.DataFrame(player_stats)
    print(f"✅ Created features for {len(result_df)} players")
    
    return result_df

def create_position_specific_models():
    """Create separate models for each position"""
    print("\n🎯 POSITION-SPECIFIC MODEL STRATEGY")
    print("-" * 35)
    
    print("Strategy: Train separate CatBoost models for:")
    print("• QB Model: Focus on passing stats, rushing upside")
    print("• RB Model: Focus on touches, goal line work")  
    print("• WR Model: Focus on targets, air yards")
    print("• TE Model: Focus on red zone usage, consistency")
    
    print("\nThis should dramatically improve predictions vs one-size-fits-all model")

def create_ensemble_approach():
    """Design ensemble modeling approach"""
    print("\n🔄 ENSEMBLE MODELING APPROACH")
    print("-" * 30)
    
    print("Combine multiple prediction approaches:")
    print("1. Position-specific CatBoost models (70% weight)")
    print("2. ADP consensus ranking (15% weight)")
    print("3. Volume-based projections (10% weight)")
    print("4. Strength of schedule adjustments (5% weight)")
    
    print("\nThis reduces model risk and improves robustness")

def model_architecture_recommendations():
    """Provide concrete next steps"""
    print("\n📋 IMMEDIATE ACTION ITEMS")
    print("=" * 30)
    
    print("🎯 Priority 1 - Data Quality:")
    print("  • Expand training data from 100 to 1000+ players")
    print("  • Fix name matching between datasets")
    print("  • Use actual 2024 fantasy points as target variable")
    
    print("\n🎯 Priority 2 - Model Architecture:")
    print("  • Build 4 separate position-specific models")
    print("  • Reduce recency weighting from 4x to 2x")
    print("  • Add ensemble approach with multiple methods")
    
    print("\n🎯 Priority 3 - Validation:")
    print("  • Test on 2023 data to predict 2024 (proper backtesting)")
    print("  • Weekly performance validation")
    print("  • Position-specific accuracy metrics")
    
    print("\n🎯 Expected Results:")
    print("  • Improve from D grade to B+ grade")
    print("  • Beat ADP consensus by 5-10%")
    print("  • Reduce bust rate from 50% to <25%")

if __name__ == "__main__":
    diagnose_model_issues()
    expanded_data = create_expanded_training_dataset()
    create_position_specific_models()
    create_ensemble_approach()
    model_architecture_recommendations() 