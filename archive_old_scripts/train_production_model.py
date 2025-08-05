#!/usr/bin/env python3
"""
Production NFL Model Training Script
====================================

This script trains the fantasy football model using our enhanced real historical data.
It leverages the comprehensive multi-year dataset with real NFL statistics.

Usage:
    python scripts/train_production_model.py
"""

import pandas as pd
import os
import sys
from datetime import datetime

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.draft_optimizer import load_players, load_projections, prepare_data, train_model

def main():
    """Train the production model with enhanced real historical data"""
    
    print("🚀 STARTING PRODUCTION MODEL TRAINING")
    print("=" * 60)
    print("🏈 Using Enhanced Real Historical Data (2022-2024)")
    print("📊 Features: Multi-year volume, trends, consistency, injury status")
    print("🤖 Algorithm: RandomForest with comprehensive validation")
    print("=" * 60)
    
    # Load the enhanced comprehensive dataset
    comprehensive_data_path = 'data/comprehensive_nfl_data_2022_2024.csv'
    if not os.path.exists(comprehensive_data_path):
        print("❌ Enhanced dataset not found!")
        print("🔧 Please run: python scripts/comprehensive_api_pipeline.py")
        return False
    
    print(f"📁 Loading enhanced dataset: {comprehensive_data_path}")
    
    try:
        # Load comprehensive data
        comprehensive_df = pd.read_csv(comprehensive_data_path)
        print(f"✅ Loaded {len(comprehensive_df)} players with {len(comprehensive_df.columns)} features")
        
        # Show data quality summary
        print(f"\n📊 ENHANCED DATASET SUMMARY:")
        print(f"   Players: {len(comprehensive_df)}")
        print(f"   Positions: {sorted(comprehensive_df['position'].unique())}")
        print(f"   Features: {len(comprehensive_df.columns)}")
        
        # Show sample of key features
        key_features = ['name', 'position', 'avg_fantasy_points_3yr', 'performance_trend', 
                       'avg_carries_3yr', 'avg_targets_3yr', 'current_fantasy_points']
        available_key_features = [f for f in key_features if f in comprehensive_df.columns]
        
        if available_key_features:
            print(f"\n📈 SAMPLE DATA (Top 5 by fantasy points):")
            sample_data = comprehensive_df.nlargest(5, 'avg_fantasy_points_3yr')[available_key_features]
            for _, player in sample_data.iterrows():
                name = player.get('name', 'Unknown')
                pos = player.get('position', 'Unknown')
                fp = player.get('avg_fantasy_points_3yr', 0)
                trend = player.get('performance_trend', 0)
                print(f"   {name} ({pos}): {fp:.1f} pts, {trend:.1f}% trend")
        
        # Load players and projections (needed for the prepare_data function)
        print(f"\n🔄 Preparing data for model training...")
        players = load_players()
        projections = load_projections()
        
        # Prepare data using the enhanced dataset
        df = prepare_data(players, projections)
        
        # Merge with comprehensive features if available
        if 'name' in comprehensive_df.columns and 'name' in df.columns:
            print(f"🔗 Merging with enhanced multi-year features...")
            # Merge the enhanced features
            enhanced_features = ['avg_fantasy_points_3yr', 'performance_trend', 'consistency_score',
                               'avg_carries_3yr', 'avg_targets_3yr', 'opportunity_score_weighted']
            
            merge_columns = ['name'] + [col for col in enhanced_features if col in comprehensive_df.columns]
            df = df.merge(comprehensive_df[merge_columns], on='name', how='left', suffixes=('', '_enhanced'))
            
            print(f"✅ Enhanced features merged successfully")
        
        print(f"\n🎯 TRAINING MODEL WITH ENHANCED DATA")
        print(f"   📊 Total features available: {len(df.columns)}")
        print(f"   🏈 Training samples: {len(df)}")
        
        # Train the model
        model_result = train_model(df)
        
        if model_result is not None:
            print(f"\n🎉 MODEL TRAINING COMPLETE!")
            print(f"✅ Production model saved to: models/draft_model.pkl")
            print(f"🏈 Model trained on REAL NFL data (2022-2024)")
            print(f"📊 Features include multi-year trends and consistency metrics")
            
            # Verify model file exists
            if os.path.exists('models/draft_model.pkl'):
                file_size = os.path.getsize('models/draft_model.pkl') / (1024 * 1024)  # MB
                print(f"📁 Model file size: {file_size:.1f} MB")
                
                print(f"\n🚀 PRODUCTION READY!")
                print(f"📋 NEXT STEPS:")
                print(f"   1. ✅ Model is trained on real historical data")
                print(f"   2. 🎯 UI will now use enhanced predictions") 
                print(f"   3. 📈 Player rankings based on real NFL performance")
                print(f"   4. 🔄 Restart UI: python scripts/draft_ui.py")
                
                return True
            else:
                print(f"❌ Model file not found after training")
                return False
        else:
            print(f"❌ Model training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🏆 SUCCESS! Production model ready with enhanced real NFL data!")
    else:
        print(f"\n❌ Training failed. Check error messages above.") 