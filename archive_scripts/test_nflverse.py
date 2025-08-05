#!/usr/bin/env python3
"""
Test nflverse Data Access
Explore what real NFL data we can get from the nflverse ecosystem

Based on: https://github.com/nflverse & https://pypi.org/project/nfl-data-py/
"""

import nfl_data_py as nfl
import pandas as pd
from datetime import datetime

def test_nflverse_data():
    """Test various nflverse data endpoints"""
    print("🔍 TESTING NFLVERSE DATA ACCESS")
    print("=" * 50)
    
    try:
        # Test 1: Get current season rosters
        print("\n📋 Testing Current Season Rosters...")
        current_year = 2024
        rosters = nfl.import_seasonal_rosters([current_year])
        print(f"✅ Got {len(rosters)} roster entries for {current_year}")
        print(f"📊 Columns: {list(rosters.columns)}")
        if not rosters.empty:
            print(f"🏈 Sample players:")
            sample = rosters[['player_name', 'position', 'team']].head()
            print(sample.to_string(index=False))
        
        # Test 2: Get weekly fantasy data
        print(f"\n🏆 Testing Weekly Fantasy Data for {current_year}...")
        weekly = nfl.import_weekly_data([current_year])
        print(f"✅ Got {len(weekly)} weekly records for {current_year}")
        print(f"📊 Columns: {list(weekly.columns)}")
        if not weekly.empty:
            # Look for fantasy-relevant columns
            fantasy_cols = [col for col in weekly.columns if any(keyword in col.lower() 
                          for keyword in ['fantasy', 'points', 'targets', 'yards', 'td'])]
            print(f"🎯 Fantasy-relevant columns: {fantasy_cols[:10]}")
            
            # Show top fantasy performers
            if 'fantasy_points_ppr' in weekly.columns:
                top_performers = weekly.nlargest(5, 'fantasy_points_ppr')[
                    ['player_name', 'position', 'team', 'week', 'fantasy_points_ppr']
                ]
                print(f"🔥 Top Fantasy Performers:")
                print(top_performers.to_string(index=False))
        
        # Test 3: Get seasonal data with market shares
        print(f"\n📈 Testing Seasonal Data with Market Shares...")
        seasonal = nfl.import_seasonal_data([current_year])
        print(f"✅ Got {len(seasonal)} seasonal records for {current_year}")
        print(f"📊 Columns: {list(seasonal.columns)}")
        if not seasonal.empty:
            # Look for market share columns
            market_cols = [col for col in seasonal.columns if '_sh' in col or 'dom' in col]
            print(f"📊 Market share columns: {market_cols}")
            
            # Show top players by targets
            if 'targets' in seasonal.columns:
                top_targets = seasonal.nlargest(5, 'targets')[
                    ['player_name', 'position', 'team', 'targets', 'receptions', 'receiving_yards']
                ]
                print(f"🎯 Top Target Share Players:")
                print(top_targets.to_string(index=False))
        
        # Test 4: Check available years
        print(f"\n📅 Testing Multiple Years...")
        multi_year = nfl.import_weekly_data([2022, 2023, 2024])
        print(f"✅ Got {len(multi_year)} records across 2022-2024")
        year_counts = multi_year['season'].value_counts().sort_index()
        print(f"📊 Records per year:")
        print(year_counts.to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing nflverse data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_nflverse_data()
    if success:
        print("\n🎉 NFLVERSE DATA ACCESS SUCCESSFUL!")
        print("Ready to build comprehensive NFL data collector!")
    else:
        print("\n💥 Issues with nflverse data access") 