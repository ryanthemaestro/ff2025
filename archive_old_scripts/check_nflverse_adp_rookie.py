#!/usr/bin/env python3
"""
Check nflverse for ADP and Rookie Data
Test what data is available for fantasy ADP and rookie rankings
"""

import nfl_data_py as nfl
import pandas as pd
from datetime import datetime

def test_nflverse_adp_rookie():
    """Test nflverse data for ADP and rookie information"""
    print("🔍 TESTING NFLVERSE FOR ADP AND ROOKIE DATA")
    print("=" * 60)
    
    try:
        # Test 1: Check what draft-related data is available
        print("\n📋 Testing Draft Data...")
        draft_picks = nfl.import_draft_picks([2024])
        print(f"✅ Draft picks 2024: {len(draft_picks)} entries")
        print(f"📊 Columns: {list(draft_picks.columns)}")
        if not draft_picks.empty:
            print("\n🏈 Sample draft picks:")
            print(draft_picks.head(10)[['pfr_player_name', 'position', 'round', 'pick', 'team']])
        
        # Test 2: Check draft values
        print("\n💰 Testing Draft Values...")
        draft_values = nfl.import_draft_values()
        print(f"✅ Draft values: {len(draft_values)} entries")
        print(f"📊 Columns: {list(draft_values.columns)}")
        if not draft_values.empty:
            print("\n📈 Sample draft values:")
            print(draft_values.head())
        
        # Test 3: Check seasonal rosters for rookies
        print("\n👥 Testing Seasonal Rosters for Rookie Info...")
        rosters = nfl.import_seasonal_rosters([2024])
        print(f"✅ Roster data 2024: {len(rosters)} entries")
        print(f"📊 Columns: {list(rosters.columns)}")
        
        # Check for rookie-related columns
        rookie_columns = [col for col in rosters.columns if 'rookie' in col.lower() or 'draft' in col.lower()]
        print(f"🆕 Rookie-related columns: {rookie_columns}")
        
        if not rosters.empty:
            # Look for recent draft picks (rookies)
            rookies_2024 = rosters[rosters.get('draft_year', 0) == 2024] if 'draft_year' in rosters.columns else pd.DataFrame()
            print(f"🎯 2024 Rookies found: {len(rookies_2024)}")
            
            if not rookies_2024.empty:
                print("\n🏈 Sample 2024 rookies:")
                print(rookies_2024.head(10)[['player_name', 'position', 'team'] + 
                                          [col for col in rookies_2024.columns if 'draft' in col.lower()]])
        
        # Test 4: Check if there's any ADP-like data
        print("\n📊 Checking for ADP-like Data...")
        adp_columns = [col for col in rosters.columns if any(term in col.lower() for term in ['adp', 'average', 'rank', 'draft_position'])]
        print(f"🎯 Potential ADP columns: {adp_columns}")
        
        # Test 5: Summary
        print("\n📋 SUMMARY:")
        print("=" * 40)
        print(f"✅ NFL Draft picks data: Available ({len(draft_picks)} entries)")
        print(f"✅ Draft values data: Available ({len(draft_values)} entries)")
        print(f"✅ Rookie identification: {'Available' if 'draft_year' in rosters.columns else 'Limited'}")
        print(f"❌ Fantasy ADP data: {'Available' if adp_columns else 'Not Available'}")
        
        return {
            'has_draft_picks': len(draft_picks) > 0,
            'has_draft_values': len(draft_values) > 0,
            'has_rookie_data': 'draft_year' in rosters.columns,
            'has_adp_data': len(adp_columns) > 0,
            'rookie_columns': rookie_columns,
            'adp_columns': adp_columns
        }
        
    except Exception as e:
        print(f"❌ Error testing nflverse data: {e}")
        return {}

if __name__ == "__main__":
    results = test_nflverse_adp_rookie()
    
    print("\n🎯 RECOMMENDATIONS:")
    print("=" * 40)
    
    if not results.get('has_adp_data', False):
        print("❌ nflverse does NOT have fantasy ADP data")
        print("💡 Need to integrate external ADP sources like:")
        print("   - FantasyPros consensus ADP")
        print("   - ESPN ADP")
        print("   - Yahoo ADP")
        print("   - Sleeper ADP")
    
    if results.get('has_rookie_data', False):
        print("✅ Can identify rookies using draft_year column")
        print("💡 Can create separate rookie rankings using draft position + performance")
    else:
        print("⚠️  Limited rookie identification capability")
        print("💡 Need to cross-reference with draft picks data") 