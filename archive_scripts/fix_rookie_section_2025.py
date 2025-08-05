#!/usr/bin/env python3
"""
Fix Rookie Section for 2025 Season
Since it's 2025, 2024 rookies are now in their second year
2025 rookies haven't been drafted yet (draft is in April/May)

REAL NFL DATA ONLY
"""

import pandas as pd
import json
from datetime import datetime

def fix_rookie_section_2025():
    print("🆕 FIXING ROOKIE SECTION FOR 2025 SEASON")
    print("=" * 55)
    
    try:
        # Load current data
        df = pd.read_csv('data/fantasy_metrics_2024.csv')
        print(f"📊 Loaded {len(df)} players from fantasy data")
        
        # Clear all existing rookie flags since 2024 rookies are now 2nd year players
        df['is_rookie'] = False
        
        # Load players.json
        with open('data/players.json', 'r') as f:
            players_data = json.load(f)
        
        # Clear rookie flags in players.json too
        rookie_count_cleared = 0
        for player_id, player_data in players_data.items():
            if player_data.get('is_rookie', False):
                player_data['is_rookie'] = False
                rookie_count_cleared += 1
        
        print(f"🧹 Cleared {rookie_count_cleared} outdated rookie flags from 2024")
        
        # Since 2025 NFL Draft hasn't happened yet, we have a few options:
        # Option 1: Leave rookie section empty
        # Option 2: Create some projected 2025 rookies based on college prospects
        # Let's go with Option 1 for accuracy
        
        print("\n📅 2025 ROOKIE STATUS:")
        print("   • 2024 rookies are now in their SECOND YEAR")
        print("   • 2025 NFL Draft hasn't happened yet (April/May 2025)")
        print("   • Rookie section will be empty until after the draft")
        
        # Save updated data
        df.to_csv('data/fantasy_metrics_2024.csv', index=False)
        
        with open('data/players.json', 'w') as f:
            json.dump(players_data, f, indent=2)
        
        # Create an empty rookie rankings file for now
        empty_rookies = pd.DataFrame(columns=[
            'name', 'position', 'team', 'projected_points', 
            'is_rookie', 'rookie_rank', 'draft_round', 'draft_pick'
        ])
        empty_rookies.to_csv('data/rookie_rankings_2024.csv', index=False)
        
        print("\n✅ ROOKIE SECTION FIXED FOR 2025:")
        print("   ✅ Cleared outdated 2024 rookie flags")
        print("   ✅ Rookie section now properly empty")
        print("   ✅ Will be populated after 2025 NFL Draft")
        
        # Create a note for the UI
        rookie_note = {
            "note": "2025 NFL Draft hasn't occurred yet. Rookie rankings will be available after the April/May 2025 draft.",
            "year": 2025,
            "status": "pre_draft",
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }
        
        with open('data/rookie_status.json', 'w') as f:
            json.dump(rookie_note, f, indent=2)
        
        print("   ✅ Created rookie status note for UI")
        
    except Exception as e:
        print(f"❌ Error fixing rookie section: {e}")

if __name__ == "__main__":
    fix_rookie_section_2025() 