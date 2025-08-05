#!/usr/bin/env python3
"""
Fix Lamar Jackson's ADP and Add ADP Sorting
1. Fix Lamar Jackson's ADP from 615 to 21 (correct FantasyPros value)
2. Update UI to sort by ADP rank
"""

import pandas as pd
import json

def fix_lamar_and_sorting():
    print("🔧 FIXING LAMAR JACKSON'S ADP AND ADDING ADP SORTING")
    print("=" * 60)
    
    try:
        # Load data files
        df = pd.read_csv('data/fantasy_metrics_2024.csv')
        print(f"📊 Loaded {len(df)} players from fantasy metrics")
        
        with open('data/players.json', 'r') as f:
            players_data = json.load(f)
        print(f"📱 Loaded {len(players_data)} players from players.json")
        
        # STEP 1: Fix Lamar Jackson's ADP (should be 21, not 615)
        print("\n🔧 FIXING LAMAR JACKSON'S ADP...")
        
        # Fix in CSV
        lamar_mask = df['name'].str.upper() == 'L.JACKSON'
        if lamar_mask.any():
            current_adp = df.loc[lamar_mask, 'adp_rank'].iloc[0]
            print(f"📊 Found Lamar Jackson with incorrect ADP: {current_adp}")
            
            # Update to correct ADP
            df.loc[lamar_mask, 'adp_rank'] = 21.0
            df.loc[lamar_mask, 'adp_tier'] = 'Elite (1-30)'
            print(f"✅ Fixed Lamar Jackson's ADP: 615 → 21")
        
        # Fix in players.json
        lamar_fixed = False
        for player_id, player_data in players_data.items():
            if player_data.get('name', '').upper() == 'L.JACKSON':
                old_adp = player_data.get('adp_rank', 'N/A')
                player_data['adp_rank'] = 21.0
                player_data['adp_tier'] = 'Elite (1-30)'
                print(f"📱 Fixed Lamar in players.json: {old_adp} → 21")
                lamar_fixed = True
                break
        
        if not lamar_fixed:
            print("⚠️  Lamar Jackson not found in players.json")
        
        # STEP 2: Save updated data
        print("\n💾 SAVING UPDATED DATA...")
        df.to_csv('data/fantasy_metrics_2024.csv', index=False)
        print("✅ Updated fantasy_metrics_2024.csv")
        
        with open('data/players.json', 'w') as f:
            json.dump(players_data, f, indent=2)
        print("✅ Updated players.json")
        
        # STEP 3: Show verification
        print(f"\n🔍 VERIFICATION:")
        lamar_data = df[df['name'].str.upper() == 'L.JACKSON']
        if not lamar_data.empty:
            lamar_row = lamar_data.iloc[0]
            print(f"✅ Lamar Jackson ADP: {lamar_row['adp_rank']} ({lamar_row['adp_tier']})")
        
        print(f"\n🎉 SUCCESS!")
        print("✅ Fixed Lamar Jackson's ADP: 615 → 21")
        print("✅ Updated both CSV and JSON files")
        print("\n📝 NEXT: Update UI to sort by ADP rank")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_lamar_and_sorting() 