#!/usr/bin/env python3
"""
Update players.json with ADP data from fantasy_metrics_2024.csv
"""

import pandas as pd
import json

def update_players_json_with_adp():
    print("🔄 UPDATING PLAYERS.JSON WITH ADP DATA")
    print("=" * 45)
    
    try:
        # Load the updated fantasy data with ADP
        df = pd.read_csv('data/fantasy_metrics_2024.csv')
        print(f"📊 Loaded {len(df)} players from fantasy metrics")
        
        # Load current players.json
        with open('data/players.json', 'r') as f:
            players_data = json.load(f)
        print(f"📱 Loaded {len(players_data)} players from players.json")
        
        # Update players with ADP data
        updated_count = 0
        adp_count = 0
        rookie_count = 0
        
        for idx, row in df.iterrows():
            player_name = row.get('name', '').strip()
            
            # Find matching player in players.json
            for player_id, player_info in players_data.items():
                json_name = player_info.get('name', '').strip()
                
                if json_name.upper() == player_name.upper():
                    # Update ADP information
                    if pd.notna(row.get('adp_rank')):
                        player_info['adp_rank'] = int(row['adp_rank'])
                        player_info['adp_avg'] = float(row.get('adp_avg', row['adp_rank']))
                        player_info['adp_tier'] = row.get('adp_tier', 'Unknown')
                        adp_count += 1
                    
                    # Update rookie status  
                    if row.get('is_rookie', False):
                        player_info['is_rookie'] = True
                        player_info['draft_round'] = row.get('draft_round', '')
                        player_info['draft_pick'] = row.get('draft_pick', '')
                        rookie_count += 1
                    
                    # Update other relevant fields
                    if pd.notna(row.get('consensus_bye_week')):
                        player_info['bye_week'] = int(row['consensus_bye_week'])
                    
                    updated_count += 1
                    break
        
        # Save updated players.json
        with open('data/players.json', 'w') as f:
            json.dump(players_data, f, indent=2)
        
        print(f"\n✅ UPDATE COMPLETE!")
        print(f"   📊 {updated_count} players updated")
        print(f"   📈 {adp_count} players with ADP data")
        print(f"   🆕 {rookie_count} rookies marked")
        print(f"   💾 Saved updated players.json")
        
        # Show sample ADP data
        adp_players = [p for p in players_data.values() if p.get('adp_rank')]
        if adp_players:
            print(f"\n📈 SAMPLE ADP DATA:")
            sorted_adp = sorted(adp_players, key=lambda x: x.get('adp_rank', 999))[:10]
            for i, player in enumerate(sorted_adp):
                print(f"  {i+1}. {player['name']} - ADP: {player['adp_rank']} ({player.get('adp_tier', 'Unknown')})")
        
        # Show sample rookies
        rookie_players = [p for p in players_data.values() if p.get('is_rookie')]
        if rookie_players:
            print(f"\n🆕 SAMPLE ROOKIES:")
            for i, player in enumerate(rookie_players[:10]):
                round_pick = f"R{player.get('draft_round', '?')}-{player.get('draft_pick', '?')}"
                print(f"  {i+1}. {player['name']} ({player.get('position', '?')}) - {round_pick}")
        
    except Exception as e:
        print(f"❌ Error updating players.json: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    update_players_json_with_adp() 