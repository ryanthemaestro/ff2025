#!/usr/bin/env python3
"""
Add Bye Weeks to Rookies Based on Team Data from FantasyPros CSV
"""

import pandas as pd
import json

def add_rookie_bye_weeks():
    print("📅 ADDING BYE WEEKS TO ROOKIES BASED ON TEAM DATA")
    print("=" * 55)
    
    try:
        # Load FantasyPros CSV to get team-to-bye mapping  
        adp_df = pd.read_csv('data/FantasyPros_2025_Overall_ADP_Rankings.csv', 
                            on_bad_lines='skip', quoting=1)
        print(f"📊 Loaded FantasyPros data with {len(adp_df)} players")
        
        # Create team-to-bye mapping from FantasyPros data
        team_bye_mapping = {}
        for _, row in adp_df.iterrows():
            team = str(row.get('Team', '')).strip().upper()
            bye = row.get('Bye', '')
            
            if team and pd.notna(bye) and bye != '':
                try:
                    bye_week = int(bye)
                    if bye_week > 0:
                        team_bye_mapping[team] = bye_week
                except:
                    continue
        
        print(f"📅 Created bye week mapping for {len(team_bye_mapping)} teams:")
        for team, bye in sorted(team_bye_mapping.items()):
            print(f"   {team}: Week {bye}")
        
        # Load our data files
        df = pd.read_csv('data/fantasy_metrics_2024.csv')
        
        with open('data/players.json', 'r') as f:
            players_data = json.load(f)
        
        # Update rookies with bye weeks based on their teams
        rookies_updated = 0
        
        # Update CSV file
        for idx, row in df.iterrows():
            if row.get('is_rookie', False):
                team = str(row.get('team', '')).strip().upper()
                if team in team_bye_mapping:
                    df.at[idx, 'bye_week'] = team_bye_mapping[team]
                    rookies_updated += 1
                    print(f"✅ {row['name']} ({team}) → Bye Week {team_bye_mapping[team]}")
        
        # Update JSON file  
        for player_id, player_data in players_data.items():
            if player_data.get('is_rookie', False):
                team = str(player_data.get('team', '')).strip().upper()
                if team in team_bye_mapping:
                    player_data['bye_week'] = team_bye_mapping[team]
        
        # Save updated files
        df.to_csv('data/fantasy_metrics_2024.csv', index=False)
        
        with open('data/players.json', 'w') as f:
            json.dump(players_data, f, indent=2)
        
        print(f"\n🎉 SUCCESS! Updated {rookies_updated} rookies with bye weeks")
        print("✅ Updated data/fantasy_metrics_2024.csv")
        print("✅ Updated data/players.json")
        
        # Show some examples
        rookie_examples = df[df['is_rookie'] == True].head(5)
        if not rookie_examples.empty:
            print("\n📊 ROOKIE BYE WEEK EXAMPLES:")
            for _, rookie in rookie_examples.iterrows():
                bye = rookie.get('bye_week', 'Unknown')
                team = rookie.get('team', 'Unknown')
                print(f"   {rookie['name']} ({team}) - Bye Week {bye}")
        
    except Exception as e:
        print(f"❌ Error adding rookie bye weeks: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    add_rookie_bye_weeks() 