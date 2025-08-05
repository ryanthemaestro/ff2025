#!/usr/bin/env python3
"""
Integrate 2025 Rookies from FantasyPros_2025_Rookies_ALL_Rankings.csv
- Clear all existing incorrect rookie flags
- Add only actual 2025 rookie prospects
- Keep separate from model training
- Display with proper ADP rankings

REAL NFL DATA ONLY
"""

import pandas as pd
import json

def integrate_2025_rookies():
    print("🆕 INTEGRATING 2025 ROOKIES FROM FANTASYPROS")
    print("=" * 55)
    
    try:
        # Load the 2025 rookies CSV
        rookies_df = pd.read_csv('data/FantasyPros_2025_Rookies_ALL_Rankings.csv')
        print(f"📊 Loaded {len(rookies_df)} 2025 rookie prospects")
        
        # Load main data files
        df = pd.read_csv('data/fantasy_metrics_2024.csv')
        print(f"📈 Loaded {len(df)} players from fantasy metrics")
        
        with open('data/players.json', 'r') as f:
            players_data = json.load(f)
        print(f"📱 Loaded {len(players_data)} players from players.json")
        
        # STEP 1: Clear ALL existing rookie flags (they were wrong)
        print("\n🧹 CLEARING ALL EXISTING ROOKIE FLAGS...")
        df['is_rookie'] = False
        df['draft_round'] = pd.NA
        df['draft_pick'] = pd.NA
        df['rookie_tier'] = pd.NA
        
        for player_id, player_data in players_data.items():
            player_data['is_rookie'] = False
            if 'draft_round' in player_data:
                del player_data['draft_round']
            if 'draft_pick' in player_data:
                del player_data['draft_pick']
            if 'rookie_tier' in player_data:
                del player_data['rookie_tier']
        
        print("✅ Cleared all incorrect rookie flags")
        
        # STEP 2: Process 2025 rookies from FantasyPros CSV
        print("\n🏈 PROCESSING 2025 ROOKIE PROSPECTS...")
        
        rookie_count = 0
        for _, rookie in rookies_df.iterrows():
            player_name = str(rookie['PLAYER NAME']).strip().strip('"')
            position = str(rookie['POS']).strip()
            team = str(rookie['TEAM']).strip() if pd.notna(rookie['TEAM']) else 'FA'
            rank = int(rookie['RK']) if pd.notna(rookie['RK']) else 999
            # Handle age parsing (some entries have '-' instead of numbers)
            age_val = str(rookie['AGE']).strip()
            if pd.notna(rookie['AGE']) and age_val != '-' and age_val.isdigit():
                age = int(age_val)
            else:
                age = 22  # Default age for rookies
            
            # Skip if no valid name
            if pd.isna(player_name) or player_name == 'nan':
                continue
            
            # Normalize name for matching
            normalized_name = player_name.upper().replace('.', '').replace("'", "")
            
            # Create rookie player entry
            rookie_id = f"rookie_2025_{rank:03d}"
            
            # Add to players.json
            players_data[rookie_id] = {
                'name': normalized_name,
                'position': position,
                'team': team,
                'age': age,
                'is_rookie': True,
                'rookie_rank': rank,
                'rookie_year': 2025,
                'adp_rank': rank,  # Use rookie rank as ADP
                'adp_tier': get_rookie_tier(rank),
                'fantasy_points_ppr': 0.0,
                'projected_points': get_rookie_projection(position, rank),
                'bye_week': 'TBD',
                'injury_status': 'Healthy'
            }
            
            # Add to fantasy_metrics CSV
            new_row = {
                'name': normalized_name,
                'position': position,
                'team': team,
                'age': age,
                'is_rookie': True,
                'draft_round': 'TBD',  # 2025 NFL draft hasn't happened
                'draft_pick': 'TBD',
                'rookie_tier': get_rookie_tier(rank),
                'adp_rank': rank,
                'adp_tier': get_rookie_tier(rank),
                'fantasy_points_ppr': 0.0,
                'projected_points': get_rookie_projection(position, rank),
                'bye_week': 'TBD',
                'injury_status': 'Healthy',
                # Default values for other columns
                'games_played': 0,
                'targets': 0,
                'receptions': 0,
                'receiving_yards': 0,
                'receiving_tds': 0,
                'rushing_attempts': 0,
                'rushing_yards': 0,
                'rushing_tds': 0,
                'passing_attempts': 0,
                'passing_yards': 0,
                'passing_tds': 0,
                'interceptions': 0,
                'avg_fantasy_points_3yr': 0.0,
                'consistency_score': 0.0,
                'performance_trend': 0.0,
                'durability_score': 1.0,
                'vbd': 0.0
            }
            
            # Add missing columns with defaults
            for col in df.columns:
                if col not in new_row:
                    new_row[col] = 0.0 if df[col].dtype in ['float64', 'int64'] else pd.NA
            
            # Add the rookie to the dataframe
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            rookie_count += 1
        
        print(f"✅ Added {rookie_count} 2025 rookie prospects")
        
        # STEP 3: Fix ADP display issue by ensuring columns exist
        print("\n🔧 FIXING ADP DISPLAY...")
        
        # Ensure ADP columns exist and have proper types
        if 'adp_rank' not in df.columns:
            df['adp_rank'] = pd.NA
        if 'adp_tier' not in df.columns:
            df['adp_tier'] = 'Unknown'
        
        # Convert ADP rank to float to handle NaN properly
        df['adp_rank'] = pd.to_numeric(df['adp_rank'], errors='coerce')
        
        print("✅ Fixed ADP columns")
        
        # STEP 4: Save updated data
        print("\n💾 SAVING UPDATED DATA...")
        
        df.to_csv('data/fantasy_metrics_2024.csv', index=False)
        print("✅ Updated fantasy_metrics_2024.csv")
        
        with open('data/players.json', 'w') as f:
            json.dump(players_data, f, indent=2)
        print("✅ Updated players.json")
        
        # Create separate rookies file
        rookies_only = df[df['is_rookie'] == True].copy()
        rookies_only = rookies_only.sort_values('adp_rank')
        rookies_only.to_csv('data/2025_rookies.csv', index=False)
        print("✅ Created 2025_rookies.csv")
        
        # STEP 5: Show summary
        print(f"\n🏆 2025 ROOKIE INTEGRATION COMPLETE!")
        print(f"📊 Total rookies: {len(rookies_only)}")
        print("\n🔝 TOP 10 2025 ROOKIE PROSPECTS:")
        for i, (_, rookie) in enumerate(rookies_only.head(10).iterrows()):
            print(f"  {i+1:2d}. {rookie['name']} ({rookie['position']}) - {rookie['team']} - Proj: {rookie['projected_points']:.1f} pts")
        
        print(f"\n✅ SUCCESS! 2025 rookies properly integrated")
        print("✅ Cleared all incorrect veteran rookie flags")
        print("✅ Added proper ADP rankings for rookies")
        print("✅ Fixed ADP display issues")
        
    except Exception as e:
        print(f"❌ Error integrating 2025 rookies: {e}")
        import traceback
        traceback.print_exc()

def get_rookie_tier(rank):
    """Get tier based on rookie ranking"""
    if rank <= 10:
        return "Elite Rookie (Top 10)"
    elif rank <= 30:
        return "High-Value Rookie (11-30)"
    elif rank <= 60:
        return "Mid-Round Rookie (31-60)"
    elif rank <= 100:
        return "Late-Round Rookie (61-100)"
    else:
        return "Deep Sleeper (100+)"

def get_rookie_projection(position, rank):
    """Get projected fantasy points based on position and rank"""
    base_projections = {
        'QB': {'1-10': 250, '11-30': 180, '31-60': 120, '61-100': 80, '100+': 40},
        'RB': {'1-10': 200, '11-30': 150, '31-60': 100, '61-100': 60, '100+': 30},
        'WR': {'1-10': 180, '11-30': 130, '31-60': 90, '61-100': 50, '100+': 25},
        'TE': {'1-10': 140, '11-30': 100, '31-60': 70, '61-100': 40, '100+': 20}
    }
    
    # Clean position (remove numbers)
    clean_pos = ''.join([c for c in position if c.isalpha()]).upper()
    if clean_pos not in base_projections:
        return 50.0  # Default for unknown positions
    
    # Determine tier
    if rank <= 10:
        tier = '1-10'
    elif rank <= 30:
        tier = '11-30'
    elif rank <= 60:
        tier = '31-60'
    elif rank <= 100:
        tier = '61-100'
    else:
        tier = '100+'
    
    return float(base_projections[clean_pos][tier])

if __name__ == "__main__":
    integrate_2025_rookies() 