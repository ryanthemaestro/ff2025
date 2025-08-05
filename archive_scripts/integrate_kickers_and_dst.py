#!/usr/bin/env python3
"""
Integrate Kickers and Defense/Special Teams from FantasyPros Projections
- Add K and DST players to the system
- Keep separate from model training (like rookies)
- Display with proper position filters

REAL NFL DATA ONLY
"""

import pandas as pd
import json

def integrate_kickers_and_dst():
    print("🏈 INTEGRATING KICKERS AND DEFENSE/SPECIAL TEAMS")
    print("=" * 55)

    try:
        # Load the K and DST projection CSVs
        kickers_df = pd.read_csv('data/FantasyPros_Fantasy_Football_Projections_K.csv')
        dst_df = pd.read_csv('data/FantasyPros_Fantasy_Football_Projections_DST.csv')
        
        print(f"📊 Loaded {len(kickers_df)} kickers")
        print(f"🛡️ Loaded {len(dst_df)} defenses")

        # Load main data files
        df = pd.read_csv('data/fantasy_metrics_2024.csv')
        print(f"📊 Loaded {len(df)} players from fantasy_metrics_2024.csv")

        with open('data/players.json', 'r') as f:
            players_data = json.load(f)
        print(f"📱 Loaded {len(players_data)} players from players.json")

        # Get team-to-bye mapping from existing ADP data for bye weeks
        try:
            adp_df = pd.read_csv('data/FantasyPros_2025_Overall_ADP_Rankings.csv',
                                on_bad_lines='skip', quoting=1)
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
            print(f"📅 Loaded bye week mapping for {len(team_bye_mapping)} teams")
        except:
            team_bye_mapping = {}
            print("⚠️ Could not load bye week mapping, will use default")

        # STEP 1: Add Kickers
        kickers_added = 0
        for _, kicker_row in kickers_df.iterrows():
            # Skip if Player name is missing/NaN
            if pd.isna(kicker_row['Player']) or not str(kicker_row['Player']).strip():
                continue
                
            name = str(kicker_row['Player']).strip()
            team = str(kicker_row['Team']).strip().upper() if pd.notna(kicker_row['Team']) else 'FA'
            projected_points = float(kicker_row['FPTS']) if pd.notna(kicker_row['FPTS']) else 0.0
            
            # Get bye week from mapping
            bye_week = team_bye_mapping.get(team, None)
            
            # Create unique ID for kicker
            player_id = f"K_{name.replace(' ', '_').upper()}_{team}"
            
            # Add to DataFrame
            new_kicker_data = {
                'name': name,
                'position': 'K',
                'team': team,
                'is_rookie': False,  # K are not rookies in our system
                'adp_rank': None,  # K don't have ADP ranks in our main ADP file
                'adp_tier': "Kicker",
                'projected_points': projected_points,
                'age': 27,  # Default age for kickers
                'fantasy_points_ppr': projected_points,  # Use projection as historical
                'games_played': 17,  # Assume full season
                'games_started': 17,
                'bye_week': bye_week,
                'injury_status': 'Healthy',
                'draft_round': None,
                'draft_pick': None,
                # Initialize other columns to avoid NaNs
                'avg_fantasy_points_3yr': projected_points,
                'consistency_score': 0.8,  # Kickers are generally consistent
                'performance_trend': 0.0,
                'durability_score': 0.9,  # Kickers rarely get injured
                'opportunity_score_weighted': projected_points,
                'age_adjusted_experience': projected_points,
                'is_prime_age': True,
                'is_declining_age': False,
                'vbd': projected_points * 0.5,  # Lower VBD for kickers
                'scarcity_mult': 1.0,
                'starting_boost': 0.0,
                'bench_boost': 0.0,
                'round_strategy': 0.0,
                'optimized_score': projected_points,
                'avg_targets_3yr': 0.0,  # Kickers don't have targets
                'durability_score_weighted': 0.9,
                'performance_intelligence_boost': 0.0,
                'reality_check_discount': 0.0,
                'injury_penalty': 0.0,
                'team_need_multiplier': 1.0,
                # Kicker-specific stats
                'field_goals': float(kicker_row['FG']) if pd.notna(kicker_row['FG']) else 0.0,
                'field_goal_attempts': float(kicker_row['FGA']) if pd.notna(kicker_row['FGA']) else 0.0,
                'extra_points': float(kicker_row['XPT']) if pd.notna(kicker_row['XPT']) else 0.0,
            }
            df = pd.concat([df, pd.DataFrame([new_kicker_data])], ignore_index=True)
            kickers_added += 1

            # Add to players.json
            players_data[player_id] = {
                'name': name,
                'position': 'K',
                'team': team,
                'is_rookie': False,
                'adp_rank': None,
                'adp_tier': "Kicker",
                'projected_points': projected_points,
                'age': 27,
                'bye_week': bye_week,
                'injury_status': 'Healthy',
                'draft_round': None,
                'draft_pick': None,
                'optimized_score': projected_points,
                'vbd': projected_points * 0.5
            }

        # STEP 2: Add Defense/Special Teams
        dst_added = 0
        for _, dst_row in dst_df.iterrows():
            # Skip if team name is missing/NaN
            if pd.isna(dst_row['Player']) or not str(dst_row['Player']).strip():
                continue
                
            team_name = str(dst_row['Player']).strip()  # Team name like "Minnesota Vikings"
            # Extract team abbreviation (approximate)
            team_mapping = {
                'Minnesota Vikings': 'MIN', 'Tampa Bay Buccaneers': 'TB', 'Seattle Seahawks': 'SEA',
                'New Orleans Saints': 'NO', 'Chicago Bears': 'CHI', 'Cincinnati Bengals': 'CIN',
                'Buffalo Bills': 'BUF', 'San Francisco 49ers': 'SF', 'Dallas Cowboys': 'DAL',
                'Arizona Cardinals': 'ARI', 'Green Bay Packers': 'GB', 'Philadelphia Eagles': 'PHI',
                'Pittsburgh Steelers': 'PIT', 'Kansas City Chiefs': 'KC', 'Los Angeles Rams': 'LAR',
                'Miami Dolphins': 'MIA', 'Houston Texans': 'HOU', 'Baltimore Ravens': 'BAL',
                'Detroit Lions': 'DET', 'Indianapolis Colts': 'IND', 'Los Angeles Chargers': 'LAC',
                'Jacksonville Jaguars': 'JAX', 'Tennessee Titans': 'TEN', 'Atlanta Falcons': 'ATL',
                'New York Jets': 'NYJ', 'New York Giants': 'NYG', 'Cleveland Browns': 'CLE',
                'Washington Commanders': 'WAS', 'Las Vegas Raiders': 'LV', 'Carolina Panthers': 'CAR',
                'Denver Broncos': 'DEN', 'New England Patriots': 'NE'
            }
            team = team_mapping.get(team_name, team_name[:3].upper())  # Fallback to first 3 chars
            
            projected_points = float(dst_row['FPTS']) if pd.notna(dst_row['FPTS']) else 0.0
            
            # Get bye week from mapping
            bye_week = team_bye_mapping.get(team, None)
            
            # Create unique ID for DST
            player_id = f"DST_{team}"
            
            # Add to DataFrame
            new_dst_data = {
                'name': f"{team} DST",
                'position': 'DST',
                'team': team,
                'is_rookie': False,  # DST are not rookies
                'adp_rank': None,  # DST don't have ADP ranks in our main ADP file
                'adp_tier': "Defense",
                'projected_points': projected_points,
                'age': 0,  # Not applicable for team defense
                'fantasy_points_ppr': projected_points,  # Use projection as historical
                'games_played': 17,  # Assume full season
                'games_started': 17,
                'bye_week': bye_week,
                'injury_status': 'Healthy',
                'draft_round': None,
                'draft_pick': None,
                # Initialize other columns to avoid NaNs
                'avg_fantasy_points_3yr': projected_points,
                'consistency_score': 0.6,  # DST can be volatile
                'performance_trend': 0.0,
                'durability_score': 1.0,  # Teams don't get injured
                'opportunity_score_weighted': projected_points,
                'age_adjusted_experience': projected_points,
                'is_prime_age': True,
                'is_declining_age': False,
                'vbd': projected_points * 0.4,  # Lower VBD for DST
                'scarcity_mult': 1.0,
                'starting_boost': 0.0,
                'bench_boost': 0.0,
                'round_strategy': 0.0,
                'optimized_score': projected_points,
                'avg_targets_3yr': 0.0,  # DST don't have targets
                'durability_score_weighted': 1.0,
                'performance_intelligence_boost': 0.0,
                'reality_check_discount': 0.0,
                'injury_penalty': 0.0,
                'team_need_multiplier': 1.0,
                # DST-specific stats
                'sacks': float(dst_row['SACK']) if pd.notna(dst_row['SACK']) else 0.0,
                'interceptions': float(dst_row['INT']) if pd.notna(dst_row['INT']) else 0.0,
                'fumble_recoveries': float(dst_row['FR']) if pd.notna(dst_row['FR']) else 0.0,
                'forced_fumbles': float(dst_row['FF']) if pd.notna(dst_row['FF']) else 0.0,
                'touchdowns': float(dst_row['TD']) if pd.notna(dst_row['TD']) else 0.0,
                'safeties': float(dst_row['SAFETY']) if pd.notna(dst_row['SAFETY']) else 0.0,
                'points_allowed': float(dst_row['PA']) if pd.notna(dst_row['PA']) else 0.0,
                'yards_allowed': float(dst_row['YDS_AGN']) if pd.notna(dst_row['YDS_AGN']) else 0.0,
            }
            df = pd.concat([df, pd.DataFrame([new_dst_data])], ignore_index=True)
            dst_added += 1

            # Add to players.json
            players_data[player_id] = {
                'name': f"{team} DST",
                'position': 'DST',
                'team': team,
                'is_rookie': False,
                'adp_rank': None,
                'adp_tier': "Defense",
                'projected_points': projected_points,
                'age': 0,
                'bye_week': bye_week,
                'injury_status': 'Healthy',
                'draft_round': None,
                'draft_pick': None,
                'optimized_score': projected_points,
                'vbd': projected_points * 0.4
            }

        # Save updated data
        df.to_csv('data/fantasy_metrics_2024.csv', index=False)
        with open('data/players.json', 'w') as f:
            json.dump(players_data, f, indent=2)

        print(f"\n🎉 SUCCESS! Added {kickers_added} kickers and {dst_added} defenses.")
        print("✅ Updated data/fantasy_metrics_2024.csv")
        print("✅ Updated data/players.json")

        # Display top performers from each position
        kickers_in_df = df[df['position'] == 'K'].sort_values('projected_points', ascending=False)
        dst_in_df = df[df['position'] == 'DST'].sort_values('projected_points', ascending=False)
        
        if not kickers_in_df.empty:
            print("\n⚡ TOP 5 KICKERS:")
            for i, (_, kicker) in enumerate(kickers_in_df.head(5).iterrows()):
                print(f"  {i+1}. {kicker['name']} ({kicker['team']}) - {kicker['projected_points']:.1f} pts | Bye Week: {kicker['bye_week']}")

        if not dst_in_df.empty:
            print("\n🛡️ TOP 5 DEFENSES:")
            for i, (_, dst) in enumerate(dst_in_df.head(5).iterrows()):
                print(f"  {i+1}. {dst['name']} - {dst['projected_points']:.1f} pts | Bye Week: {dst['bye_week']}")

    except Exception as e:
        print(f"❌ Error integrating kickers and DST: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    integrate_kickers_and_dst() 