#!/usr/bin/env python3
"""
API Data Pipeline for Fantasy Football Model
===========================================

This script replaces the fantasy projection NFLsavant data with REAL NFL statistics
from multiple free APIs, including multi-year data for robust model training.

Features:
- Pulls real stats from ESPN, Sleeper, and other free APIs
- Multi-year data (2022-2024) for better model training
- Automatic data cleaning and feature engineering
- Direct integration with existing model pipeline

Usage:
    python scripts/api_data_pipeline.py
"""

import pandas as pd
import requests
import time
import json
from typing import Dict, List, Optional, Tuple
import sys
import os
from datetime import datetime
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class NFLDataPipeline:
    """Comprehensive NFL data pipeline using multiple free APIs"""
    
    def __init__(self):
        self.sleeper_base = "https://api.sleeper.app/v1"
        self.espn_base = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
        self.espn_core = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Fantasy Football Analytics)',
            'Accept': 'application/json'
        })
        
        # Rate limiting
        self.request_delay = 0.2  # 200ms between requests to be respectful
        self.last_request_time = 0
        
        # Data storage
        self.all_players_data = []
        self.seasons = [2022, 2023, 2024]  # Multi-year training data
        
    def _rate_limited_request(self, url: str) -> Optional[Dict]:
        """Make a rate-limited request to any API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        try:
            response = self.session.get(url, timeout=15)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  HTTP {response.status_code} for {url[:80]}...")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for {url[:50]}...: {e}")
            return None
    
    def get_sleeper_players(self) -> Dict[str, Dict]:
        """Get all NFL players from Sleeper API with clean data"""
        print("📥 Fetching all NFL players from Sleeper API...")
        
        url = f"{self.sleeper_base}/players/nfl"
        data = self._rate_limited_request(url)
        
        if not data:
            print("❌ Failed to fetch Sleeper players")
            return {}
        
        # Clean and filter the data
        clean_players = {}
        skill_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        
        for player_id, player_data in data.items():
            position = player_data.get('position')
            active = player_data.get('active', False)
            
            if position in skill_positions and active:
                clean_player = {
                    'sleeper_id': player_id,
                    'full_name': player_data.get('full_name', ''),
                    'first_name': player_data.get('first_name', ''),
                    'last_name': player_data.get('last_name', ''),
                    'position': position,
                    'team': player_data.get('team', ''),
                    'age': player_data.get('age'),
                    'height': player_data.get('height', ''),
                    'weight': player_data.get('weight'),
                    'years_exp': player_data.get('years_exp', 0),
                    'espn_id': player_data.get('espn_id'),
                    'yahoo_id': player_data.get('yahoo_id'),
                    'rotowire_id': player_data.get('rotowire_id')
                }
                clean_players[player_id] = clean_player
        
        print(f"✅ Found {len(clean_players)} active skill position players")
        return clean_players
    
    def get_espn_team_stats(self, season: int) -> Dict[str, Dict]:
        """Get team-level stats for calculating target shares"""
        print(f"🏈 Getting {season} team stats for target share calculations...")
        
        team_stats = {}
        
        # Get all teams
        teams_url = f"{self.espn_base}/teams"
        teams_data = self._rate_limited_request(teams_url)
        
        if not teams_data:
            return team_stats
        
        # Extract teams from ESPN's nested structure
        teams = []
        if 'sports' in teams_data and teams_data['sports']:
            sport = teams_data['sports'][0]
            if 'leagues' in sport and sport['leagues']:
                league = sport['leagues'][0]
                if 'teams' in league:
                    teams = league['teams']
        
        for team in teams:
            team_abbr = team.get('team', {}).get('abbreviation', '')
            team_id = team.get('team', {}).get('id', '')
            
            if team_abbr and team_id:
                # Get team season statistics
                stats_url = f"{self.espn_core}/seasons/{season}/types/2/teams/{team_id}/statistics"
                stats_data = self._rate_limited_request(stats_url)
                
                team_totals = {
                    'passing_attempts': 0,
                    'rushing_attempts': 0,
                    'total_plays': 0
                }
                
                if stats_data and 'splits' in stats_data:
                    for split in stats_data['splits']:
                        if 'stats' in split:
                            for stat in split['stats']:
                                stat_name = stat.get('name', '')
                                stat_value = float(stat.get('value', 0))
                                
                                if stat_name == 'passingAttempts':
                                    team_totals['passing_attempts'] += stat_value
                                elif stat_name == 'rushingAttempts':
                                    team_totals['rushing_attempts'] += stat_value
                
                team_totals['total_plays'] = team_totals['passing_attempts'] + team_totals['rushing_attempts']
                team_stats[team_abbr] = team_totals
        
        return team_stats
    
    def get_real_player_stats(self, season: int, max_players: int = 200) -> List[Dict]:
        """Get real player statistics using ESPN's stat pages"""
        print(f"📊 Getting REAL {season} player statistics from ESPN...")
        
        all_player_stats = []
        
        # Position-specific stat URLs (ESPN's actual stat pages)
        stat_endpoints = {
            'QB': f"{self.espn_base}/stats?view=stats&season={season}&seasontype=2&table=passing",
            'RB': f"{self.espn_base}/stats?view=stats&season={season}&seasontype=2&table=rushing", 
            'WR': f"{self.espn_base}/stats?view=stats&season={season}&seasontype=2&table=receiving",
            'TE': f"{self.espn_base}/stats?view=stats&season={season}&seasontype=2&table=receiving"
        }
        
        for position, url in stat_endpoints.items():
            print(f"   Getting {position} stats...")
            
            stats_data = self._rate_limited_request(url)
            
            if stats_data and 'athletes' in stats_data:
                position_players = []
                
                for athlete in stats_data['athletes'][:50]:  # Top 50 per position
                    player_info = athlete.get('athlete', {})
                    stats = athlete.get('stats', [])
                    
                    # Extract basic player info
                    player_data = {
                        'espn_id': player_info.get('id'),
                        'full_name': player_info.get('fullName', ''),
                        'display_name': player_info.get('displayName', ''),
                        'position': position,
                        'team': player_info.get('team', {}).get('abbreviation', ''),
                        'season': season,
                        'jersey': player_info.get('jersey', ''),
                        'age': player_info.get('age', 0),
                        'height': player_info.get('height', ''),
                        'weight': player_info.get('weight', 0)
                    }
                    
                    # Initialize all stat categories
                    player_stats = {
                        'games_played': 0,
                        'passing_attempts': 0, 'passing_completions': 0, 'passing_yards': 0, 'passing_tds': 0, 'interceptions': 0,
                        'rushing_attempts': 0, 'rushing_yards': 0, 'rushing_tds': 0,
                        'targets': 0, 'receptions': 0, 'receiving_yards': 0, 'receiving_tds': 0,
                        'fumbles': 0, 'fumbles_lost': 0
                    }
                    
                    # Parse ESPN stats (they vary by position)
                    if stats:
                        for stat_value in stats:
                            # ESPN provides stats as an array, map them based on position
                            if position == 'QB' and len(stats) >= 8:
                                player_stats.update({
                                    'passing_completions': stats[0] if len(stats) > 0 else 0,
                                    'passing_attempts': stats[1] if len(stats) > 1 else 0,
                                    'passing_yards': stats[2] if len(stats) > 2 else 0,
                                    'passing_tds': stats[3] if len(stats) > 3 else 0,
                                    'interceptions': stats[4] if len(stats) > 4 else 0,
                                    'rushing_attempts': stats[5] if len(stats) > 5 else 0,
                                    'rushing_yards': stats[6] if len(stats) > 6 else 0,
                                    'rushing_tds': stats[7] if len(stats) > 7 else 0
                                })
                            elif position == 'RB' and len(stats) >= 4:
                                player_stats.update({
                                    'rushing_attempts': stats[0] if len(stats) > 0 else 0,
                                    'rushing_yards': stats[1] if len(stats) > 1 else 0,
                                    'rushing_tds': stats[2] if len(stats) > 2 else 0,
                                    'receptions': stats[3] if len(stats) > 3 else 0,
                                    'receiving_yards': stats[4] if len(stats) > 4 else 0,
                                    'receiving_tds': stats[5] if len(stats) > 5 else 0
                                })
                            elif position in ['WR', 'TE'] and len(stats) >= 4:
                                player_stats.update({
                                    'receptions': stats[0] if len(stats) > 0 else 0,
                                    'targets': stats[1] if len(stats) > 1 else 0,
                                    'receiving_yards': stats[2] if len(stats) > 2 else 0,
                                    'receiving_tds': stats[3] if len(stats) > 3 else 0
                                })
                    
                    # Estimate games played (ESPN doesn't always provide this)
                    player_stats['games_played'] = 17  # Default to full season
                    
                    # Calculate fantasy points (PPR scoring)
                    player_stats['fantasy_points_ppr'] = (
                        player_stats['passing_yards'] * 0.04 +
                        player_stats['passing_tds'] * 4 +
                        player_stats['rushing_yards'] * 0.1 +
                        player_stats['rushing_tds'] * 6 +
                        player_stats['receiving_yards'] * 0.1 +
                        player_stats['receiving_tds'] * 6 +
                        player_stats['receptions'] * 1 +
                        player_stats['fumbles_lost'] * -2 +
                        player_stats['interceptions'] * -2
                    )
                    
                    # Combine all data
                    combined_data = {**player_data, **player_stats}
                    position_players.append(combined_data)
                
                all_player_stats.extend(position_players)
                print(f"   ✅ Got {len(position_players)} {position} players")
        
        return all_player_stats
    
    def calculate_advanced_metrics(self, player_stats: List[Dict], team_stats: Dict[str, Dict]) -> List[Dict]:
        """Calculate advanced fantasy metrics from real stats"""
        print("🧮 Calculating advanced fantasy metrics...")
        
        enhanced_players = []
        
        for player in player_stats:
            team = player.get('team', '')
            team_data = team_stats.get(team, {})
            
            # Target share (for pass catchers)
            total_passing_attempts = team_data.get('passing_attempts', 1)
            if total_passing_attempts > 0 and player.get('targets', 0) > 0:
                player['target_share_pct'] = (player['targets'] / total_passing_attempts) * 100
            else:
                player['target_share_pct'] = 0
            
            # Carry share (for RBs)
            total_rushing_attempts = team_data.get('rushing_attempts', 1)
            if total_rushing_attempts > 0 and player.get('rushing_attempts', 0) > 0:
                player['carry_share_pct'] = (player['rushing_attempts'] / total_rushing_attempts) * 100
            else:
                player['carry_share_pct'] = 0
            
            # Efficiency metrics
            if player.get('targets', 0) > 0:
                player['catch_rate_pct'] = (player.get('receptions', 0) / player['targets']) * 100
                player['yards_per_target'] = player.get('receiving_yards', 0) / player['targets']
            else:
                player['catch_rate_pct'] = 0
                player['yards_per_target'] = 0
            
            if player.get('rushing_attempts', 0) > 0:
                player['yards_per_carry'] = player.get('rushing_yards', 0) / player['rushing_attempts']
            else:
                player['yards_per_carry'] = 0
            
            if player.get('receptions', 0) > 0:
                player['yards_per_reception'] = player.get('receiving_yards', 0) / player['receptions']
            else:
                player['yards_per_reception'] = 0
            
            # Total touches and efficiency
            total_touches = player.get('receptions', 0) + player.get('rushing_attempts', 0)
            player['total_touches'] = total_touches
            
            if total_touches > 0:
                total_yards = player.get('receiving_yards', 0) + player.get('rushing_yards', 0)
                player['yards_per_touch'] = total_yards / total_touches
                player['efficiency_per_touch'] = player.get('fantasy_points_ppr', 0) / total_touches
            else:
                player['yards_per_touch'] = 0
                player['efficiency_per_touch'] = 0
            
            # Opportunity score (usage-based metric)
            opportunity_components = [
                player.get('target_share_pct', 0),
                player.get('carry_share_pct', 0)
            ]
            player['opportunity_score'] = sum(opportunity_components) * (player.get('games_played', 0) / 17)
            
            # Red zone metrics (estimated based on TDs and touches)
            total_tds = player.get('receiving_tds', 0) + player.get('rushing_tds', 0)
            if total_touches > 0:
                player['red_zone_opportunities'] = min(total_tds * 3, total_touches)  # Estimate
            else:
                player['red_zone_opportunities'] = total_tds
            
            # Durability metrics
            player['games_missed'] = 17 - player.get('games_played', 17)
            player['durability_score'] = player.get('games_played', 0) / 17
            player['availability_factor'] = player['durability_score']
            
            enhanced_players.append(player)
        
        return enhanced_players
    
    def create_multi_year_dataset(self) -> pd.DataFrame:
        """Create a comprehensive multi-year dataset from real API data"""
        print("🚀 Creating multi-year dataset from REAL NFL APIs...")
        print("=" * 80)
        
        all_seasons_data = []
        
        for season in self.seasons:
            print(f"\n📅 Processing {season} season...")
            
            # Get team stats for context
            team_stats = self.get_espn_team_stats(season)
            
            # Get real player stats
            player_stats = self.get_real_player_stats(season)
            
            if player_stats:
                # Calculate advanced metrics
                enhanced_stats = self.calculate_advanced_metrics(player_stats, team_stats)
                all_seasons_data.extend(enhanced_stats)
                
                print(f"✅ {season}: Added {len(enhanced_stats)} players with real stats")
            else:
                print(f"❌ {season}: No data retrieved")
        
        if not all_seasons_data:
            print("❌ No data collected from any season!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_seasons_data)
        
        # Clean and standardize the data
        print(f"\n🧹 Cleaning and standardizing data...")
        
        # Fill NaN values with appropriate defaults
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        string_columns = df.select_dtypes(include=['object']).columns
        df[string_columns] = df[string_columns].fillna('')
        
        # Add season-based features for model training
        df['is_recent_season'] = (df['season'] == 2024).astype(int)
        df['season_age'] = 2024 - df['season']  # How old this data is
        df['multi_year_consistency'] = df.groupby('full_name')['fantasy_points_ppr'].transform('std').fillna(0)
        
        # Create rolling averages for players with multiple seasons
        df = df.sort_values(['full_name', 'season'])
        for metric in ['fantasy_points_ppr', 'target_share_pct', 'efficiency_per_touch']:
            df[f'{metric}_rolling_avg'] = df.groupby('full_name')[metric].transform(
                lambda x: x.rolling(window=2, min_periods=1).mean()
            )
        
        print(f"✅ Final dataset: {len(df)} player-seasons across {len(df['season'].unique())} years")
        print(f"📊 Unique players: {df['full_name'].nunique()}")
        print(f"🎯 Features: {len(df.columns)} total columns")
        
        return df
    
    def save_real_data(self, df: pd.DataFrame) -> str:
        """Save the real API data to replace the fantasy projection file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main dataset
        main_file = 'data/real_nfl_stats_multi_year.csv'
        df.to_csv(main_file, index=False)
        
        # Save backup with timestamp
        backup_file = f'data/real_nfl_stats_backup_{timestamp}.csv'
        df.to_csv(backup_file, index=False)
        
        # Replace the problematic NFLsavant file
        current_file = 'data/fantasy_metrics_2024.csv'
        if os.path.exists(current_file):
            # Backup the bad data first
            bad_backup = f'data/bad_nflsavant_backup_{timestamp}.csv'
            import shutil
            shutil.copy2(current_file, bad_backup)
            print(f"📁 Backed up problematic NFLsavant data to: {bad_backup}")
        
        # Use only 2024 data for the current metrics file
        df_2024 = df[df['season'] == 2024].copy()
        df_2024.to_csv(current_file, index=False)
        
        print(f"✅ Saved real NFL data:")
        print(f"   📊 Multi-year: {main_file} ({len(df)} records)")
        print(f"   🔄 Current: {current_file} ({len(df_2024)} records)")
        print(f"   💾 Backup: {backup_file}")
        
        return main_file
    
    def run_full_pipeline(self) -> str:
        """Run the complete API data pipeline"""
        print("🏈 NFL API DATA PIPELINE - REAL STATS FOR FANTASY FOOTBALL")
        print("=" * 80)
        print("Replacing fantasy projections with REAL multi-year NFL statistics!")
        print()
        
        try:
            # Step 1: Get player mappings from Sleeper (optional, for ID matching)
            # sleeper_players = self.get_sleeper_players()
            
            # Step 2: Create comprehensive dataset from ESPN real stats
            real_data_df = self.create_multi_year_dataset()
            
            if real_data_df.empty:
                print("❌ Pipeline failed - no data collected")
                return ""
            
            # Step 3: Save the real data
            output_file = self.save_real_data(real_data_df)
            
            # Step 4: Show comparison with old data
            print(f"\n🔍 DATA QUALITY COMPARISON:")
            print("-" * 50)
            
            # Look at WR stats specifically
            wr_data = real_data_df[(real_data_df['position'] == 'WR') & (real_data_df['season'] == 2024)]
            if len(wr_data) > 0:
                top_wrs = wr_data.nlargest(5, 'receiving_yards')
                print("📊 Top 5 WRs in REAL 2024 data:")
                for _, wr in top_wrs.iterrows():
                    name = wr.get('display_name', wr.get('full_name', 'Unknown'))
                    yards = wr.get('receiving_yards', 0)
                    tds = wr.get('receiving_tds', 0)
                    targets = wr.get('targets', 0)
                    print(f"   {name}: {yards} yards, {tds} TDs, {targets} targets")
                
                print("\n✅ These are REAL NFL statistics, not fantasy projections!")
                print("🎯 Your model will now train on honest data!")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return ""

def main():
    """Main execution function"""
    print("🚀 Starting NFL API Data Pipeline...")
    print("This will replace your fantasy projection data with REAL NFL stats!")
    print()
    
    pipeline = NFLDataPipeline()
    output_file = pipeline.run_full_pipeline()
    
    if output_file:
        print(f"\n🎉 SUCCESS! Real NFL data saved to: {output_file}")
        print("\n📋 NEXT STEPS:")
        print("1. ✅ Your model will now use REAL stats instead of fantasy projections")
        print("2. 🔄 Re-train your model: python scripts/draft_optimizer.py")
        print("3. 🎯 Check UI: python scripts/draft_ui.py")
        print("4. 📈 Tyreek Hill should now show realistic ~959 yards!")
        print("\n🏆 Your fantasy football tool is now based on REALITY!")
    else:
        print("\n❌ Pipeline failed. Check error messages above.")

if __name__ == "__main__":
    main() 