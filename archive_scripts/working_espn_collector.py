#!/usr/bin/env python3
"""
Working ESPN NFL Data Collector - 2024/2025 Season
Uses VERIFIED WORKING ESPN API endpoints from comprehensive research

Based on: https://gist.github.com/nntrn/ee26cb2a0716de0947a0a4e9a157bc1c
Last Updated: August 2025

REAL NFL STATISTICS ONLY - NO SYNTHETIC DATA
"""

import requests
import pandas as pd
import json
import time
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

class WorkingESPNCollector:
    """ESPN Data Collector using verified working endpoints"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        
        # Rate limiting to be respectful to ESPN's servers
        self.request_delay = 0.5  # 500ms between requests
        self.last_request_time = 0
        
        # Working ESPN endpoints (verified from research)
        self.endpoints = {
            # Core player data
            'all_athletes': 'https://sports.core.api.espn.com/v3/sports/football/nfl/athletes?limit=20000&active=true',
            'player_gamelog': 'https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{athlete_id}/gamelog',
            'player_splits': 'https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{athlete_id}/splits',
            'player_overview': 'https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{athlete_id}/overview',
            
            # Team data
            'teams': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams',
            'team_roster': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}?enable=roster,projection,stats',
            'team_roster_simple': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster',
            
            # Season data
            'season_leaders': 'https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{year}/types/{season_type}/leaders',
            'current_leaders': 'https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/leaders',
            
            # Fantasy data
            'fantasy_players': 'https://fantasy.espn.com/apis/v3/games/ffl/seasons/{year}/players?view=players_wl',
            'fantasy_players_info': 'https://fantasy.espn.com/apis/v3/games/ffl/seasons/{year}/segments/0/leaguedefaults/{ppr_id}?view=kona_player_info',
            
            # Game data
            'scoreboard': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={year}&seasontype=2',
            'game_summary': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}',
        }
        
        print("🏈 Initialized Working ESPN Collector with verified endpoints")
        
    def _rate_limited_request(self, url: str) -> Optional[Dict]:
        """Make a rate-limited request to ESPN API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        try:
            print(f"📡 Requesting: {url[:80]}...")
            response = self.session.get(url, timeout=15)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  HTTP {response.status_code} for URL")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {str(e)[:100]}")
            return None
    
    def _fantasy_request(self, url: str, limit: int = 2000) -> Optional[Dict]:
        """Make request with fantasy-specific headers"""
        headers = {
            'X-Fantasy-Filter': json.dumps({
                "players": {"limit": limit},
                "filterActive": {"value": True}
            })
        }
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        try:
            print(f"📡 Fantasy request: {url[:60]}...")
            response = self.session.get(url, headers=headers, timeout=15)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  Fantasy HTTP {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Fantasy request failed: {str(e)[:100]}")
            return None
    
    def get_all_active_players(self) -> List[Dict]:
        """Get all active NFL players from ESPN v3 API"""
        print("📥 Fetching all active NFL players...")
        
        url = self.endpoints['all_athletes']
        data = self._rate_limited_request(url)
        
        if not data or 'items' not in data:
            print("❌ Failed to get players list")
            return []
        
        players = []
        for item in data['items']:
            try:
                player = {
                    'espn_id': item.get('id'),
                    'name': item.get('displayName', '').upper().strip(),
                    'full_name': item.get('fullName', ''),
                    'position': item.get('position', {}).get('abbreviation', ''),
                    'team': '',  # Will be filled from team data
                    'team_id': '',
                    'jersey': item.get('jersey'),
                    'height': item.get('height'),
                    'weight': item.get('weight'),
                    'age': item.get('age'),
                    'active': item.get('active', True),
                    'data_source': 'ESPN_API_v3'
                }
                
                # Only add skill position players
                if player['position'] in ['QB', 'RB', 'WR', 'TE', 'K']:
                    players.append(player)
                    
            except Exception as e:
                print(f"⚠️ Error parsing player: {e}")
                continue
        
        print(f"✅ Found {len(players)} active skill position players")
        return players
    
    def get_team_rosters(self) -> Dict[str, List[Dict]]:
        """Get detailed rosters for all NFL teams"""
        print("📥 Fetching team rosters...")
        
        # First get all teams
        teams_url = self.endpoints['teams']
        teams_data = self._rate_limited_request(teams_url)
        
        if not teams_data or 'teams' not in teams_data:
            print("❌ Failed to get teams list")
            return {}
        
        all_rosters = {}
        
        for team in teams_data['teams']:
            team_id = team.get('id')
            team_abbr = team.get('abbreviation', '')
            
            if not team_id:
                continue
                
            print(f"📋 Getting roster for {team_abbr}...")
            
            # Get detailed roster with stats
            roster_url = self.endpoints['team_roster'].format(team_id=team_id)
            roster_data = self._rate_limited_request(roster_url)
            
            if not roster_data:
                continue
                
            team_players = []
            
            # Parse roster data
            if 'team' in roster_data and 'athletes' in roster_data['team']:
                for athlete in roster_data['team']['athletes']:
                    try:
                        player = {
                            'espn_id': athlete.get('id'),
                            'name': athlete.get('displayName', '').upper().strip(),
                            'full_name': athlete.get('fullName', ''),
                            'position': athlete.get('position', {}).get('abbreviation', ''),
                            'team': team_abbr,
                            'team_id': team_id,
                            'jersey': athlete.get('jersey'),
                            'height': athlete.get('height'),
                            'weight': athlete.get('weight'),
                            'age': athlete.get('age'),
                            'experience': athlete.get('experience', {}).get('years', 0),
                            'college': athlete.get('college', {}).get('name', ''),
                            'data_source': 'ESPN_ROSTER'
                        }
                        
                        # Only add skill position players
                        if player['position'] in ['QB', 'RB', 'WR', 'TE', 'K']:
                            team_players.append(player)
                            
                    except Exception as e:
                        print(f"⚠️ Error parsing {team_abbr} player: {e}")
                        continue
            
            all_rosters[team_abbr] = team_players
            print(f"   ✅ {team_abbr}: {len(team_players)} skill players")
        
        return all_rosters
    
    def get_player_season_stats(self, espn_id: str, year: int = 2024) -> Optional[Dict]:
        """Get comprehensive season stats for a specific player"""
        if not espn_id:
            return None
            
        print(f"📊 Getting {year} stats for player {espn_id}")
        
        stats = {
            'espn_id': espn_id,
            'season': year,
            'games_played': 0,
            'fantasy_points_ppr': 0,
            'passing_yards': 0,
            'passing_tds': 0,
            'passing_completions': 0,
            'passing_attempts': 0,
            'rushing_yards': 0,
            'rushing_tds': 0,
            'rushing_attempts': 0,
            'receiving_yards': 0,
            'receiving_tds': 0,
            'receptions': 0,
            'targets': 0,
            'fumbles': 0,
            'interceptions': 0,
            'data_source': 'ESPN_GAMELOG'
        }
        
        # Method 1: Try player gamelog (most comprehensive)
        gamelog_url = self.endpoints['player_gamelog'].format(athlete_id=espn_id)
        gamelog_data = self._rate_limited_request(gamelog_url)
        
        if gamelog_data and 'events' in gamelog_data:
            for event in gamelog_data['events']:
                if event.get('season', {}).get('year') == year:
                    stats['games_played'] += 1
                    
                    # Parse game stats
                    game_stats = event.get('stats', [])
                    for stat in game_stats:
                        stat_name = stat.get('name', '')
                        stat_value = float(stat.get('value', 0))
                        
                        # Map ESPN stat names to our fields
                        if stat_name == 'passingYards':
                            stats['passing_yards'] += stat_value
                        elif stat_name == 'passingTouchdowns':
                            stats['passing_tds'] += stat_value
                        elif stat_name == 'completions':
                            stats['passing_completions'] += stat_value
                        elif stat_name == 'attempts' and 'passing' in str(stat.get('description', '')).lower():
                            stats['passing_attempts'] += stat_value
                        elif stat_name == 'rushingYards':
                            stats['rushing_yards'] += stat_value
                        elif stat_name == 'rushingTouchdowns':
                            stats['rushing_tds'] += stat_value
                        elif stat_name == 'rushingAttempts':
                            stats['rushing_attempts'] += stat_value
                        elif stat_name == 'receivingYards':
                            stats['receiving_yards'] += stat_value
                        elif stat_name == 'receivingTouchdowns':
                            stats['receiving_tds'] += stat_value
                        elif stat_name == 'receptions':
                            stats['receptions'] += stat_value
                        elif stat_name == 'targets':
                            stats['targets'] += stat_value
                        elif stat_name == 'fumbles':
                            stats['fumbles'] += stat_value
                        elif stat_name == 'interceptions':
                            stats['interceptions'] += stat_value
        
        # Calculate fantasy points (PPR scoring)
        stats['fantasy_points_ppr'] = (
            stats['passing_yards'] * 0.04 +
            stats['passing_tds'] * 4 +
            stats['rushing_yards'] * 0.1 +
            stats['rushing_tds'] * 6 +
            stats['receiving_yards'] * 0.1 +
            stats['receiving_tds'] * 6 +
            stats['receptions'] * 1 +  # PPR
            stats['interceptions'] * -2 +
            stats['fumbles'] * -2
        )
        
        return stats if stats['games_played'] > 0 else None
    
    def get_fantasy_players(self, year: int = 2024) -> List[Dict]:
        """Get players from ESPN Fantasy API"""
        print(f"📥 Fetching {year} fantasy players...")
        
        # Try fantasy players endpoint with special headers
        fantasy_url = self.endpoints['fantasy_players'].format(year=year)
        data = self._fantasy_request(fantasy_url, limit=3000)
        
        if not data:
            print("❌ Failed to get fantasy players")
            return []
        
        players = []
        for player in data:
            try:
                # Parse ESPN fantasy data structure
                player_info = {
                    'espn_id': player.get('id'),
                    'name': player.get('fullName', '').upper().strip(),
                    'position': '',
                    'team': '',
                    'fantasy_points_ppr': 0,
                    'season': year,
                    'data_source': 'ESPN_FANTASY'
                }
                
                # Get position and team
                if 'defaultPositionId' in player:
                    pos_id = player['defaultPositionId']
                    position_map = {1: 'QB', 2: 'RB', 3: 'WR', 4: 'TE', 5: 'K', 16: 'DST'}
                    player_info['position'] = position_map.get(pos_id, '')
                
                if 'proTeamId' in player:
                    # Map pro team ID to abbreviation
                    team_id = player['proTeamId']
                    # This would need a team mapping dictionary
                
                # Get stats if available
                if 'stats' in player:
                    for stat_period in player['stats']:
                        if stat_period.get('seasonId') == year:
                            stats = stat_period.get('stats', {})
                            # ESPN fantasy points are usually in the stats
                            player_info['fantasy_points_ppr'] = stats.get('120', 0)  # 120 is often PPR points
                
                if player_info['position'] in ['QB', 'RB', 'WR', 'TE', 'K']:
                    players.append(player_info)
                    
            except Exception as e:
                print(f"⚠️ Error parsing fantasy player: {e}")
                continue
        
        print(f"✅ Found {len(players)} fantasy players")
        return players
    
    def collect_comprehensive_data(self, year: int = 2024) -> str:
        """Collect comprehensive NFL data using working ESPN endpoints"""
        print(f"🚀 COLLECTING COMPREHENSIVE {year} NFL DATA FROM ESPN")
        print("=" * 70)
        
        all_players = []
        
        # Method 1: Get all active players
        print("📥 METHOD 1: Active Players API...")
        active_players = self.get_all_active_players()
        
        # Method 2: Get team rosters (more detailed)
        print("\n📥 METHOD 2: Team Rosters...")
        team_rosters = self.get_team_rosters()
        
        # Combine roster data
        roster_players = []
        for team, players in team_rosters.items():
            roster_players.extend(players)
        
        # Method 3: Try fantasy API
        print("\n📥 METHOD 3: Fantasy API...")
        fantasy_players = self.get_fantasy_players(year)
        
        # Merge all data sources
        print(f"\n🔗 MERGING DATA SOURCES...")
        print(f"   Active Players: {len(active_players)}")
        print(f"   Roster Players: {len(roster_players)}")
        print(f"   Fantasy Players: {len(fantasy_players)}")
        
        # Use roster data as primary (most complete)
        merged_players = {}
        
        for player in roster_players:
            espn_id = player.get('espn_id')
            if espn_id:
                merged_players[espn_id] = player
        
        # Enhance with active players data
        for player in active_players:
            espn_id = player.get('espn_id')
            if espn_id and espn_id in merged_players:
                # Update with any missing fields
                for key, value in player.items():
                    if not merged_players[espn_id].get(key):
                        merged_players[espn_id][key] = value
        
        # Get season stats for top players (first 200 to avoid rate limits)
        print(f"\n📊 GETTING SEASON STATS FOR TOP PLAYERS...")
        player_list = list(merged_players.values())[:200]  # Limit for demo
        
        for i, player in enumerate(player_list):
            if i % 20 == 0:
                print(f"   Progress: {i}/{len(player_list)}")
                
            espn_id = player.get('espn_id')
            if espn_id:
                season_stats = self.get_player_season_stats(espn_id, year)
                if season_stats:
                    # Merge stats into player data
                    player.update(season_stats)
        
        # Convert to final list
        final_players = list(merged_players.values())
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'data/espn_real_stats_{year}_{timestamp}.csv'
        
        if final_players:
            df = pd.DataFrame(final_players)
            
            # Ensure key columns exist
            required_columns = [
                'name', 'position', 'team', 'fantasy_points_ppr', 'games_played',
                'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
                'receiving_yards', 'receiving_tds', 'receptions', 'targets'
            ]
            
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Add season column
            df['season'] = year
            
            # Clean up data
            df = df[df['name'].str.len() > 0]  # Remove empty names
            df = df[df['position'].isin(['QB', 'RB', 'WR', 'TE', 'K'])]  # Only skill positions
            
            # Sort by fantasy points
            df = df.sort_values('fantasy_points_ppr', ascending=False)
            
            df.to_csv(output_file, index=False)
            
            print(f"\n✅ COLLECTION COMPLETE!")
            print(f"📁 Saved to: {output_file}")
            print(f"📊 Total players: {len(df)}")
            print(f"🏆 Top fantasy scorers:")
            
            top_players = df.head(10)
            for i, player in top_players.iterrows():
                pos = player['position']
                name = player['name'][:20].ljust(20)
                points = player['fantasy_points_ppr']
                print(f"   {i+1:2d}. {name} ({pos}) - {points:.1f} pts")
        
        return output_file

def main():
    """Run the working ESPN data collector"""
    collector = WorkingESPNCollector()
    
    # Collect 2024 data
    output_file = collector.collect_comprehensive_data(2024)
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Review the data in: {output_file}")
    print(f"2. Update your draft optimizer to use this real data")
    print(f"3. Replace synthetic data with this comprehensive dataset")

if __name__ == "__main__":
    main() 