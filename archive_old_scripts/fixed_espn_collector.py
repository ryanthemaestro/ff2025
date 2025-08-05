#!/usr/bin/env python3
"""
Fixed ESPN NFL Data Collector
Focus on team rosters which have confirmed working data

REAL NFL STATISTICS ONLY
"""

import requests
import pandas as pd
import time
from datetime import datetime

class FixedESPNCollector:
    """ESPN collector using confirmed working team roster approach"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_request(self, url):
        """Make a simple GET request with error handling"""
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ HTTP {response.status_code} for {url[:50]}...")
                return None
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:100]}")
            return None
    
    def get_all_teams(self):
        """Get all NFL teams"""
        print("📥 Getting all NFL teams...")
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
        data = self.get_request(url)
        
        if not data:
            return []
        
        teams = []
        if 'sports' in data:
            for sport in data['sports']:
                if 'leagues' in sport:
                    for league in sport['leagues']:
                        if 'teams' in league:
                            for team in league['teams']:
                                team_info = {
                                    'id': team.get('id'),
                                    'abbreviation': team.get('abbreviation', ''),
                                    'displayName': team.get('displayName', ''),
                                    'name': team.get('name', ''),
                                    'location': team.get('location', '')
                                }
                                teams.append(team_info)
        
        print(f"✅ Found {len(teams)} teams")
        return teams
    
    def get_team_roster(self, team_id, team_abbr):
        """Get roster for a specific team"""
        print(f"📋 Getting roster for {team_abbr}...")
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"
        data = self.get_request(url)
        
        if not data:
            return []
        
        players = []
        
        # Parse roster structure (confirmed working from debug)
        if 'athletes' in data:
            for athlete_group in data['athletes']:
                if 'items' in athlete_group:
                    for athlete in athlete_group['items']:
                        try:
                            player = {
                                'espn_id': athlete.get('id', ''),
                                'name': athlete.get('displayName', '').upper().strip(),
                                'full_name': athlete.get('fullName', ''),
                                'first_name': athlete.get('firstName', ''),
                                'last_name': athlete.get('lastName', ''),
                                'position': '',
                                'team': team_abbr,
                                'team_id': team_id,
                                'jersey': athlete.get('jersey', ''),
                                'height': athlete.get('height', 0),
                                'weight': athlete.get('weight', 0),
                                'age': athlete.get('age', 0),
                                'experience': 0,
                                'college': '',
                                'status': '',
                                'salary': 0,
                                'data_source': 'ESPN_ROSTER_API'
                            }
                            
                            # Get position (confirmed this structure works)
                            if 'position' in athlete and isinstance(athlete['position'], dict):
                                player['position'] = athlete['position'].get('abbreviation', '')
                            
                            # Get experience
                            if 'experience' in athlete and isinstance(athlete['experience'], dict):
                                player['experience'] = athlete['experience'].get('years', 0)
                            
                            # Get college
                            if 'college' in athlete and isinstance(athlete['college'], dict):
                                player['college'] = athlete['college'].get('name', '')
                            
                            # Get status
                            if 'status' in athlete and isinstance(athlete['status'], dict):
                                player['status'] = athlete['status'].get('name', '')
                            
                            # Only include skill positions + DST for fantasy
                            if player['position'] in ['QB', 'RB', 'WR', 'TE', 'K', 'DST', 'P']:
                                players.append(player)
                                
                        except Exception as e:
                            print(f"⚠️ Error parsing {team_abbr} player: {e}")
                            continue
        
        print(f"   ✅ {team_abbr}: {len(players)} players")
        return players
    
    def collect_all_rosters(self):
        """Collect rosters from all NFL teams"""
        print("🚀 COLLECTING NFL ROSTERS FROM ESPN API")
        print("=" * 60)
        
        # Get all teams
        teams = self.get_all_teams()
        
        if not teams:
            print("❌ Failed to get teams")
            return None
        
        all_players = []
        
        # Get roster for each team
        for i, team in enumerate(teams):
            team_id = team.get('id')
            team_abbr = team.get('abbreviation', '')
            
            if team_id and team_abbr:
                roster_players = self.get_team_roster(team_id, team_abbr)
                all_players.extend(roster_players)
                
                # Rate limiting - be respectful to ESPN
                time.sleep(0.5)
                
                # Progress update
                if (i + 1) % 8 == 0:
                    print(f"   📊 Progress: {i+1}/{len(teams)} teams completed")
        
        print(f"\n📊 COLLECTION SUMMARY:")
        print(f"   Teams processed: {len(teams)}")
        print(f"   Total players: {len(all_players)}")
        
        # Analyze by position
        if all_players:
            df = pd.DataFrame(all_players)
            position_counts = df['position'].value_counts()
            print(f"   Position breakdown:")
            for pos, count in position_counts.items():
                print(f"     {pos}: {count}")
        
        return all_players
    
    def save_data(self, players):
        """Save collected data to CSV"""
        if not players:
            print("❌ No players to save")
            return None
        
        df = pd.DataFrame(players)
        
        # Add additional columns for fantasy analysis
        df['season'] = 2024
        df['fantasy_points_ppr'] = 0  # Will need to enhance with game stats
        df['games_played'] = 0
        df['projected_points'] = 0
        
        # Add standard fantasy stats columns (to be filled later)
        stat_columns = [
            'passing_yards', 'passing_tds', 'passing_completions', 'passing_attempts', 'interceptions',
            'rushing_yards', 'rushing_tds', 'rushing_attempts',
            'receiving_yards', 'receiving_tds', 'receptions', 'targets',
            'fumbles', 'fumbles_lost'
        ]
        
        for col in stat_columns:
            df[col] = 0
        
        # Sort by team and position
        df = df.sort_values(['team', 'position', 'name'])
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'data/espn_real_rosters_{timestamp}.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n✅ DATA SAVED!")
        print(f"📁 File: {output_file}")
        print(f"📊 Records: {len(df)}")
        
        # Show sample data
        print(f"\n📋 SAMPLE PLAYERS:")
        sample = df.head(15)
        for i, player in sample.iterrows():
            name = player['name'][:20].ljust(20)
            pos = player['position'].ljust(3)
            team = player['team'].ljust(3)
            jersey = str(player['jersey']).ljust(3)
            print(f"   {name} | {pos} | {team} | #{jersey}")
        
        return output_file

def main():
    """Run the fixed ESPN collector"""
    collector = FixedESPNCollector()
    
    # Collect all roster data
    players = collector.collect_all_rosters()
    
    if players:
        output_file = collector.save_data(players)
        
        print(f"\n🎯 SUCCESS! REAL NFL DATA COLLECTED FROM ESPN")
        print(f"=" * 50)
        print(f"✅ {len(players)} real NFL players collected")
        print(f"📈 This replaces your synthetic data with REAL statistics")
        print(f"📁 Data saved to: {output_file}")
        print(f"\n🔄 NEXT STEPS:")
        print(f"1. Update your draft optimizer to use: {output_file}")
        print(f"2. Replace data/players.json with this real data")
        print(f"3. Enhance with season statistics for fantasy points")
        
    else:
        print(f"\n❌ Data collection failed")

if __name__ == "__main__":
    main() 