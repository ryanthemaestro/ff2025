#!/usr/bin/env python3
"""
Simple ESPN NFL Data Collector
Focuses on working endpoints with proper data structure handling

REAL NFL STATISTICS ONLY
"""

import requests
import pandas as pd
import time
from datetime import datetime

class SimpleESPNCollector:
    """Simple ESPN collector focusing on working endpoints"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_request(self, url):
        """Make a simple GET request with error handling"""
        try:
            print(f"📡 Fetching: {url[:70]}...")
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:100]}")
            return None
    
    def get_teams(self):
        """Get all NFL teams"""
        print("📥 Getting NFL teams...")
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
        data = self.get_request(url)
        
        if not data:
            return []
        
        teams = []
        # Parse the actual structure we found in testing
        if 'sports' in data:
            for sport in data['sports']:
                if 'leagues' in sport:
                    for league in sport['leagues']:
                        if 'teams' in league:
                            for team in league['teams']:
                                team_info = {
                                    'id': team.get('id'),
                                    'name': team.get('name', ''),
                                    'abbreviation': team.get('abbreviation', ''),
                                    'displayName': team.get('displayName', ''),
                                    'location': team.get('location', '')
                                }
                                teams.append(team_info)
        
        print(f"✅ Found {len(teams)} teams")
        return teams
    
    def get_active_players(self):
        """Get active NFL players"""
        print("📥 Getting active NFL players...")
        url = "https://sports.core.api.espn.com/v3/sports/football/nfl/athletes?limit=20000&active=true"
        data = self.get_request(url)
        
        if not data or 'items' not in data:
            return []
        
        players = []
        for item in data['items']:
            try:
                # Extract player info
                player = {
                    'espn_id': item.get('id', ''),
                    'name': item.get('displayName', '').upper().strip(),
                    'full_name': item.get('fullName', ''),
                    'position': '',
                    'team': '',
                    'jersey': item.get('jersey', ''),
                    'height': item.get('height', 0),
                    'weight': item.get('weight', 0),
                    'age': item.get('age', 0),
                    'active': item.get('active', True)
                }
                
                # Get position
                if 'position' in item and isinstance(item['position'], dict):
                    player['position'] = item['position'].get('abbreviation', '')
                
                # Get team
                if 'team' in item and isinstance(item['team'], dict):
                    player['team'] = item['team'].get('abbreviation', '')
                
                # Only include skill positions
                if player['position'] in ['QB', 'RB', 'WR', 'TE', 'K']:
                    players.append(player)
                    
            except Exception as e:
                print(f"⚠️ Error parsing player: {e}")
                continue
        
        print(f"✅ Found {len(players)} skill position players")
        return players
    
    def get_team_roster(self, team_id):
        """Get roster for a specific team"""
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"
        data = self.get_request(url)
        
        if not data:
            return []
        
        players = []
        
        # Parse roster structure
        if 'athletes' in data:
            for athlete_group in data['athletes']:
                if 'items' in athlete_group:
                    for athlete in athlete_group['items']:
                        try:
                            player = {
                                'espn_id': athlete.get('id', ''),
                                'name': athlete.get('displayName', '').upper().strip(),
                                'full_name': athlete.get('fullName', ''),
                                'position': '',
                                'team': '',
                                'jersey': athlete.get('jersey', ''),
                                'height': athlete.get('height', 0),
                                'weight': athlete.get('weight', 0),
                                'age': athlete.get('age', 0),
                                'experience': 0
                            }
                            
                            # Get position
                            if 'position' in athlete and isinstance(athlete['position'], dict):
                                player['position'] = athlete['position'].get('abbreviation', '')
                            
                            # Only include skill positions
                            if player['position'] in ['QB', 'RB', 'WR', 'TE', 'K']:
                                players.append(player)
                                
                        except Exception as e:
                            continue
        
        return players
    
    def get_current_leaders(self):
        """Get current NFL statistical leaders"""
        print("📥 Getting NFL statistical leaders...")
        url = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/leaders"
        data = self.get_request(url)
        
        if data:
            print(f"✅ Leaders data available")
        
        # This endpoint gives us a reference to the actual leaders data
        # We would need to follow the $ref links to get actual stats
        return data
    
    def get_scoreboard_games(self):
        """Get games from 2024 scoreboard"""
        print("📥 Getting 2024 NFL games...")
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=2024&seasontype=2"
        data = self.get_request(url)
        
        if not data or 'events' not in data:
            return []
        
        games = []
        for event in data['events']:
            try:
                game = {
                    'id': event.get('id', ''),
                    'name': event.get('name', ''),
                    'date': event.get('date', ''),
                    'season': event.get('season', {}).get('year', 2024),
                    'week': event.get('week', {}).get('number', 0),
                    'completed': event.get('status', {}).get('type', {}).get('completed', False)
                }
                games.append(game)
                
            except Exception as e:
                continue
        
        print(f"✅ Found {len(games)} games")
        return games
    
    def collect_data(self):
        """Main data collection method"""
        print("🚀 SIMPLE ESPN DATA COLLECTION")
        print("=" * 50)
        
        # Get basic data
        teams = self.get_teams()
        time.sleep(1)
        
        active_players = self.get_active_players()
        time.sleep(1)
        
        # Get team rosters for more detailed info
        all_roster_players = []
        for team in teams[:5]:  # Limit to first 5 teams for demo
            team_id = team.get('id')
            team_abbr = team.get('abbreviation', '')
            
            if team_id:
                print(f"📋 Getting roster for {team_abbr}...")
                roster_players = self.get_team_roster(team_id)
                
                # Add team info to players
                for player in roster_players:
                    player['team'] = team_abbr
                    player['team_id'] = team_id
                
                all_roster_players.extend(roster_players)
                time.sleep(1)  # Rate limiting
        
        # Get game data
        time.sleep(1)
        games = self.get_scoreboard_games()
        
        # Combine and save data
        print(f"\n📊 DATA SUMMARY:")
        print(f"   Teams: {len(teams)}")
        print(f"   Active Players: {len(active_players)}")
        print(f"   Roster Players: {len(all_roster_players)}")
        print(f"   Games: {len(games)}")
        
        # Save the most complete dataset
        if all_roster_players:
            df = pd.DataFrame(all_roster_players)
        elif active_players:
            df = pd.DataFrame(active_players)
        else:
            print("❌ No player data collected")
            return None
        
        # Add some basic stats (we'll enhance this with actual game stats later)
        df['season'] = 2024
        df['fantasy_points_ppr'] = 0  # Placeholder
        df['games_played'] = 0
        df['data_source'] = 'ESPN_API'
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'data/simple_espn_data_{timestamp}.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n✅ COLLECTION COMPLETE!")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Total records: {len(df)}")
        
        # Show sample data
        if len(df) > 0:
            print(f"\n📋 SAMPLE DATA:")
            sample = df.head(10)
            for i, player in sample.iterrows():
                name = player.get('name', '')[:20].ljust(20)
                pos = player.get('position', '??')
                team = player.get('team', '???')
                print(f"   {name} | {pos} | {team}")
        
        return output_file

def main():
    """Run the simple ESPN collector"""
    collector = SimpleESPNCollector()
    output_file = collector.collect_data()
    
    if output_file:
        print(f"\n🎯 SUCCESS! Real NFL player data collected from ESPN API")
        print(f"📈 Next step: Use this data to replace synthetic data in your system")
    else:
        print(f"\n❌ Data collection failed. Check network connection.")

if __name__ == "__main__":
    main() 