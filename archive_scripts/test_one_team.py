#!/usr/bin/env python3
"""
Test Single Team Roster Collection
Debug why team roster collection is failing
"""

import requests
import json

def test_single_team():
    """Test collecting roster for just one team"""
    print("🔍 TESTING SINGLE TEAM ROSTER COLLECTION")
    print("=" * 50)
    
    # Test with Chiefs (ID 12)
    team_id = "12"
    team_name = "Chiefs"
    
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"
    
    try:
        print(f"📡 URL: {url}")
        response = requests.get(url, timeout=10)
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            
            # Detailed parsing
            skill_players = []
            
            if 'athletes' in data:
                print(f"📈 Found {len(data['athletes'])} athlete groups")
                
                for group_idx, athlete_group in enumerate(data['athletes']):
                    if 'items' in athlete_group:
                        items = athlete_group['items']
                        print(f"   Group {group_idx}: {len(items)} players")
                        
                        for athlete in items:
                            try:
                                # Extract basic info
                                name = athlete.get('displayName', 'Unknown')
                                position = ''
                                
                                # Get position
                                if 'position' in athlete and isinstance(athlete['position'], dict):
                                    position = athlete['position'].get('abbreviation', '')
                                
                                # Check if skill position
                                if position in ['QB', 'RB', 'WR', 'TE', 'K', 'P']:
                                    player_info = {
                                        'name': name,
                                        'position': position,
                                        'jersey': athlete.get('jersey', ''),
                                        'id': athlete.get('id', '')
                                    }
                                    skill_players.append(player_info)
                                    print(f"     ✅ {name} ({position}) #{athlete.get('jersey', '?')}")
                                
                            except Exception as e:
                                print(f"     ❌ Error parsing player: {e}")
                                continue
                
                print(f"\n📊 RESULTS:")
                print(f"   Total skill players found: {len(skill_players)}")
                
                if skill_players:
                    print(f"\n📋 SKILL PLAYERS:")
                    for player in skill_players:
                        name = player['name'][:20].ljust(20)
                        pos = player['position'].ljust(3)
                        jersey = str(player['jersey']).ljust(3)
                        print(f"   {name} | {pos} | #{jersey}")
                else:
                    print("❌ No skill players found!")
                    
                return len(skill_players) > 0
                
            else:
                print("❌ No 'athletes' key in response")
                print(f"Available keys: {list(data.keys())}")
                return False
                
        else:
            print(f"❌ HTTP {response.status_code}")
            print(f"Response: {response.text[:300]}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_teams_list():
    """Test getting teams list"""
    print(f"\n🔍 TESTING TEAMS LIST")
    print("=" * 30)
    
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
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
                                        'name': team.get('name', '')
                                    }
                                    teams.append(team_info)
            
            print(f"✅ Found {len(teams)} teams")
            
            # Show first few teams
            print(f"\n📋 FIRST 5 TEAMS:")
            for team in teams[:5]:
                print(f"   {team['abbreviation']} - {team['name']} (ID: {team['id']})")
            
            return teams
            
        else:
            print(f"❌ HTTP {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    # Test teams list first
    teams = test_teams_list()
    
    # Test single team roster
    if teams:
        success = test_single_team()
        
        if success:
            print(f"\n✅ SUCCESS! Single team roster collection works")
            print(f"📈 The issue might be in the loop logic")
        else:
            print(f"\n❌ Single team test failed")
    else:
        print(f"\n❌ Teams list failed") 