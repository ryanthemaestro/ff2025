#!/usr/bin/env python3
"""
Debug ESPN Players API Response
Inspect the actual data structure to understand why we're getting 0 players
"""

import requests
import json

def debug_players_api():
    """Debug the players API response"""
    print("🔍 DEBUGGING ESPN PLAYERS API")
    print("=" * 50)
    
    # Test the athletes endpoint
    url = "https://sports.core.api.espn.com/v3/sports/football/nfl/athletes?limit=10&active=true"
    
    try:
        print(f"📡 URL: {url}")
        response = requests.get(url, timeout=10)
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Response type: {type(data)}")
            
            # Show top-level keys
            if isinstance(data, dict):
                print(f"📋 Top-level keys: {list(data.keys())}")
                
                # Look for items
                if 'items' in data:
                    items = data['items']
                    print(f"📈 Items count: {len(items)}")
                    
                    if len(items) > 0:
                        print(f"\n🔍 FIRST PLAYER STRUCTURE:")
                        first_player = items[0]
                        print(json.dumps(first_player, indent=2)[:1000] + "...")
                        
                        # Check specific fields we're looking for
                        print(f"\n📊 FIELD ANALYSIS:")
                        print(f"   ID: {first_player.get('id', 'MISSING')}")
                        print(f"   Name: {first_player.get('displayName', 'MISSING')}")
                        print(f"   Position: {first_player.get('position', 'MISSING')}")
                        print(f"   Team: {first_player.get('team', 'MISSING')}")
                        print(f"   Active: {first_player.get('active', 'MISSING')}")
                        
                        # Check position structure
                        if 'position' in first_player:
                            pos_data = first_player['position']
                            print(f"   Position structure: {pos_data}")
                            if isinstance(pos_data, dict):
                                print(f"   Position abbreviation: {pos_data.get('abbreviation', 'MISSING')}")
                        
                        # Check team structure  
                        if 'team' in first_player:
                            team_data = first_player['team']
                            print(f"   Team structure: {team_data}")
                            if isinstance(team_data, dict):
                                print(f"   Team abbreviation: {team_data.get('abbreviation', 'MISSING')}")
                    else:
                        print("❌ No items in response")
                else:
                    print("❌ No 'items' key in response")
                    print(f"Available keys: {list(data.keys())}")
            else:
                print(f"❌ Response is not a dict: {type(data)}")
                
        else:
            print(f"❌ HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def debug_team_roster():
    """Debug a specific team's roster"""
    print(f"\n🔍 DEBUGGING TEAM ROSTER")
    print("=" * 30)
    
    # Try getting a specific team roster (Chiefs = 12)
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/12/roster"
    
    try:
        print(f"📡 URL: {url}")
        response = requests.get(url, timeout=10)
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Response type: {type(data)}")
            
            # Show structure
            if isinstance(data, dict):
                print(f"📋 Top-level keys: {list(data.keys())}")
                
                if 'athletes' in data:
                    athletes = data['athletes']
                    print(f"📈 Athletes groups: {len(athletes)}")
                    
                    for i, group in enumerate(athletes):
                        if 'items' in group:
                            items = group['items']
                            print(f"   Group {i}: {len(items)} players")
                            
                            if len(items) > 0:
                                first_player = items[0]
                                pos = first_player.get('position', {}).get('abbreviation', '??')
                                name = first_player.get('displayName', 'Unknown')
                                print(f"     Sample: {name} ({pos})")
                        
                        if i == 0:  # Show detailed structure for first group
                            print(f"\n🔍 FIRST ATHLETE STRUCTURE:")
                            if 'items' in group and len(group['items']) > 0:
                                first_athlete = group['items'][0]
                                print(json.dumps(first_athlete, indent=2)[:800] + "...")
        else:
            print(f"❌ HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_players_api()
    debug_team_roster() 