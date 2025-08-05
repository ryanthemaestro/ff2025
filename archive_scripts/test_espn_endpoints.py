#!/usr/bin/env python3
"""
Test ESPN API Endpoints
Quick verification that the working endpoints are accessible
"""

import requests
import json
import time

def test_endpoint(name, url, headers=None):
    """Test a single endpoint"""
    print(f"\n🔍 Testing {name}...")
    print(f"📡 URL: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Data keys: {list(data.keys())[:5]}")
            
            # Show some sample data
            if isinstance(data, dict):
                if 'items' in data:
                    print(f"📈 Items found: {len(data['items'])}")
                elif 'teams' in data:
                    print(f"🏈 Teams found: {len(data['teams'])}")
                elif 'events' in data:
                    print(f"🎮 Events found: {len(data['events'])}")
            elif isinstance(data, list):
                print(f"📋 List items: {len(data)}")
                
            return True
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")
        return False

def main():
    """Test key ESPN endpoints"""
    print("🚀 TESTING ESPN API ENDPOINTS")
    print("=" * 50)
    
    # Test basic endpoints from our research
    endpoints = [
        ("All NFL Teams", "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"),
        ("All Active Athletes", "https://sports.core.api.espn.com/v3/sports/football/nfl/athletes?limit=100&active=true"),
        ("Current NFL Leaders", "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/leaders"),
        ("2024 NFL Scoreboard", "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=2024&seasontype=2"),
    ]
    
    results = []
    
    for name, url in endpoints:
        success = test_endpoint(name, url)
        results.append((name, success))
        time.sleep(1)  # Rate limit
    
    # Test fantasy endpoint with special header
    print(f"\n🔍 Testing Fantasy Players (with special header)...")
    fantasy_url = "https://fantasy.espn.com/apis/v3/games/ffl/seasons/2024/players?view=players_wl"
    fantasy_headers = {
        'X-Fantasy-Filter': json.dumps({
            "players": {"limit": 50},
            "filterActive": {"value": True}
        })
    }
    
    success = test_endpoint("Fantasy Players", fantasy_url, fantasy_headers)
    results.append(("Fantasy Players", success))
    
    # Summary
    print(f"\n📊 ENDPOINT TEST SUMMARY")
    print("=" * 30)
    
    successful = 0
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
        if success:
            successful += 1
    
    print(f"\n🏆 Success Rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    if successful > 0:
        print(f"\n🎯 Good news! {successful} endpoints are working.")
        print(f"📈 You can proceed with the full data collection.")
    else:
        print(f"\n⚠️  All endpoints failed. Check your internet connection.")

if __name__ == "__main__":
    main() 