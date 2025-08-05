#!/usr/bin/env python3
"""
Test SportsDataIO NFL API
Check what endpoints are available and test data quality

Based on: https://sportsdata.io/developers/api-documentation/nfl
"""

import requests
import json
import time
from datetime import datetime

class SportsDataIOTester:
    def __init__(self, api_key=None):
        self.base_url = "https://api.sportsdata.io/v3/nfl"
        self.headers = {}
        if api_key:
            self.headers['Ocp-Apim-Subscription-Key'] = api_key
        
    def test_endpoint(self, endpoint, params=None):
        """Test a single endpoint"""
        url = f"{self.base_url}/{endpoint}"
        
        print(f"\n🔍 Testing: {endpoint}")
        print(f"📡 URL: {url}")
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            print(f"📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success! Data type: {type(data)}")
                
                if isinstance(data, list) and len(data) > 0:
                    print(f"📈 Items found: {len(data)}")
                    print(f"🎯 Sample item keys: {list(data[0].keys())[:10]}")
                    return data[:5]  # Return first 5 items
                elif isinstance(data, dict):
                    print(f"📈 Dict keys: {list(data.keys())[:10]}")
                    return data
                else:
                    print(f"📈 Data: {str(data)[:200]}...")
                    return data
                    
            elif response.status_code == 401:
                print("❌ 401 Unauthorized - Need valid API key")
                return None
            elif response.status_code == 403:
                print("❌ 403 Forbidden - Access denied or subscription required")
                return None
            else:
                print(f"❌ Error: {response.status_code} - {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
            return None

def test_free_endpoints():
    """Test endpoints that might be free or have free tiers"""
    tester = SportsDataIOTester()
    
    print("🏈 TESTING SPORTSDATA.IO NFL API")
    print("=" * 50)
    
    # Test common endpoints that might be free
    endpoints_to_test = [
        # Basic info (often free)
        "teams",
        "players", 
        "stadiums",
        "timeframes/2024",
        
        # Season data (might be free)
        "standings/2024",
        "schedules/2024",
        
        # Player stats (likely premium)
        "playerstats/2024",
        "playergamestats/2024/1",
        
        # Fantasy data (likely premium)
        "fantasyleaders/2024/1/QB",
        "fantasydefensegame/2024/1",
        
        # Projections (likely premium)  
        "playergameprojectionstats/2024/1",
        "fantasyplayerprojectionstats/2024",
    ]
    
    results = {}
    
    for endpoint in endpoints_to_test:
        result = tester.test_endpoint(endpoint)
        results[endpoint] = result
        time.sleep(1)  # Be respectful with rate limiting
    
    return results

def test_with_api_key(api_key):
    """Test with a provided API key"""
    tester = SportsDataIOTester(api_key)
    
    print(f"\n🔑 TESTING WITH API KEY: {api_key[:10]}...")
    print("=" * 50)
    
    # Test key endpoints for fantasy football
    endpoints_to_test = [
        "players",  # Get all players
        "playerstats/2024",  # Season stats
        "fantasyplayerprojectionstats/2024",  # Fantasy projections
        "playergameprojectionstats/2024/1",  # Weekly projections
    ]
    
    results = {}
    
    for endpoint in endpoints_to_test:
        result = tester.test_endpoint(endpoint)
        results[endpoint] = result
        if result:
            # Show sample data structure
            if isinstance(result, list) and len(result) > 0:
                sample = result[0]
                print(f"   🎯 Sample player: {sample.get('Name', 'Unknown')}")
                if 'FantasyPoints' in sample:
                    print(f"   💰 Fantasy Points: {sample['FantasyPoints']}")
                if 'PassingYards' in sample:
                    print(f"   🏈 Passing Yards: {sample['PassingYards']}")
        time.sleep(1)
    
    return results

def check_discovery_lab():
    """Check if there's a Discovery Lab endpoint"""
    print("\n🔬 CHECKING DISCOVERY LAB...")
    print("=" * 50)
    
    # Try the Discovery Lab URL mentioned in research
    discovery_urls = [
        "https://api.sportsdata.io/discovery/v3/nfl/players",
        "https://discovery.sportsdata.io/v3/nfl/players", 
        "https://api.fantasydata.com/v3/nfl/players",
    ]
    
    for url in discovery_urls:
        print(f"\n🔍 Trying: {url}")
        try:
            response = requests.get(url, timeout=10)
            print(f"📊 Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success! Found {len(data) if isinstance(data, list) else 'data'}")
                return data
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
    
    return None

if __name__ == "__main__":
    print("🏈 SPORTSDATA.IO NFL API TESTER")
    print("🎯 Testing for NFL player statistics and fantasy points")
    print("📅 " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("\n")
    
    # Test free endpoints first
    print("PHASE 1: Testing potentially free endpoints...")
    free_results = test_free_endpoints()
    
    # Check Discovery Lab
    print("\nPHASE 2: Checking Discovery Lab...")
    discovery_result = check_discovery_lab()
    
    # Prompt for API key if user has one
    print("\nPHASE 3: Testing with API key (if available)...")
    print("📝 To get an API key, visit: https://sportsdata.io/")
    print("💡 They offer free trials and Discovery Lab for personal projects")
    
    api_key = input("\n🔑 Enter your SportsDataIO API key (or press Enter to skip): ").strip()
    
    if api_key:
        key_results = test_with_api_key(api_key)
    else:
        print("⏭️  Skipping API key tests")
    
    print("\n" + "="*60)
    print("📋 SUMMARY:")
    print("="*60)
    
    working_endpoints = []
    for endpoint, result in free_results.items():
        if result:
            working_endpoints.append(endpoint)
    
    if working_endpoints:
        print(f"✅ Working endpoints: {len(working_endpoints)}")
        for endpoint in working_endpoints:
            print(f"   • {endpoint}")
    else:
        print("❌ No free endpoints found - API key likely required")
    
    if discovery_result:
        print("✅ Discovery Lab: Available")
    else:
        print("❌ Discovery Lab: Not found")
    
    print("\n🎯 NEXT STEPS:")
    if working_endpoints:
        print("✅ Some endpoints working - proceed with integration")
    else:
        print("🔑 Sign up for SportsDataIO API key at: https://sportsdata.io/")
        print("💡 Try their Discovery Lab for personal projects")
        print("🆓 Look for free trial options") 