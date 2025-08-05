#!/usr/bin/env python3
"""
Test User Roster Scenario
==========================

Simulate the user's exact roster to test team construction logic:
- RB1: Saquon Barkley
- RB2: Jahmyr Gibbs 
- WR1: Ja'Marr Chase
- WR2: Nico Collins
- FLEX: Bijan Robinson (RB)
- Bench: Derrick Henry (RB)
- Missing: QB, TE, K, DST

Should prioritize: QB, TE, K, DST over more RBs
"""

import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.draft_ui import calculate_team_needs
from scripts.draft_optimizer import optimize_starting_lineup_first, prepare_data, load_players, load_projections

def test_user_roster_scenario():
    """Test the exact user roster scenario"""
    
    print("🧪 TESTING USER'S EXACT ROSTER SCENARIO")
    print("=" * 60)
    
    # User's current roster
    user_roster = {
        'QB': None,
        'RB1': 'Saquon Barkley', 
        'RB2': 'Jahmyr Gibbs',
        'WR1': "Ja'Marr Chase",
        'WR2': 'Nico Collins', 
        'TE': None,
        'FLEX': 'Bijan Robinson',  # This is an RB
        'K': None,
        'DST': None,
        'Bench': ['Derrick Henry', None, None, None, None, None]  # RB on bench
    }
    
    print("👥 USER'S ROSTER:")
    print(f"   RBs: Saquon Barkley, Jahmyr Gibbs, Bijan Robinson (FLEX), Derrick Henry (Bench)")
    print(f"   WRs: Ja'Marr Chase, Nico Collins") 
    print(f"   Missing: QB, TE, K, DST")
    print()
    
    # Load data
    print("📊 Loading player data...")
    players = load_players()
    projections = load_projections()
    df = prepare_data(players, projections)
    
    # Calculate team needs
    team_needs = calculate_team_needs(user_roster, df)
    print(f"🎯 CALCULATED TEAM NEEDS: {team_needs}")
    
    # Get suggestions  
    print(f"\n🏈 GETTING SUGGESTIONS FOR USER'S ROSTER...")
    
    # Remove drafted players from available pool
    drafted_players = ['Saquon Barkley', 'Jahmyr Gibbs', "Ja'Marr Chase", 'Nico Collins', 'Bijan Robinson', 'Derrick Henry']
    available_df = df[~df['name'].isin(drafted_players)].copy()
    
    print(f"📊 Available players: {len(available_df)} (removed {len(drafted_players)} drafted)")
    
    # Get optimized suggestions
    suggestions = optimize_starting_lineup_first(
        available_df=available_df,
        current_roster=user_roster,
        team_needs=team_needs,
        round_num=5,  # Mid-draft
        league_size=12
    )
    
    print(f"\n🏆 TOP 10 SUGGESTIONS FOR USER'S ROSTER:")
    print("-" * 60)
    
    for i, suggestion in enumerate(suggestions[:10], 1):
        name = suggestion['name']
        pos = suggestion['position'] 
        score = suggestion['optimized_score']
        print(f"{i:2d}. {name:<20} ({pos:3s}) - Score: {score:.1f}")
    
    # Analyze position distribution in top 10
    positions = [s['position'] for s in suggestions[:10]]
    print(f"\n📊 POSITION BREAKDOWN (TOP 10):")
    for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
        count = positions.count(pos)
        if count > 0:
            print(f"   {pos}: {count} players")
    
    # Check if essential positions are prioritized
    top_5_positions = [s['position'] for s in suggestions[:5]]
    essential_in_top_5 = len([p for p in top_5_positions if p in ['QB', 'TE', 'K', 'DST']])
    
    print(f"\n🎯 ANALYSIS:")
    print(f"   Essential positions in top 5: {essential_in_top_5}/5")
    print(f"   RBs in top 5: {top_5_positions.count('RB')}/5")
    
    if essential_in_top_5 >= 2:
        print(f"   ✅ GOOD: System prioritizes missing essential positions!")
    else:
        print(f"   ❌ ISSUE: System still favoring skill positions over team needs!")
    
    return suggestions

if __name__ == "__main__":
    test_user_roster_scenario() 