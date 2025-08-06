import sys
import os
sys.path.insert(0, '.')

# Suppress matplotlib GUI backends to avoid display issues in serverless
import matplotlib
matplotlib.use('Agg')
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import numpy as np
import json
import os
from scripts.injury_tracker import get_current_injury_data, add_injury_status_to_dataframe

app = Flask(__name__, template_folder='../static')

# Constants
STATE_FILE = 'draft_state.json'

def clean_nan_for_json(data):
    """Convert pandas objects to JSON-safe format"""
    if isinstance(data, pd.DataFrame):
        return data.where(pd.notna(data), None)
    elif isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            try:
                if pd.isna(v):
                    cleaned[k] = None
                else:
                    cleaned[k] = v
            except (TypeError, ValueError):
                # Handle cases where pd.isna() doesn't work
                cleaned[k] = v
        return cleaned
    else:
        return data

# SIMPLE DATA LOADING - ONLY FROM FANTASYPROS CSV
print("🏈 Loading FantasyPros ADP Rankings...")
df = pd.read_csv('data/FantasyPros_2025_Overall_ADP_Rankings.csv', on_bad_lines='skip')

# Clean up column names and data
df = df.rename(columns={
    'Player': 'name',
    'Rank': 'adp_rank', 
    'POS': 'position',
    'Bye': 'bye_week',
    'Team': 'team'
})

# Clean position (remove numbers like WR1 -> WR)
df['position'] = df['position'].str.replace(r'\d+$', '', regex=True)

# Convert adp_rank to numeric
df['adp_rank'] = pd.to_numeric(df['adp_rank'], errors='coerce')

# Simple projected points estimation based on ADP
def estimate_projected_points(row):
    """Simple projected points based on ADP rank and position"""
    adp = row['adp_rank']
    pos = row['position']
    
    if pd.isna(adp):
        return 100  # Default for unranked players
    
    # Position-based base scores
    base_scores = {'QB': 350, 'RB': 280, 'WR': 260, 'TE': 150, 'K': 120, 'DST': 110}
    base = base_scores.get(pos, 100)
    
    # Simple decay: lose points based on ADP rank
    decay_rate = 0.8  # 0.8 points lost per ADP rank
    projected = max(50, base - (adp * decay_rate))
    return round(projected, 1)

df['projected_points'] = df.apply(estimate_projected_points, axis=1)

# Initially mark all as non-rookies (will update after rookie data is loaded)
df['is_rookie'] = False

# Sort by ADP rank
df = df.sort_values('adp_rank', ascending=True)

print(f"✅ Loaded {len(df)} players from FantasyPros ADP Rankings")
print(f"   Positions: {df['position'].value_counts().to_dict()}")

# LOAD ROOKIE DATA SEPARATELY
print("🆕 Loading FantasyPros Rookie Rankings...")
rookie_df = pd.read_csv('data/FantasyPros_2025_Rookies_ALL_Rankings.csv', on_bad_lines='skip')

# Clean up rookie column names
rookie_df = rookie_df.rename(columns={
    'PLAYER NAME': 'name',
    'RK': 'adp_rank',
    'POS': 'position', 
    'TEAM': 'team',
    'AGE': 'age'
})

# Clean position for rookies (remove numbers like RB1 -> RB)
rookie_df['position'] = rookie_df['position'].str.replace(r'\d+$', '', regex=True)

# Convert rookie adp_rank to numeric
rookie_df['adp_rank'] = pd.to_numeric(rookie_df['adp_rank'], errors='coerce')

# Add simple projected points for rookies (lower than veterans)
def estimate_rookie_projected_points(row):
    """Simple projected points for rookies based on rank and position"""
    rank = row['adp_rank']
    pos = row['position']
    
    if pd.isna(rank):
        return 80  # Default for unranked rookies
    
    # Rookie base scores (lower than veterans)
    base_scores = {'QB': 250, 'RB': 200, 'WR': 180, 'TE': 120, 'K': 100, 'DST': 100}
    base = base_scores.get(pos, 150)
    
    # Decay based on rookie rank
    decay_rate = 1.5  # Steeper decay for rookies
    projected = max(30, base - (rank * decay_rate))
    return round(projected, 1)

rookie_df['projected_points'] = rookie_df.apply(estimate_rookie_projected_points, axis=1)
rookie_df['is_rookie'] = True

# Create team-to-bye-week mapping from main ADP data
print("📅 Creating team bye week mapping from ADP data...")
team_bye_mapping = {}
for _, player in df.iterrows():
    team = player.get('team')
    bye_week = player.get('bye_week')
    if team and not pd.isna(bye_week) and team not in team_bye_mapping:
        team_bye_mapping[team] = bye_week

print(f"✅ Created bye week mapping for {len(team_bye_mapping)} teams")

# Apply bye weeks to rookies based on their teams
def get_bye_week_for_rookie(row):
    """Get bye week for rookie based on team"""
    team = row.get('team')
    if team in team_bye_mapping:
        return team_bye_mapping[team]
    return 'TBD'  # Fallback for teams not found

rookie_df['bye_week'] = rookie_df.apply(get_bye_week_for_rookie, axis=1)
bye_weeks_assigned = (rookie_df['bye_week'] != 'TBD').sum()
print(f"✅ Assigned bye weeks to {bye_weeks_assigned} rookies based on team mapping")

# Sort rookies by rank
rookie_df = rookie_df.sort_values('adp_rank', ascending=True)

print(f"✅ Loaded {len(rookie_df)} rookies from FantasyPros Rookie Rankings")
print(f"   Rookie Positions: {rookie_df['position'].value_counts().to_dict()}")

# Mark players as rookies if they appear in the rookie dataset
print("🆕 Marking rookie players in main dataset...")
rookie_names = set(rookie_df['name'].tolist())
df.loc[df['name'].isin(rookie_names), 'is_rookie'] = True
rookies_marked = df['is_rookie'].sum()
print(f"✅ Marked {rookies_marked} players as rookies in main dataset")

# ADD INJURY STATUS TO BOTH DATAFRAMES
print("🏥 Adding injury status to player data...")
try:
    injury_data = get_current_injury_data()
    if not injury_data.empty:
        df = add_injury_status_to_dataframe(df, injury_data)
        rookie_df = add_injury_status_to_dataframe(rookie_df, injury_data)
        print("✅ Injury status added to all players")
    else:
        print("⚠️ No injury data available")
        # Add empty injury columns
        df['injury_status'] = None
        df['injury_description'] = ''
        df['injury_icon'] = ''
        rookie_df['injury_status'] = None
        rookie_df['injury_description'] = ''
        rookie_df['injury_icon'] = ''
except Exception as e:
    print(f"⚠️ Could not load injury data: {e}")
    # Add empty injury columns
    df['injury_status'] = None
    df['injury_description'] = ''
    df['injury_icon'] = ''
    rookie_df['injury_status'] = None
    rookie_df['injury_description'] = ''
    rookie_df['injury_icon'] = ''

def init_state():
    """Initialize draft state"""
    state = {
        'our_team': {
            'QB': None, 'RB1': None, 'RB2': None, 'WR1': None, 'WR2': None,
            'TE': None, 'FLEX': None, 'K': None, 'DST': None,
            'Bench': [None] * 6
        },
        'drafted_by_others': [],
        'team_needs': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'K': 1, 'DST': 1}
    }
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)
    
    print("Initialized new state.")
    return state

def load_state():
    """Load current draft state"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            print("Loaded state from file.")
            return state
        else:
            return init_state()
    except Exception as e:
        print(f"Error loading state: {e}")
        return init_state()

@app.route('/')
def index():
    """Main draft page"""
    state = load_state()
    return render_template('index.html', state=state)

@app.route('/suggest')
def suggest():
    """Suggest draft picks using our proper AI model"""
    try:
        # Check for filtering parameters
        position_filter = request.args.get('position')
        all_available = request.args.get('all_available') == 'true'
        
        # Load draft state
        our_team = []
        opponent_team = []
        drafted_names = []
        
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                our_team = state.get('our_team', [])
                opponent_team = state.get('opponent_team', [])
                
                # Collect all drafted player names
                if isinstance(our_team, dict):
                    for position, player in our_team.items():
                        if isinstance(player, dict) and player.get('name'):
                            drafted_names.append(player['name'])
                        # Handle bench (list of players)
                        elif position == 'Bench' and isinstance(player, list):
                            for bench_player in player:
                                if isinstance(bench_player, dict) and bench_player.get('name'):
                                    drafted_names.append(bench_player['name'])
                elif isinstance(our_team, list):
                    for player in our_team:
                        if isinstance(player, dict) and player.get('name'):
                            drafted_names.append(player['name'])
                
                if isinstance(opponent_team, list):
                    for player in opponent_team:
                        if isinstance(player, dict) and player.get('name'):
                            drafted_names.append(player['name'])
        
        print(f"🚫 Excluding {len(drafted_names)} drafted players from AI recommendations: {drafted_names}")
        
        # Filter out drafted players from ADP data  
        available_df = df[~df['name'].isin(drafted_names)].copy()
        
        # Handle filtering requests
        if all_available:
            print("📋 Returning ALL available players")
            if 'adp_rank' in available_df.columns:
                sorted_df = available_df.sort_values('adp_rank', ascending=True, na_position='last')
            else:
                sorted_df = available_df
            result_df = sorted_df.head(100)  # Limit for performance
            cleaned_df = clean_nan_for_json(result_df)
            return jsonify(cleaned_df.to_dict('records'))
        
        elif position_filter == 'ROOKIE':
            print("🆕 Filtering for ROOKIE players only")
            # Use the globally loaded rookie_df
            try:
                # Filter out any drafted rookies
                available_rookies = rookie_df[~rookie_df['name'].isin(drafted_names)].copy()
                
                # Sort by rookie ADP rank (already loaded properly)
                available_rookies = available_rookies.sort_values('adp_rank', ascending=True)
                
                # Take top 50 rookies
                rookies = available_rookies.head(50)
                cleaned_rookies = clean_nan_for_json(rookies)
                
                print(f"📊 Returning {len(rookies)} available rookies (out of {len(rookie_df)} total)")
                return jsonify(cleaned_rookies.to_dict('records'))
            except Exception as e:
                print(f"❌ Error filtering rookies: {e}")
                return jsonify([])
        
        elif position_filter in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            print(f"🎯 Filtering for {position_filter} players only")
            position_players = available_df[available_df['position'] == position_filter].head(50)
            cleaned_position = clean_nan_for_json(position_players)
            return jsonify(cleaned_position.to_dict('records'))
        
        # Estimate current round
        total_picks = len(drafted_names)
        current_round = (total_picks // 10) + 1
        print(f"📊 Estimated draft round: {current_round}")
        print("🎯 Returning AI recommendations")
        
        # Simple scarcity boost function
        def apply_simple_scarcity_boost(players_df, drafted_team, round_num):
            """Apply a simple position-based scarcity boost"""
            boosted_df = players_df.copy()
            
            # Count how many of each position we already have
            position_counts = {}
            if isinstance(drafted_team, dict):
                for position, player in drafted_team.items():
                    if isinstance(player, dict) and player.get('position'):
                        pos = player['position']
                        position_counts[pos] = position_counts.get(pos, 0) + 1
                    elif position == 'Bench' and isinstance(player, list):
                        for bench_player in player:
                            if isinstance(bench_player, dict) and bench_player.get('position'):
                                pos = bench_player['position']
                                position_counts[pos] = position_counts.get(pos, 0) + 1
            elif isinstance(drafted_team, list):
                for player in drafted_team:
                    if isinstance(player, dict) and player.get('position'):
                        pos = player['position']
                        position_counts[pos] = position_counts.get(pos, 0) + 1
            
            # Define position needs (typical draft strategy)
            position_targets = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'K': 1, 'DST': 1}
            
            # Calculate scarcity boost for each player
            scarcity_boosts = []
            for _, player in boosted_df.iterrows():
                pos = player['position']
                have_count = position_counts.get(pos, 0)
                target_count = position_targets.get(pos, 1)
                
                # Boost if we need more of this position
                if have_count < target_count:
                    # Higher boost for urgent needs
                    scarcity_boost = 1.5 if have_count == 0 else 1.2
                else:
                    scarcity_boost = 0.8  # Lower priority if we have enough
                
                scarcity_boosts.append(scarcity_boost)
            
            boosted_df['scarcity_boost'] = scarcity_boosts
            return boosted_df
        
        # Use the NEW PROPER AI model (no data leakage)
        try:
            from scripts.proper_model_adapter import predict_players, is_model_available
            
            if is_model_available():
                print("🤖 Using PROPER AI model with no data leakage")
                # Get AI predictions for available players
                ai_results = predict_players(available_df)
                
                if ai_results is not None:
                    # Apply scarcity boost to AI predictions
                    enhanced_suggestions = apply_simple_scarcity_boost(ai_results, our_team, current_round)
                    
                    # Calculate boosted score using AI predictions
                    enhanced_suggestions['boosted_score'] = (
                        enhanced_suggestions['ai_prediction'] * enhanced_suggestions['scarcity_boost']
                    )
                    
                    # Sort by boosted AI scores
                    enhanced_suggestions = enhanced_suggestions.sort_values('boosted_score', ascending=False)
                    print(f"✅ Using PROPER AI × Scarcity for {len(enhanced_suggestions)} players")
                else:
                    print("❌ AI prediction failed, using scarcity-only")
                    enhanced_suggestions = apply_simple_scarcity_boost(available_df, our_team, current_round)
                    enhanced_suggestions['boosted_score'] = (
                        enhanced_suggestions['projected_points'] * enhanced_suggestions['scarcity_boost']
                    )
                    enhanced_suggestions = enhanced_suggestions.sort_values('boosted_score', ascending=False)
            else:
                print("❌ Proper AI model not available, using scarcity-based sorting")
                enhanced_suggestions = apply_simple_scarcity_boost(available_df, our_team, current_round)
                enhanced_suggestions['boosted_score'] = (
                    enhanced_suggestions['projected_points'] * enhanced_suggestions['scarcity_boost']
                )
                enhanced_suggestions = enhanced_suggestions.sort_values('boosted_score', ascending=False)
        except Exception as e:
            print(f"❌ Error with AI model: {e}")
            enhanced_suggestions = apply_simple_scarcity_boost(available_df, our_team, current_round)
            enhanced_suggestions['boosted_score'] = (
                enhanced_suggestions['projected_points'] * enhanced_suggestions['scarcity_boost']
            )
            enhanced_suggestions = enhanced_suggestions.sort_values('boosted_score', ascending=False)
        
        # Format for frontend
        formatted_suggestions = []
        for _, suggestion in enhanced_suggestions.head(8).iterrows():  # Top 8 picks
            # Use boosted score as the optimized score
            boosted_score = suggestion.get('boosted_score', suggestion.get('projected_points', 0))
            scarcity_boost = suggestion.get('scarcity_boost', 1.0)
            ai_score = suggestion.get('ai_prediction', 0)
            
            formatted_suggestions.append({
                'name': suggestion['name'],
                'position': suggestion['position'],
                'adp_rank': suggestion.get('adp_rank', None),
                'projected_points': float(suggestion['projected_points']) if not pd.isna(suggestion.get('projected_points', 0)) else 0.0,
                'bye_week': suggestion.get('bye_week', 'Unknown'),
                'team': suggestion.get('team', ''),
                'optimized_score': float(boosted_score) if not pd.isna(boosted_score) else 0.0,
                'ai_score': float(ai_score) if not pd.isna(ai_score) else 0.0,
                'scarcity_boost': round(float(scarcity_boost), 2) if not pd.isna(scarcity_boost) else 1.0
            })
        
        return jsonify(formatted_suggestions)
        
    except Exception as e:
        print(f"❌ Error in suggest route: {e}")
        # Final fallback to basic ADP
        try:
            available_df = df[~df['name'].isin(drafted_names)] if 'drafted_names' in locals() else df
            top_picks = available_df.head(10)
            fallback_suggestions = []
            
            for _, player in top_picks.iterrows():
                fallback_suggestions.append({
                    'name': player['name'],
                    'position': player['position'],
                    'adp_rank': player.get('adp_rank', None),
                    'projected_points': float(player['projected_points']) if not pd.isna(player.get('projected_points', 0)) else 0.0,
                    'bye_week': player.get('bye_week', 'Unknown'),
                    'team': player.get('team', ''),
                    'optimized_score': float(player['projected_points']) if not pd.isna(player.get('projected_points', 0)) else 0.0,
                    'ai_score': 0.0,
                    'scarcity_boost': 1.0
                })
            
            return jsonify(fallback_suggestions)
        except Exception as e2:
            print(f"❌ Even fallback failed: {e2}")
            return jsonify([])

@app.route('/search')
def search():
    """Search for players by name"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])
    
    # Load draft state to exclude drafted players
    drafted_names = []
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            our_team = state.get('our_team', [])
            opponent_team = state.get('opponent_team', [])
            
            # Collect all drafted player names (simplified)
            if isinstance(our_team, dict):
                for position, player in our_team.items():
                    if isinstance(player, dict) and player.get('name'):
                        drafted_names.append(player['name'])
                    elif position == 'Bench' and isinstance(player, list):
                        for bench_player in player:
                            if isinstance(bench_player, dict) and bench_player.get('name'):
                                drafted_names.append(bench_player['name'])
            
            if isinstance(opponent_team, list):
                for player in opponent_team:
                    if isinstance(player, dict) and player.get('name'):
                        drafted_names.append(player['name'])
    
    # Filter out drafted players
    available_df = df[~df['name'].isin(drafted_names)]
    
    # Simple name search
    matches = available_df[available_df['name'].str.contains(query, case=False, na=False)]
    result = clean_nan_for_json(matches.head(20))
    return jsonify(result.to_dict('records'))

@app.route('/draft', methods=['POST'])
def draft():
    """Draft a player"""
    try:
        data = request.get_json()
        player_name = data.get('player')
        
        if not player_name:
            return jsonify({'success': False, 'error': 'No player specified'})
        
        # Find player in either main df or rookie_df
        player_data = None
        
        # Check main df first (case-insensitive)
        player_matches = df[df['name'].str.lower() == player_name.lower()]
        if not player_matches.empty:
            player_data = player_matches.iloc[0].to_dict()
        else:
            # Check rookie df (case-insensitive)
            rookie_matches = rookie_df[rookie_df['name'].str.lower() == player_name.lower()]
            if not rookie_matches.empty:
                player_data = rookie_matches.iloc[0].to_dict()
        
        if not player_data:
            return jsonify({'success': False, 'error': 'Player not found'})
        
        # Load current state
        state = load_state()
        
        # Try to add to starting lineup first
        our_team = state['our_team']
        position = player_data['position']
        
        # Position priority mapping
        position_slots = {
            'QB': ['QB'],
            'RB': ['RB1', 'RB2', 'FLEX'],
            'WR': ['WR1', 'WR2', 'FLEX'],
            'TE': ['TE', 'FLEX'],
            'K': ['K'],
            'DST': ['DST']
        }
        
        # Find available slot
        placed = False
        if position in position_slots:
            for slot in position_slots[position]:
                if our_team.get(slot) is None:
                    our_team[slot] = player_data
                    placed = True
                    break
        
        # If no starting slot available, add to bench
        if not placed:
            bench = our_team.get('Bench', [None] * 6)
            for i, slot in enumerate(bench):
                if slot is None:
                    bench[i] = player_data
                    our_team['Bench'] = bench
                    placed = True
                    break
        
        if not placed:
            return jsonify({'success': False, 'error': 'No available roster spots'})
        
        # Save state
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        
        return jsonify({'success': True, 'player': player_data})
        
    except Exception as e:
        print(f"Draft error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/mark_taken', methods=['POST'])
def mark_taken():
    """Mark a player as taken by another team"""
    try:
        data = request.get_json()
        player_name = data.get('player')
        
        if not player_name:
            return jsonify({'success': False, 'error': 'No player specified'})
        
        # Find player data
        player_data = None
        player_matches = df[df['name'].str.lower() == player_name.lower()]
        if not player_matches.empty:
            player_data = player_matches.iloc[0].to_dict()
        else:
            rookie_matches = rookie_df[rookie_df['name'].str.lower() == player_name.lower()]
            if not rookie_matches.empty:
                player_data = rookie_matches.iloc[0].to_dict()
        
        if not player_data:
            return jsonify({'success': False, 'error': 'Player not found'})
        
        # Load state and add to drafted_by_others
        state = load_state()
        if 'drafted_by_others' not in state:
            state['drafted_by_others'] = []
        
        # Check if already marked
        already_drafted = any(p.get('name') == player_name for p in state['drafted_by_others'] if isinstance(p, dict))
        if not already_drafted:
            state['drafted_by_others'].append(player_data)
        
        # Save state
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        
        return jsonify({'success': True, 'player': player_data})
        
    except Exception as e:
        print(f"Mark taken error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset', methods=['POST'])
def reset():
    """Reset the draft"""
    try:
        init_state()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Reset error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/undo', methods=['POST'])
def undo():
    """Undo last draft action"""
    try:
        state = load_state()
        
        # Find last drafted player and remove
        our_team = state['our_team']
        last_player = None
        
        # Check bench first (reverse order)
        bench = our_team.get('Bench', [])
        for i in range(len(bench) - 1, -1, -1):
            if bench[i] is not None:
                last_player = bench[i]
                bench[i] = None
                our_team['Bench'] = bench
                break
        
        # If bench is empty, check starting lineup
        if last_player is None:
            for position in ['FLEX', 'DST', 'K', 'TE', 'WR2', 'WR1', 'RB2', 'RB1', 'QB']:
                if our_team.get(position) is not None:
                    last_player = our_team[position]
                    our_team[position] = None
                    break
        
        if last_player is None:
            return jsonify({'success': False, 'error': 'No players to undo'})
        
        # Save state
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        
        return jsonify({'success': True, 'undone_player': last_player})
        
    except Exception as e:
        print(f"Undo error: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 