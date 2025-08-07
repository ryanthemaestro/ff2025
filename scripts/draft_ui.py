import sys
sys.path.insert(0, '.')
from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import numpy as np
import json
import os
from injury_tracker import get_current_injury_data, add_injury_status_to_dataframe

app = Flask(__name__, template_folder='../templates')

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

# Add simple projected points based on ADP rank and position
def estimate_projected_points(row):
    """Simple projected points based on ADP rank and position"""
    adp = row['adp_rank']
    pos = row['position']
    
    if pd.isna(adp):
        return 100  # Default for unranked
    
    # Position base scores
    base_scores = {'QB': 350, 'RB': 280, 'WR': 260, 'TE': 180, 'K': 140, 'DST': 130}
    base = base_scores.get(pos, 200)
    
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
injury_data = get_current_injury_data()
if not injury_data.empty:
    df = add_injury_status_to_dataframe(df, injury_data)
    rookie_df = add_injury_status_to_dataframe(rookie_df, injury_data)
    print("✅ Injury status added to all players")
else:
    # Add empty injury columns if no data
    df['injury_status'] = None
    df['injury_description'] = ""
    df['injury_icon'] = ""
    rookie_df['injury_status'] = None
    rookie_df['injury_description'] = ""
    rookie_df['injury_icon'] = ""
    print("⚠️ No injury data available, added empty injury columns")

def clean_nan_for_json(df):
    """Convert pandas DataFrame NaN values to None for valid JSON"""
    return df.replace([np.nan], [None])

def load_state():
    """Load draft state from file"""
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        
        # Convert back to DataFrames
        global df, rookie_df
        available_df = pd.DataFrame(state['available_df']) if isinstance(state['available_df'], list) else pd.read_json(state['available_df'], orient='records')
        our_team = state['our_team']
        drafted_by_others = state['drafted_by_others']
        team_needs = state['team_needs']
        
        # IMPORTANT: Re-add injury data if missing (state was saved before injury integration)
        if 'injury_status' not in available_df.columns:
            print("🏥 Re-adding injury data to loaded state...")
            injury_data = get_current_injury_data()
            if not injury_data.empty:
                available_df = add_injury_status_to_dataframe(available_df, injury_data)
                print("✅ Injury data re-added to loaded state")
            else:
                # Add empty injury columns if no data
                available_df['injury_status'] = None
                available_df['injury_description'] = ""
                available_df['injury_icon'] = ""
        
        print("Loaded state from file.")
        return available_df, our_team, drafted_by_others, team_needs
        
    except FileNotFoundError:
        print("Initialized new state.")
        # Use the global FantasyPros data (veterans only)
        global df, rookie_df
        available_df = df.copy()
        our_team = {
            'QB': None,
            'RB1': None,
            'RB2': None,
            'WR1': None,
            'WR2': None,
            'TE': None,
            'FLEX': None,
            'K': None,
            'DST': None,
            'Bench': [None] * 6
        }
        drafted_by_others = []
        team_needs = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'K': 1, 'DST': 1}
        
        return available_df, our_team, drafted_by_others, team_needs

def save_state(available_df, our_team, drafted_by_others, team_needs):
    """Save draft state to file"""
    state = {
        'available_df': available_df.to_dict('records'),
        'our_team': our_team,
        'drafted_by_others': drafted_by_others,
        'team_needs': team_needs
    }
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

@app.route('/')
def index():
    available_df, our_team, drafted_by_others, team_needs = load_state()
    
    # Get top 50 available players for initial display
    available_sorted = available_df.head(50)
    cleaned_available = clean_nan_for_json(available_sorted)
    
    context = {
        'available_players': cleaned_available.to_dict('records'),
        'our_team': our_team,
        'team_with_byes': our_team,  # Template expects this name
        'drafted_by_others': drafted_by_others,
        'team_needs': team_needs
    }
    
    response = make_response(render_template('index.html', **context))
    # Add cache-busting headers for development
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

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
            from proper_model_adapter import predict_players, is_model_available
            
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
        
        # Apply Value Over Replacement (VOR)
        try:
            pos_replacement_index = {'QB': 12, 'RB': 28, 'WR': 28, 'TE': 12}
            replacement_points = {}
            for pos, k in pos_replacement_index.items():
                pos_df = enhanced_suggestions[enhanced_suggestions['position'] == pos].sort_values('boosted_score', ascending=False)
                if len(pos_df) >= k:
                    rep = pos_df['boosted_score'].iloc[k - 1]
                elif len(pos_df) > 0:
                    rep = pos_df['boosted_score'].iloc[-1]
                else:
                    rep = 0.0
                replacement_points[pos] = float(rep)
 
            vor_values = []
            final_scores = []
            for _, row in enhanced_suggestions.iterrows():
                pos = row.get('position', '')
                boosted = row.get('boosted_score', 0.0)
                rep = replacement_points.get(pos, 0.0)
                vor = float(boosted) - float(rep)
                final_score = 0.7 * vor + 0.3 * float(boosted)
                vor_values.append(vor)
                final_scores.append(final_score)
 
            enhanced_suggestions = enhanced_suggestions.copy()
            enhanced_suggestions['vor'] = vor_values
            enhanced_suggestions['final_score'] = final_scores
            enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)
        except Exception:
            pass
 
        # Round- and roster-aware position weighting (deprioritize early QBs, delay K/DST)
        try:
            position_counts = {}
            if isinstance(our_team, dict):
                for slot, player in our_team.items():
                    if slot == 'Bench' and isinstance(player, list):
                        for bench_p in player:
                            if isinstance(bench_p, dict) and bench_p.get('position'):
                                p = bench_p['position']
                                position_counts[p] = position_counts.get(p, 0) + 1
                    elif isinstance(player, dict) and player.get('position'):
                        p = player['position']
                        position_counts[p] = position_counts.get(p, 0) + 1

            def position_round_multiplier(pos: str, rnd: int, counts: dict) -> float:
                have_qb = counts.get('QB', 0)
                have_te = counts.get('TE', 0)
                have_rb = counts.get('RB', 0)
                have_wr = counts.get('WR', 0)

                w = 1.0
                if pos in ('K', 'DST'):
                    if rnd <= 12:
                        return 0.2
                    elif rnd <= 14:
                        return 0.5
                    else:
                        return 0.9

                if pos == 'QB':
                    if rnd <= 2:
                        w = 0.3
                    elif rnd == 3:
                        w = 0.5
                    elif rnd in (4, 5):
                        w = 0.7
                    elif rnd in (6, 7):
                        w = 0.85
                    else:
                        w = 1.0
                    if have_qb == 0 and rnd >= 9:
                        w = max(w, 1.05)
                    if have_qb >= 1:
                        w = min(w, 0.6)
                    return w

                if pos in ('RB', 'WR'):
                    need_two = 2
                    have = have_rb if pos == 'RB' else have_wr
                    if rnd <= 2:
                        w = 1.25
                    elif rnd == 3:
                        w = 1.15
                    elif rnd in (4, 5):
                        w = 1.1
                    else:
                        w = 1.0
                    if have < need_two:
                        w *= 1.1
                    return w

                if pos == 'TE':
                    if rnd <= 2:
                        w = 0.85
                    elif rnd == 3:
                        w = 0.9
                    else:
                        w = 1.0
                    if have_te == 0 and rnd >= 6:
                        w = max(w, 1.05)
                    return w

                return w

            round_weights = []
            adjusted_scores = []
            for _, row in enhanced_suggestions.iterrows():
                pos = row.get('position', '')
                base_score = float(row.get('final_score', row.get('boosted_score', 0.0)) or 0.0)
                w = position_round_multiplier(pos, current_round, position_counts)
                round_weights.append(w)
                adjusted_scores.append(base_score * w)

            enhanced_suggestions = enhanced_suggestions.copy()
            enhanced_suggestions['round_weight'] = round_weights
            enhanced_suggestions['final_score'] = adjusted_scores
            enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)
        except Exception:
            pass

        # Format for frontend
        formatted_suggestions = []
        for _, suggestion in enhanced_suggestions.head(8).iterrows():  # Top 8 picks
            boosted_score = suggestion.get('boosted_score', suggestion.get('projected_points', 0))
            scarcity_boost = suggestion.get('scarcity_boost', 1.0)
            ai_score = suggestion.get('ai_prediction', 0)
            final_score = suggestion.get('final_score', boosted_score)
 
            formatted_suggestions.append({
                'name': suggestion['name'],
                'position': suggestion['position'],
                'adp_rank': suggestion.get('adp_rank', None),
                'projected_points': float(suggestion['projected_points']) if not pd.isna(suggestion.get('projected_points', 0)) else 0.0,
                'bye_week': suggestion.get('bye_week', 'Unknown'),
                'team': suggestion.get('team', ''),
                'optimized_score': float(final_score) if not pd.isna(final_score) else (float(boosted_score) if not pd.isna(boosted_score) else 0.0),
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
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        player_name = data.get('player')
        if not player_name:
            return jsonify({'success': False, 'message': 'No player name provided'})
        
        available_df, our_team, drafted_by_others, team_needs = load_state()
        
        # Find the player (case-insensitive)
        player_row = available_df[available_df['name'].str.lower() == player_name.lower()]
        if player_row.empty:
            return jsonify({'success': False, 'message': 'Player not found'})
        
        # Get player data
        player_dict = player_row.iloc[0].to_dict()
        # Convert any numpy/pandas types to native Python types
        for key, value in player_dict.items():
            if pd.isna(value):
                player_dict[key] = None
            elif hasattr(value, 'item'):  # numpy types
                player_dict[key] = value.item()
        
        # Assign to lineup position
        assigned_slot = assign_player_to_lineup(our_team, player_dict)
        if assigned_slot is None:
            return jsonify({'success': False, 'message': 'No available roster spots'})
        
        # Remove from available
        available_df = available_df[available_df['name'] != player_name]
        
        # Save state
        save_state(available_df, our_team, drafted_by_others, team_needs)
        
        return jsonify({
            'success': True, 
            'message': f'Drafted {player_name} to {assigned_slot}',
            'assigned_slot': assigned_slot
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error drafting player: {str(e)}'})

@app.route('/mark_taken', methods=['POST'])
def mark_taken():
    """Mark a player as taken by another team"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        player_name = data.get('player')
        if not player_name:
            return jsonify({'success': False, 'message': 'No player name provided'})
        
        available_df, our_team, drafted_by_others, team_needs = load_state()
        
        # Find the player (case-insensitive)
        player_row = available_df[available_df['name'].str.lower() == player_name.lower()]
        if player_row.empty:
            return jsonify({'success': False, 'message': 'Player not found'})
        
        # Add to drafted by others
        player_dict = player_row.iloc[0].to_dict()
        # Convert any numpy/pandas types to native Python types
        for key, value in player_dict.items():
            if pd.isna(value):
                player_dict[key] = None
            elif hasattr(value, 'item'):  # numpy types
                player_dict[key] = value.item()
        drafted_by_others.append(player_dict)
        
        # Remove from available
        available_df = available_df[available_df['name'] != player_name]
        
        # Save state
        save_state(available_df, our_team, drafted_by_others, team_needs)
        
        return jsonify({'success': True, 'message': f'Marked {player_name} as taken'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error marking player as taken: {str(e)}'})

@app.route('/reset', methods=['POST'])
def reset_draft():
    """Reset the entire draft"""
    try:
        # Initialize fresh state
        initial_state = {
            'available_df': df.to_json(orient='records'),
            'our_team': {
                'QB': None,
                'RB1': None,
                'RB2': None,
                'WR1': None,
                'WR2': None,
                'TE': None,
                'FLEX': None,
                'K': None,
                'DST': None,
                'Bench': [None] * 6
            },
            'drafted_by_others': [],
            'team_needs': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'K': 1, 'DST': 1}
        }
        
        with open(STATE_FILE, 'w') as f:
            json.dump(initial_state, f)
        
        return jsonify({'success': True, 'message': 'Draft reset successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error resetting draft: {str(e)}'})

@app.route('/undo', methods=['POST'])
def undo_player():
    """Remove a player from our team and put them back in available pool"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        player_name = data.get('player')
        if not player_name:
            return jsonify({'success': False, 'message': 'No player name provided'})
        
        available_df, our_team, drafted_by_others, team_needs = load_state()
        
        # Find the player in our team and remove them
        player_found = False
        player_data = None
        
        # Check starting lineup positions
        for position in ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE', 'FLEX', 'K', 'DST']:
            if our_team.get(position) and our_team[position].get('name') == player_name:
                player_data = our_team[position]
                our_team[position] = None
                player_found = True
                break
        
        # Check bench if not found in starting lineup
        if not player_found:
            for i, bench_player in enumerate(our_team.get('Bench', [])):
                if bench_player and bench_player.get('name') == player_name:
                    player_data = our_team['Bench'][i]
                    our_team['Bench'][i] = None
                    player_found = True
                    break
        
        if not player_found:
            return jsonify({'success': False, 'message': 'Player not found in your team'})
        
        # Add player back to available pool
        # Convert player_data back to a DataFrame row format and ensure data types match
        try:
            # Ensure available_df is a DataFrame
            if not isinstance(available_df, pd.DataFrame):
                available_df = pd.DataFrame(available_df)
            
            # Create new row with proper data types
            new_row = pd.DataFrame([player_data])
            
            # Ensure column compatibility
            for col in available_df.columns:
                if col not in new_row.columns:
                    new_row[col] = None
            for col in new_row.columns:
                if col not in available_df.columns:
                    available_df[col] = None
            
            # Add the player back
            available_df = pd.concat([available_df, new_row], ignore_index=True)
            
            # Sort by ADP rank to maintain proper order
            if 'adp_rank' in available_df.columns:
                available_df = available_df.sort_values('adp_rank', ascending=True, na_last=True)
                available_df = available_df.reset_index(drop=True)
                
            print(f"✅ Added {player_name} back to available players pool")
            
        except Exception as e:
            print(f"⚠️ Error adding player back to pool: {e}")
            # Still remove from team even if adding back fails
            pass
        
        # Save state
        save_state(available_df, our_team, drafted_by_others, team_needs)
        
        return jsonify({'success': True, 'message': f'Removed {player_name} from your team'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error removing player: {str(e)}'})

def assign_player_to_lineup(our_team, player_dict):
    """Assign player to the first available appropriate position"""
    position = player_dict['position']
    
    # Position priority mapping
    position_slots = {
        'QB': ['QB'],
        'RB': ['RB1', 'RB2', 'FLEX'],
        'WR': ['WR1', 'WR2', 'FLEX'],
        'TE': ['TE', 'FLEX'],
        'K': ['K'],
        'DST': ['DST']
    }
    
    # Try to assign to appropriate starting slots first
    if position in position_slots:
        for slot in position_slots[position]:
            if our_team.get(slot) is None:
                our_team[slot] = player_dict
                return slot
    
    # If no starting slot available, put on bench
    for i, bench_slot in enumerate(our_team['Bench']):
        if bench_slot is None:
            our_team['Bench'][i] = player_dict
            return f'Bench {i+1}'
    
    # If bench is full, return None (shouldn't happen in normal draft)
    return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)