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

# Remove is_rookie flag since this is for veterans only
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
rookie_df['bye_week'] = 'TBD'  # Rookies don't have bye weeks assigned yet

# Sort rookies by rank
rookie_df = rookie_df.sort_values('adp_rank', ascending=True)

print(f"✅ Loaded {len(rookie_df)} rookies from FantasyPros Rookie Rankings")
print(f"   Rookie Positions: {rookie_df['position'].value_counts().to_dict()}")

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
    """Main suggestion endpoint"""
    global rookie_df  # Access the global rookie DataFrame
    available_df, our_team, drafted_by_others, team_needs = load_state()
    
    position_filter = request.args.get('position')
    all_available = request.args.get('all_available') == 'true'
    
    if all_available:
        print("📋 Returning ALL available players")
        # First sort by ADP rank, then return top 100
        if 'adp_rank' in available_df.columns:
            sorted_df = available_df.sort_values('adp_rank', ascending=True, na_position='last')
        else:
            sorted_df = available_df
        result_df = sorted_df.head(100)  # Limit for performance
        cleaned_df = clean_nan_for_json(result_df)
        return jsonify(cleaned_df.to_dict('records'))
    
    elif position_filter == 'ROOKIE':
        print("🆕 Filtering for ROOKIE players only")
        rookies = rookie_df.head(50) # Use rookie_df directly
        cleaned_rookies = clean_nan_for_json(rookies)
        return jsonify(cleaned_rookies.to_dict('records'))
    
    elif position_filter in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
        print(f"🎯 Filtering for {position_filter} players only")
        position_players = available_df[available_df['position'] == position_filter].head(50)
        cleaned_position = clean_nan_for_json(position_players)
        return jsonify(cleaned_position.to_dict('records'))
    
    else:
        print("🎯 Returning AI recommendations from trained CatBoost model")
        # Use ONLY available players (excludes drafted players) for AI recommendations
        try:
            # Get list of drafted player names to exclude from AI suggestions
            drafted_names = []
            
            # Extract all drafted player names from our_team
            if isinstance(our_team, dict):
                for position, player in our_team.items():
                    if position != 'Bench' and isinstance(player, dict) and player.get('name'):
                        drafted_names.append(player['name'])
                
                # Check bench players too
                if 'Bench' in our_team and isinstance(our_team['Bench'], list):
                    for bench_player in our_team['Bench']:
                        if isinstance(bench_player, dict) and bench_player.get('name'):
                            drafted_names.append(bench_player['name'])
            
            print(f"🚫 Excluding {len(drafted_names)} drafted players from AI recommendations: {drafted_names}")
            
            # Import the original data loading for AI recommendations
            from draft_optimizer import load_players, load_projections, prepare_data
            from draft_optimizer import suggest_picks
            
            # Load complex data for AI model only
            players = load_players()
            players_list = []
            for player_id, player_data in players.items():
                player_name = player_data.get('name', '')
                
                # Check if this AI player matches any drafted player
                # AI uses abbreviated names like "J.CHASE", drafted uses full names like "Ja'Marr Chase"
                is_drafted = False
                
                for drafted_name in drafted_names:
                    # Try different matching strategies:
                    
                    # 1. Exact match (unlikely but check anyway)
                    if player_name == drafted_name:
                        is_drafted = True
                        break
                    
                    # 2. Check if abbreviated name matches full name pattern
                    if '.' in player_name and ' ' in drafted_name:
                        # Split AI name: "J.CHASE" -> ["J", "CHASE"]
                        ai_parts = player_name.split('.')
                        if len(ai_parts) == 2:
                            ai_first_initial = ai_parts[0].strip()
                            ai_last_name = ai_parts[1].strip()
                            
                            # Split drafted name: "Ja'Marr Chase" -> ["Ja'Marr", "Chase"] 
                            drafted_parts = drafted_name.split()
                            if len(drafted_parts) >= 2:
                                drafted_first = drafted_parts[0].strip()
                                drafted_last = drafted_parts[-1].strip()  # Take last part as surname
                                
                                # Check if first initial and last name match
                                if (ai_first_initial.upper() == drafted_first[0].upper() and 
                                    ai_last_name.upper() == drafted_last.upper()):
                                    is_drafted = True
                                    print(f"✅ Matched AI player '{player_name}' with drafted player '{drafted_name}'")
                                    break
                
                # SKIP drafted players
                if not is_drafted:
                    player_info = player_data.copy()
                    player_info['player_id'] = player_id
                    players_list.append(player_info)
                else:
                    print(f"🚫 Skipping drafted player: AI '{player_name}' matches drafted '{[d for d in drafted_names if d in player_name or player_name in d or any(p in d for p in player_name.split('.'))]}')")
            
            complex_df = pd.DataFrame(players_list)
            complex_df = load_projections(complex_df)
            
            # Remove duplicates (keep first occurrence)
            complex_df = complex_df.drop_duplicates(subset=['name'], keep='first')
            print(f"🧹 Removed duplicates, {len(complex_df)} unique players remaining")
            
            print(f"📊 AI analyzing {len(complex_df)} available players (drafted players excluded)")
            
            # Calculate team needs based on current roster
            team_needs_calc = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'K': 1, 'DST': 1}
            
            # Count filled positions in our team
            if isinstance(our_team, dict):
                for position, player in our_team.items():
                    if position != 'Bench' and isinstance(player, dict) and player.get('position'):
                        pos = player['position']
                        if pos in team_needs_calc:
                            team_needs_calc[pos] = max(0, team_needs_calc[pos] - 1)
            
            print(f"🎯 Current team needs: {team_needs_calc}")
            
            # Apply positional scarcity model
            from positional_scarcity import PositionalScarcityModel, estimate_current_round
            current_round = estimate_current_round(our_team, league_size=10)
            scarcity_model = PositionalScarcityModel(league_size=10)
            
            print(f"📊 Estimated draft round: {current_round}")
            
            # Use the NEW PROPER CatBoost model (no leakage)
            import joblib
            try:
                model = joblib.load('models/proper_fantasy_model.pkl')
                print("✅ Loaded PROPER CatBoost model (no data leakage)")
                
                # Define the proper model features
                model_features = [
                    'hist_games_played',
                    'hist_avg_passing_yards', 'hist_avg_passing_tds', 'hist_avg_interceptions',
                    'hist_avg_rushing_yards', 'hist_avg_rushing_tds',
                    'hist_avg_receiving_yards', 'hist_avg_receiving_tds', 
                    'hist_avg_receptions', 'hist_avg_targets', 'hist_avg_carries',
                    'hist_std_fantasy_points', 'hist_max_fantasy_points', 'hist_min_fantasy_points',
                    'recent_avg_fantasy_points', 'recent_vs_season_trend',
                    'is_qb', 'is_rb', 'is_wr', 'is_te',
                    'season_2022', 'season_2023', 'season_2024',
                    'week', 'early_season', 'mid_season', 'late_season'
                ]
            except Exception as e:
                print(f"❌ Could not load proper model: {e}")
                model, model_features = None, None
            
            # Handle old model format by getting features from create_football_features
            if model is not None and model_features is None:
                print("🔧 Handling old model format - extracting features...")
                # Create a small sample to get feature names
                sample_df = complex_df.head(10).copy()
                sample_features = create_football_features(sample_df)
                # Get feature names from the sample (excluding non-numeric columns)
                model_features = [col for col in sample_features.columns if 
                                sample_features[col].dtype in ['int64', 'float64'] and 
                                col not in ['name', 'position', 'team', 'projected_points']]
                print(f"🔧 Extracted {len(model_features)} features from old model format")
            
            if model is not None and model_features is not None:
                feature_df = create_football_features(complex_df.copy())
                
                if len(feature_df) > 0:
                    missing_features = [f for f in model_features if f not in feature_df.columns]
                    for f in missing_features:
                        feature_df[f] = 0
                    
                    X = feature_df[model_features]
                    catboost_predictions = model.predict(X)
                    
                    complex_df['catboost_prediction'] = catboost_predictions
                    
                    # 🎯 AMPLIFY CatBoost with Scarcity (not replace)
                    enhanced_suggestions = scarcity_model.apply_scarcity_boost(complex_df, our_team, current_round)
                    
                    # Sort by boosted scores (CatBoost × Scarcity)
                    enhanced_suggestions = enhanced_suggestions.sort_values('boosted_score', ascending=False)
                    print(f"🤖 Using CatBoost × Scarcity amplification for {len(enhanced_suggestions)} players")
                else:
                    enhanced_suggestions = complex_df.head(10)
            else:
                print("❌ CatBoost model not available, using scarcity-based sorting")
                # Still apply scarcity model even without CatBoost
                enhanced_suggestions = scarcity_model.apply_scarcity_boost(complex_df, our_team, current_round)
                enhanced_suggestions = enhanced_suggestions.sort_values('boosted_score', ascending=False)
            
            # Format for frontend
            formatted_suggestions = []
            for _, suggestion in enhanced_suggestions.head(10).iterrows():  # Top 10 AI picks
                # Use boosted score as the optimized score (includes scarcity boost)
                boosted_score = suggestion.get('boosted_score', suggestion.get('catboost_prediction', suggestion.get('projected_points', 0)))
                scarcity_boost = suggestion.get('scarcity_boost', 1.0)
                
                formatted_suggestions.append({
                    'name': suggestion['name'],
                    'position': suggestion['position'],
                    'adp_rank': suggestion.get('adp_rank', None),
                    'projected_points': float(suggestion['projected_points']) if not pd.isna(suggestion.get('projected_points', 0)) else 0.0,
                    'bye_week': suggestion.get('bye_week', 'Unknown'),
                    'team': suggestion.get('team', ''),
                    'optimized_score': float(boosted_score) if not pd.isna(boosted_score) else 0.0,
                    'scarcity_boost': round(float(scarcity_boost), 2) if not pd.isna(scarcity_boost) else 1.0
                })
            
            return jsonify(formatted_suggestions)
            
        except Exception as e:
            print(f"❌ Error with AI model, falling back to scarcity-only ADP: {e}")
            # Fallback: Apply scarcity model to basic ADP rankings
            try:
                scarcity_enhanced = scarcity_model.apply_scarcity_boost(available_df.head(50), our_team, current_round)
                scarcity_enhanced = scarcity_enhanced.sort_values('boosted_score', ascending=False)
                top_picks = scarcity_enhanced.head(10)
                
                enhanced_suggestions = []
                for _, player in top_picks.iterrows():
                    boosted_score = player.get('boosted_score', player.get('projected_points', 0))
                    scarcity_boost = player.get('scarcity_boost', 1.0)
                    
                    enhanced_suggestions.append({
                        'name': player['name'],
                        'position': player['position'],
                        'adp_rank': player.get('adp_rank', None),
                        'projected_points': float(player['projected_points']) if not pd.isna(player.get('projected_points', 0)) else 0.0,
                        'bye_week': player.get('bye_week', 'Unknown'),
                        'team': player.get('team', ''),
                        'optimized_score': float(boosted_score) if not pd.isna(boosted_score) else 0.0,
                        'scarcity_boost': round(float(scarcity_boost), 2) if not pd.isna(scarcity_boost) else 1.0
                    })
                
            except Exception as e2:
                print(f"❌ Even scarcity fallback failed: {e2}")
                # Final fallback to basic ADP
                top_picks = available_df.head(10)
                enhanced_suggestions = []
                
                for _, player in top_picks.iterrows():
                    enhanced_suggestions.append({
                        'name': player['name'],
                        'position': player['position'],
                        'adp_rank': player.get('adp_rank', None),
                        'projected_points': float(player['projected_points']) if not pd.isna(player.get('projected_points', 0)) else 0.0,
                        'bye_week': player.get('bye_week', 'Unknown'),
                        'team': player.get('team', ''),
                        'optimized_score': float(player['projected_points']) if not pd.isna(player.get('projected_points', 0)) else 0.0,
                        'scarcity_boost': None  # Explicitly null for basic fallback
                    })
            
            return jsonify(enhanced_suggestions)

@app.route('/search')
def search():
    """Search for players by name"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])
    
    available_df, our_team, drafted_by_others, team_needs = load_state()
    
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
        
        # Find the player
        player_row = available_df[available_df['name'] == player_name]
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
        
        # Find the player
        player_row = available_df[available_df['name'] == player_name]
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