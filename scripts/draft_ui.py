print("🚀 DRAFT_UI.PY LOADED!")
import sys
sys.path.insert(0, '.')
from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import numpy as np
import json
import os
# Robust import for injury_tracker to support both direct script run and package import
try:
    from injury_tracker import get_current_injury_data, add_injury_status_to_dataframe
except Exception:
    try:
        from scripts.injury_tracker import get_current_injury_data, add_injury_status_to_dataframe
    except Exception:
        def get_current_injury_data():
            return pd.DataFrame(columns=['full_name', 'report_status', 'report_primary_injury'])
        def add_injury_status_to_dataframe(player_df, injury_df):
            player_df = player_df.copy()
            player_df['injury_status'] = None
            player_df['injury_description'] = ""
            player_df['injury_icon'] = ""
            return player_df

app = Flask(__name__, template_folder='../templates')

# Suggestion mode: 'heuristic' (default) or 'raw'/'model_only' to return model-only ranking
AI_SUGGEST_MODE = os.getenv('AI_SUGGEST_MODE', 'heuristic').lower()

# 🚀 FAST AI CACHE: Store AI predictions to avoid recalculating on every request
AI_PREDICTIONS_CACHE = None
CACHE_PLAYER_COUNT = 0
CACHE_TIMESTAMP = 0

def can_use_ai_cache(current_player_count):
    """Check if we can use cached AI predictions"""
    global CACHE_PLAYER_COUNT, CACHE_TIMESTAMP
    if AI_PREDICTIONS_CACHE is None:
        return False
    # Use cache if player count changed by 5 or less (typical draft scenario)
    if abs(current_player_count - CACHE_PLAYER_COUNT) <= 5:
        return True
    return False

def get_cached_ai_predictions(available_df):
    """Get AI predictions from cache or calculate new ones"""
    global AI_PREDICTIONS_CACHE, CACHE_PLAYER_COUNT, CACHE_TIMESTAMP
    current_count = len(available_df)

    if can_use_ai_cache(current_count):
        print(f"⚡ Using cached AI predictions ({current_count} players)")
        # Filter cached predictions to match current available players
        cached_filtered = AI_PREDICTIONS_CACHE[AI_PREDICTIONS_CACHE['name'].isin(available_df['name'])]
        return cached_filtered

    print(f"🔄 Calculating new AI predictions ({current_count} players)")
    # Calculate new predictions
    if not MODEL_ADAPTER_AVAILABLE:
        return None

    ai_results = predict_players(available_df)
    if ai_results is not None:
        # Update cache
        AI_PREDICTIONS_CACHE = ai_results.copy()
        CACHE_PLAYER_COUNT = current_count
        CACHE_TIMESTAMP = pd.Timestamp.now().timestamp()
        print(f"💾 Cached AI predictions for {current_count} players")

    return ai_results

def clear_ai_cache():
    """Clear the AI predictions cache"""
    global AI_PREDICTIONS_CACHE, CACHE_PLAYER_COUNT, CACHE_TIMESTAMP
    AI_PREDICTIONS_CACHE = None
    CACHE_PLAYER_COUNT = 0
    CACHE_TIMESTAMP = 0
    print("🗑️ AI predictions cache cleared")

# Load the AI model adapter
MODEL_ADAPTER_AVAILABLE = False
try:
    # Set the model path before importing - using the new weighted model with sample weights and temporal validation
    # Use relative path that works on both local and deployed environments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'models', 'weighted_fantasy_model_fixed.pkl')
    os.environ['PROPER_MODEL_PATH'] = os.path.abspath(model_path)

    # Add scripts directory to path
    scripts_dir = os.path.join(os.path.dirname(__file__), '.')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from proper_model_adapter import predict_players, is_model_available
    MODEL_ADAPTER_AVAILABLE = True
    print("✅ Model adapter loaded successfully")
except Exception as e:
    print(f"⚠️ Model adapter not available: {e}")
    MODEL_ADAPTER_AVAILABLE = False

# Constants
STATE_FILE = 'draft_state.json'

# Default roster configuration (can be updated by user)
DEFAULT_ROSTER_CONFIG = {
    'num_teams': 10,
    'use_positions': True,
    'QB': 1,
    'RB': 2,
    'WR': 2,
    'TE': 1,
    'FLEX': 1,
    'K': 1,
    'DST': 1,
    'Bench': 6,
    'STARTERS': 8
}

# Supported draft strategies (user-selectable via /strategy)
ALLOWED_STRATEGIES = {"balanced", "wr_first", "hero_rb"}

def build_team_from_config(config: dict) -> dict:
    cfg = {**DEFAULT_ROSTER_CONFIG, **(config or {})}
    team = {}
    if cfg.get('use_positions', True):
        if cfg['QB'] >= 1:
            team['QB'] = None
        for i in range(1, cfg['RB'] + 1):
            team[f'RB{i}'] = None
        for i in range(1, cfg['WR'] + 1):
            team[f'WR{i}'] = None
        if cfg['TE'] >= 1:
            team['TE'] = None
        if cfg['FLEX'] == 1:
            team['FLEX'] = None
        elif cfg['FLEX'] > 1:
            for i in range(1, cfg['FLEX'] + 1):
                team[f'FLEX{i}'] = None
        if cfg['K'] >= 1:
            team['K'] = None
        if cfg['DST'] >= 1:
            team['DST'] = None
    else:
        starters = max(1, int(cfg.get('STARTERS', 8)))
        for i in range(1, starters + 1):
            team[f'STARTER{i}'] = None
    team['Bench'] = [None] * int(cfg.get('Bench', 6))
    return team

def build_position_slot_map(config: dict) -> dict:
    cfg = {**DEFAULT_ROSTER_CONFIG, **(config or {})}
    slot_map = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'K': [], 'DST': []}
    if cfg.get('use_positions', True):
        if cfg['QB'] >= 1:
            slot_map['QB'].append('QB')
        for i in range(1, cfg['RB'] + 1):
            slot_map['RB'].append(f'RB{i}')
        for i in range(1, cfg['WR'] + 1):
            slot_map['WR'].append(f'WR{i}')
        if cfg['TE'] >= 1:
            slot_map['TE'].append('TE')
        if cfg['K'] >= 1:
            slot_map['K'].append('K')
        if cfg['DST'] >= 1:
            slot_map['DST'].append('DST')
        # FLEX allows RB/WR/TE
        flex_slots = []
        if cfg['FLEX'] == 1:
            flex_slots = ['FLEX']
        elif cfg['FLEX'] > 1:
            flex_slots = [f'FLEX{i}' for i in range(1, cfg['FLEX'] + 1)]
        for s in flex_slots:
            slot_map['RB'].append(s)
            slot_map['WR'].append(s)
            slot_map['TE'].append(s)
    else:
        starters = int(cfg.get('STARTERS', 8))
        starter_slots = [f'STARTER{i}' for i in range(1, starters + 1)]
        for p in slot_map.keys():
            slot_map[p].extend(starter_slots)
    return slot_map

def build_position_targets_from_config(config: dict) -> dict:
    cfg = {**DEFAULT_ROSTER_CONFIG, **(config or {})}
    if cfg.get('use_positions', True):
        return {
            'QB': cfg.get('QB', 1),
            'RB': cfg.get('RB', 2),
            'WR': cfg.get('WR', 2),
            'TE': cfg.get('TE', 1),
            'K': cfg.get('K', 1),
            'DST': cfg.get('DST', 1)
        }
    starters = int(cfg.get('STARTERS', 8))
    return {'QB': 1, 'RB': max(2, starters // 3), 'WR': max(2, starters // 3), 'TE': 1, 'K': 1, 'DST': 1}

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
        team_needs = state.get('team_needs') or build_position_targets_from_config(state.get('roster_config', {}))
        
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
        roster_config = DEFAULT_ROSTER_CONFIG.copy()
        our_team = build_team_from_config(roster_config)
        drafted_by_others = []
        team_needs = build_position_targets_from_config(roster_config)
        
        return available_df, our_team, drafted_by_others, team_needs

def save_state(available_df, our_team, drafted_by_others, team_needs):
    """Save draft state to file"""
    # Preserve existing fields like roster_config, num_teams, draft_strategy, etc.
    try:
        with open(STATE_FILE, 'r') as f:
            prev = json.load(f)
    except Exception:
        prev = {}

    state = prev.copy()
    state.update({
        'available_df': available_df.to_dict('records'),
        'our_team': our_team,
        'drafted_by_others': drafted_by_others,
        'team_needs': team_needs
    })

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
    print("🎯 SUGGEST FUNCTION CALLED!")
    try:
        debug_mode = request.args.get('diag') == 'true'
        diag = {}
        mode_arg = (request.args.get('mode') or '').lower()
        raw_mode = (mode_arg in ('raw', 'model', 'model_only', 'ai')) or (AI_SUGGEST_MODE in ('raw','model','model_only','ai'))
        if debug_mode:
            diag['mode'] = 'raw' if raw_mode else 'heuristic'
        # Check for filtering parameters
        position_filter = request.args.get('position')
        all_available = request.args.get('all_available') == 'true'
        num_teams = DEFAULT_ROSTER_CONFIG['num_teams']
        # Load draft state
        our_team = []
        opponent_team = []
        drafted_names = []
        
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                our_team = state.get('our_team', [])
                opponent_team = state.get('opponent_team', [])
                drafted_by_others = state.get('drafted_by_others', [])
                # Use league size from state when available
                try:
                    num_teams = int(state.get('num_teams', num_teams) or num_teams)
                except Exception:
                    pass
                
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

                if isinstance(drafted_by_others, list):
                    for player in drafted_by_others:
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
        current_round = (total_picks // max(1, int(num_teams))) + 1
        print(f"📊 Estimated draft round: {current_round} (total_picks={total_picks}, num_teams={num_teams})")
        try:
            with open(STATE_FILE, 'r') as f:
                _st_strategy = json.load(f)
            draft_strategy = _st_strategy.get('draft_strategy', 'balanced')
        except Exception:
            draft_strategy = 'balanced'
        print(f"🎛️ Strategy: {draft_strategy}")
        print("🎯 Returning AI recommendations")
        
        # Round-aware position gating
        def allowed_positions_for_round(rnd: int) -> set:
            if rnd <= 3:
                return {'WR', 'RB'}
            elif rnd <= 11:
                # Allow core positions; avoid K/DST until late
                return {'WR', 'RB', 'QB', 'TE'}
            else:
                return {'WR', 'RB', 'QB', 'TE', 'K', 'DST'}
        allowed_positions = allowed_positions_for_round(current_round)

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
            
            # Define position needs from config if present
            try:
                with open(STATE_FILE, 'r') as f:
                    st = json.load(f)
                position_targets = build_position_targets_from_config(st.get('roster_config', {}))
            except Exception:
                position_targets = build_position_targets_from_config({})
            
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
            print("🔍 Entering model-based suggestion logic")
            # Use pre-imported modules to ensure new model path is used
            if not MODEL_ADAPTER_AVAILABLE:
                print("❌ Model adapter not available")
                raise ImportError("Model adapter not available")
            print("✅ Model adapter is available")
            # Diagnostics for model presence
            try:
                model_available = bool(is_model_available())
                diag['model_available'] = model_available
                print(f"🔍 Model available check: {model_available}")
                print(f"🔍 is_model_available function: {is_model_available}")
                print(f"🔍 is_model_available() result: {is_model_available()}")
            except Exception as e:
                diag['model_available'] = False
                print(f"❌ Model available check failed: {e}")
                import traceback
                traceback.print_exc()
            try:
                # Use global os import to avoid shadowing errors
                diag['models_dir'] = sorted(os.listdir('models')) if os.path.isdir('models') else []
                model_path = '/home/nar/ff2025/models/leakage_free_model_20250821_084127.pkl'
                diag['model_file_exists'] = os.path.exists(model_path)
                if diag['model_file_exists']:
                    diag['model_file_size'] = os.path.getsize(model_path)
                # Quantile model presence
                q05_path = 'models/proper_fantasy_model_q05.pkl'
                q50_path = 'models/proper_fantasy_model_q50.pkl'
                q95_path = 'models/proper_fantasy_model_q95.pkl'
                diag['quantile_files'] = {
                    'q05': os.path.exists(q05_path),
                    'q50': os.path.exists(q50_path),
                    'q95': os.path.exists(q95_path),
                }
            except Exception:
                pass

            if diag.get('model_available', False):
                print("🤖 Using PROPER AI model with no data leakage")
                print(f"📊 Model available: {diag.get('model_available')}")
                print(f"📁 Model files exist: {diag.get('model_file_exists')}")
                print(f"📊 Quantile files: {diag.get('quantile_files')}")
                # Get AI predictions for available players
                print(f"📊 Available DF shape: {available_df.shape}")
                print(f"📊 Available DF columns: {list(available_df.columns)}")
                print(f"📊 First player data: {available_df.iloc[0].to_dict() if len(available_df) > 0 else 'No data'}")

                try:
                    ai_results = get_cached_ai_predictions(available_df)
                except Exception as e:
                    print(f"❌ Error calling predict_players: {e}")
                    import traceback
                    traceback.print_exc()
                    ai_results = None
                
                if ai_results is not None:
                    # Prefer median if available for point estimate
                    ai_results = ai_results.copy()
                    print(f"🔍 AI results columns: {list(ai_results.columns)}")
                    print(f"🔍 AI results shape: {ai_results.shape}")
                    print(f"🔍 First AI prediction: {ai_results['ai_prediction'].iloc[0] if 'ai_prediction' in ai_results.columns else 'N/A'}")

                    if 'ai_p50' in ai_results.columns:
                        ai_results['ai_point'] = ai_results['ai_p50']
                        print("✅ Using ai_p50 for ai_point")
                    else:
                        ai_results['ai_point'] = ai_results['ai_prediction']
                        print("✅ Using ai_prediction for ai_point")

                    print(f"🔍 First ai_point value: {ai_results['ai_point'].iloc[0] if len(ai_results) > 0 else 'N/A'}")
                    # Raw/model-only mode: return pure model ranking, no boosts/heuristics
                    if raw_mode:
                        print("🧪 Model-only mode active: returning raw AI ranking (no boosts).")
                        # Apply strict position gating before ranking
                        gated_ai = ai_results[ai_results['position'].isin(list(allowed_positions))].copy()
                        sorted_ai = gated_ai.sort_values('ai_point', ascending=False)
                        top_ai = sorted_ai.head(8)
                        formatted = []
                        for _, suggestion in top_ai.iterrows():
                            ai_p5_val = suggestion.get('ai_p5', None)
                            ai_p50_val = suggestion.get('ai_p50', None)
                            ai_p95_val = suggestion.get('ai_p95', None)
                            ai_lower_val = suggestion.get('ai_lower', None)
                            ai_upper_val = suggestion.get('ai_upper', None)
                            ai_point_val = suggestion.get('ai_point', None)
                            formatted.append({
                                'name': suggestion.get('name'),
                                'position': suggestion.get('position'),
                                'adp_rank': suggestion.get('adp_rank', None),
                                'projected_points': float(suggestion.get('projected_points')) if (suggestion.get('projected_points') is not None and not pd.isna(suggestion.get('projected_points'))) else 0.0,
                                'bye_week': suggestion.get('bye_week', 'Unknown'),
                                'team': suggestion.get('team', ''),
                                'optimized_score': float(ai_point_val) if (ai_point_val is not None and not pd.isna(ai_point_val)) else 0.0,
                                'ai_score': float(ai_point_val) if (ai_point_val is not None and not pd.isna(ai_point_val)) else 0.0,
                                'scarcity_boost': 1.0,
                                'ai_p5': (float(ai_p5_val) if (ai_p5_val is not None and not pd.isna(ai_p5_val)) else None),
                                'ai_p50': (float(ai_p50_val) if (ai_p50_val is not None and not pd.isna(ai_p50_val)) else None),
                                'ai_p95': (float(ai_p95_val) if (ai_p95_val is not None and not pd.isna(ai_p95_val)) else None),
                                'ai_lower': (float(ai_lower_val) if (ai_lower_val is not None and not pd.isna(ai_lower_val)) else None),
                                'ai_upper': (float(ai_upper_val) if (ai_upper_val is not None and not pd.isna(ai_upper_val)) else None),
                            })
                        diag['used'] = 'MODEL_ONLY'
                        diag['current_round'] = int(current_round)
                        diag['allowed_positions'] = sorted(list(allowed_positions)) if 'allowed_positions' in locals() else None
                        if debug_mode:
                            return jsonify({'diag': diag, 'suggestions': formatted})
                        return jsonify(formatted)
                    # Apply scarcity boost to AI predictions
                    enhanced_suggestions = apply_simple_scarcity_boost(ai_results, our_team, current_round)
                    
                    # Calculate boosted score using AI predictions
                    enhanced_suggestions['boosted_score'] = (
                        enhanced_suggestions['ai_point'].astype(float) * enhanced_suggestions['scarcity_boost'].astype(float)
                    )
                    
                    # Sort by boosted AI scores
                    enhanced_suggestions = enhanced_suggestions.sort_values('boosted_score', ascending=False)
                    print(f"✅ Using PROPER AI × Scarcity for {len(enhanced_suggestions)} players")
                else:
                    print("❌ AI prediction failed, using scarcity-only")
                    diag['used'] = 'SCARCITY_FALLBACK_AI_NONE'
                    enhanced_suggestions = apply_simple_scarcity_boost(available_df, our_team, current_round)
                    enhanced_suggestions['boosted_score'] = (
                        enhanced_suggestions['projected_points'] * enhanced_suggestions['scarcity_boost']
                    )
                    enhanced_suggestions = enhanced_suggestions.sort_values('boosted_score', ascending=False)
            else:
                print("❌ Proper AI model not available.")
                if raw_mode:
                    diag['used'] = 'RAW_NO_MODEL'
                    fallback_df = available_df.sort_values('adp_rank', ascending=True, na_position='last') if 'adp_rank' in available_df.columns else available_df
                    # Apply strict position gating before taking top N
                    gated_fallback = fallback_df[fallback_df['position'].isin(list(allowed_positions))]
                    top_df = gated_fallback.head(8)
                    formatted = []
                    for _, row in top_df.iterrows():
                        formatted.append({
                            'name': row.get('name'),
                            'position': row.get('position'),
                            'adp_rank': row.get('adp_rank', None),
                            'projected_points': float(row.get('projected_points')) if (row.get('projected_points') is not None and not pd.isna(row.get('projected_points'))) else 0.0,
                            'bye_week': row.get('bye_week', 'Unknown'),
                            'team': row.get('team', ''),
                            'optimized_score': float(row.get('projected_points')) if (row.get('projected_points') is not None and not pd.isna(row.get('projected_points'))) else 0.0,
                            'ai_score': 0.0,
                            'scarcity_boost': 1.0,
                            'ai_p5': None,
                            'ai_p50': None,
                            'ai_p95': None,
                            'ai_lower': None,
                            'ai_upper': None,
                        })
                    if debug_mode:
                        diag['current_round'] = int(current_round)
                        diag['allowed_positions'] = sorted(list(allowed_positions)) if 'allowed_positions' in locals() else None
                        return jsonify({'diag': diag, 'suggestions': formatted})
                    return jsonify(formatted)
                print("🔍 Using scarcity-based sorting (fallback mode)")
                diag['used'] = 'SCARCITY_NO_MODEL'
                enhanced_suggestions = apply_simple_scarcity_boost(available_df, our_team, current_round)
                enhanced_suggestions['boosted_score'] = (
                    enhanced_suggestions['projected_points'] * enhanced_suggestions['scarcity_boost']
                )
                enhanced_suggestions = enhanced_suggestions.sort_values('boosted_score', ascending=False)
        except Exception as e:
            print(f"❌ Error with AI model: {e}")
            diag['used'] = 'SCARCITY_EXCEPTION'
            diag['error'] = str(e)
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
                    # 2025 strategy: Moderate WR priority early; RB anchor possible
                    if rnd <= 2:
                        w = 1.15 if pos == 'WR' else 0.94
                    elif rnd == 3:
                        w = 1.08 if pos == 'WR' else 0.97
                    elif rnd in (4, 5):
                        w = 1.05 if pos == 'WR' else 1.00
                    else:
                        w = 1.0
                    if have < need_two and pos == 'WR':
                        # If we don't yet have two WRs, give a small bump
                        w *= 1.10
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

        # Anchor RB rule in early rounds: prefer WR unless RB clearly clears by margin
        try:
            if current_round <= 2 and 'position' in enhanced_suggestions.columns:
                top_rb = enhanced_suggestions[enhanced_suggestions['position'] == 'RB']['final_score']
                top_wr = enhanced_suggestions[enhanced_suggestions['position'] == 'WR']['final_score']
                if len(top_rb) > 0 and len(top_wr) > 0:
                    rb_max = float(top_rb.max())
                    wr_max = float(top_wr.max())
                    # Strategy-driven RB anchor margin
                    try:
                        with open(STATE_FILE, 'r') as f:
                            _st = json.load(f)
                        _strat = _st.get('draft_strategy', 'balanced')
                    except Exception:
                        _strat = 'balanced'
                    if _strat == 'hero_rb':
                        margin = 1.06
                    elif _strat == 'wr_first':
                        margin = 1.14
                    else:
                        margin = 1.10
                    if rb_max >= wr_max * margin:
                        # small universal RB nudge to acknowledge anchor RB path
                        mask = enhanced_suggestions['position'] == 'RB'
                        enhanced_suggestions.loc[mask, 'final_score'] = (
                            enhanced_suggestions.loc[mask, 'final_score'].astype(float) * 1.00
                        )
                    else:
                        # default to WR preference in R1-2
                        mask = enhanced_suggestions['position'] == 'WR'
                        enhanced_suggestions.loc[mask, 'final_score'] = (
                            enhanced_suggestions.loc[mask, 'final_score'].astype(float) * 1.08
                        )
                    enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)
        except Exception:
            pass

        # Strategy macro-weights
        try:
            # Read strategy once here
            try:
                with open(STATE_FILE, 'r') as f:
                    _st2 = json.load(f)
                _strategy = _st2.get('draft_strategy', 'balanced')
            except Exception:
                _strategy = 'balanced'
            if 'position' in enhanced_suggestions.columns:
                if _strategy == 'wr_first' and current_round <= 3:
                    mask_wr = enhanced_suggestions['position'] == 'WR'
                    enhanced_suggestions.loc[mask_wr, 'final_score'] = (
                        enhanced_suggestions.loc[mask_wr, 'final_score'].astype(float) * 1.06
                    )
                    if current_round <= 2:
                        mask_rb = enhanced_suggestions['position'] == 'RB'
                        enhanced_suggestions.loc[mask_rb, 'final_score'] = (
                            enhanced_suggestions.loc[mask_rb, 'final_score'].astype(float) * 0.98
                        )
                    enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)
                elif _strategy == 'hero_rb' and current_round <= 2:
                    mask_rb = enhanced_suggestions['position'] == 'RB'
                    enhanced_suggestions.loc[mask_rb, 'final_score'] = (
                        enhanced_suggestions.loc[mask_rb, 'final_score'].astype(float) * 1.03
                    )
                    enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)
        except Exception:
            pass

        # Dual‑threat QB micro‑bonus in Rounds 3–4
        try:
            if current_round in (3, 4):
                dual_qbs = {
                    'Lamar Jackson', 'Josh Allen', 'Jalen Hurts', 'Jayden Daniels'
                }
                mask = (enhanced_suggestions['position'] == 'QB') & (
                    enhanced_suggestions['name'].isin(list(dual_qbs))
                )
                if mask.any():
                    enhanced_suggestions.loc[mask, 'final_score'] = (
                        enhanced_suggestions.loc[mask, 'final_score'].astype(float) * 1.04
                    )
                    enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)
        except Exception:
            pass

        # Tight end gating: only boost elite TE early; otherwise penalize
        try:
            if current_round <= 3:
                elite_tes = {'Sam LaPorta', 'Trey McBride', 'Brock Bowers'}
                te_mask = (enhanced_suggestions['position'] == 'TE')
                elite_mask = te_mask & enhanced_suggestions['name'].isin(list(elite_tes))
                non_elite_mask = te_mask & ~enhanced_suggestions['name'].isin(list(elite_tes))
                if elite_mask.any():
                    enhanced_suggestions.loc[elite_mask, 'final_score'] = (
                        enhanced_suggestions.loc[elite_mask, 'final_score'].astype(float) * 1.03
                    )
                if non_elite_mask.any():
                    enhanced_suggestions.loc[non_elite_mask, 'final_score'] = (
                        enhanced_suggestions.loc[non_elite_mask, 'final_score'].astype(float) * 0.88
                    )
                enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)
        except Exception:
            pass

        # Blend AI-driven score with ADP baseline; add early-round ADP anchoring
        try:
            adp_baseline = []
            for _, row in enhanced_suggestions.iterrows():
                proj = float(row.get('projected_points', 0.0) or 0.0)
                sc = float(row.get('scarcity_boost', 1.0) or 1.0)
                adp_baseline.append(proj * sc)

            enhanced_suggestions = enhanced_suggestions.copy()
            enhanced_suggestions['adp_baseline'] = adp_baseline

            # Round-based blend (moderate ADP anchoring in Round 1)
            if current_round == 1:
                # Give AI some voice while still leaning ADP
                alpha = 0.25
            elif current_round == 2:
                alpha = 0.65
            elif current_round <= 4:
                alpha = 0.75
            elif current_round <= 7:
                alpha = 0.80
            elif current_round <= 10:
                alpha = 0.85
            else:
                alpha = 0.90

            blended_scores = (
                enhanced_suggestions['final_score'].astype(float) * alpha +
                enhanced_suggestions['adp_baseline'].astype(float) * (1.0 - alpha)
            )

            # Early/mid ADP anchoring
            if current_round <= 7 and 'adp_rank' in enhanced_suggestions.columns:
                adp_series = enhanced_suggestions['adp_rank'].astype(float)
                adp_min = adp_series.min(skipna=True)
                adp_max = adp_series.max(skipna=True)
                denom = (adp_max - adp_min) if pd.notna(adp_max) and pd.notna(adp_min) and (adp_max - adp_min) > 0 else 1.0
                if current_round == 1:
                    # Moderate ADP anchor in Round 1 (allow some variation)
                    anchor_strength = 0.15
                elif current_round == 2:
                    anchor_strength = 0.06
                elif current_round <= 4:
                    anchor_strength = 0.04
                elif current_round <= 7:
                    anchor_strength = 0.03
                else:
                    anchor_strength = 0.00

                def anchor_multiplier(adp_val: float) -> float:
                    if pd.isna(adp_val):
                        return 1.0
                    norm = (float(adp_val) - float(adp_min)) / denom
                    return 1.0 + anchor_strength * (1.0 - max(0.0, min(1.0, norm)))

                multipliers = enhanced_suggestions['adp_rank'].apply(anchor_multiplier).astype(float)
                blended_scores = blended_scores * multipliers

            enhanced_suggestions['final_score'] = blended_scores
            enhanced_suggestions['blend_alpha'] = alpha
            enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)
        except Exception:
            pass
 
        # Encourage slight divergence from ADP when AI strongly disagrees (Rounds 1-2)
        try:
            if current_round <= 2 and 'adp_rank' in enhanced_suggestions.columns:
                # Use ai_point (median) if present, else fall back to ai_prediction
                if ('ai_point' in enhanced_suggestions.columns) or ('ai_prediction' in enhanced_suggestions.columns):
                    enhanced_suggestions = enhanced_suggestions.copy()
                    ai_for_rank = enhanced_suggestions.get('ai_point', enhanced_suggestions.get('ai_prediction', 0.0))
                    enhanced_suggestions['ai_rank'] = ai_for_rank.rank(ascending=False, method='min')
                    enhanced_suggestions['adp_rank_num'] = pd.to_numeric(enhanced_suggestions['adp_rank'], errors='coerce')
                    n = max(1.0, float(len(enhanced_suggestions)))
                    enhanced_suggestions['rank_delta_norm'] = (
                        (enhanced_suggestions['adp_rank_num'] - enhanced_suggestions['ai_rank']) / n
                    ).fillna(0.0)
                    # Softer exploration in Round 1, slightly stronger in Round 2
                    gamma = 0.15 if current_round == 1 else 0.35
                    base_boost = 1.0 + gamma * enhanced_suggestions['rank_delta_norm']
                    delta = (enhanced_suggestions['adp_rank_num'] - enhanced_suggestions['ai_rank']).fillna(0.0)
                    extra = pd.Series(1.0, index=delta.index)
                    if current_round == 1:
                        # Tiny extra bump only for big deltas
                        extra = extra.where(~(delta >= 10), 1.05)
                        extra = extra.where(~(delta >= 14), 1.08)
                    else:
                        extra = extra.where(~(delta >= 8), 1.12)
                        extra = extra.where(~(delta >= 12), 1.18)
                    boost = (base_boost * extra)
                    boost = boost.clip(lower=0.95 if current_round == 1 else 0.88,
                                       upper=1.08 if current_round == 1 else 1.20)
                    enhanced_suggestions['final_score'] = enhanced_suggestions['final_score'].astype(float) * boost.astype(float)
                    enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)
        except Exception:
            pass

        # Final ordering tweak for Round 1: use score first, then ADP as tie-breaker
        try:
            if current_round == 1 and 'adp_rank' in enhanced_suggestions.columns:
                enhanced_suggestions['adp_rank_num'] = pd.to_numeric(enhanced_suggestions['adp_rank'], errors='coerce')
                enhanced_suggestions = enhanced_suggestions.sort_values(
                    by=['final_score', 'adp_rank_num'], ascending=[False, True]
                )
        except Exception:
            pass

        # Enforce round-based position gating and QB deprioritization
        try:
            # Hard gate by allowed positions for this round
            if 'position' in enhanced_suggestions.columns:
                enhanced_suggestions = enhanced_suggestions[enhanced_suggestions['position'].isin(list(allowed_positions))].copy()
                enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)

            # Apply additional soft QB penalty beyond gating
            qb_penalty = 1.0
            if current_round <= 2:
                qb_penalty = 0.10
            elif current_round == 3:
                qb_penalty = 0.35
            elif current_round in (4, 5):
                qb_penalty = 0.60
            elif current_round in (6, 7):
                qb_penalty = 0.85
            else:
                qb_penalty = 1.0

            if qb_penalty < 1.0 and 'position' in enhanced_suggestions.columns:
                mask_qb = enhanced_suggestions['position'] == 'QB'
                enhanced_suggestions.loc[mask_qb, 'final_score'] = (
                    enhanced_suggestions.loc[mask_qb, 'final_score'].astype(float) * qb_penalty
                )
                enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)
        except Exception:
            pass

        # Format for frontend
        formatted_suggestions = []
        for _, suggestion in enhanced_suggestions.head(8).iterrows():  # Top 8 picks
            boosted_score = suggestion.get('boosted_score', suggestion.get('projected_points', 0))
            scarcity_boost = suggestion.get('scarcity_boost', 1.0)
            ai_score = suggestion.get('ai_point', suggestion.get('ai_prediction', 0))
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
                'scarcity_boost': round(float(scarcity_boost), 2) if not pd.isna(scarcity_boost) else 1.0,
                # Optional quantile outputs
                'ai_p5': (float(suggestion['ai_p5']) if ('ai_p5' in suggestion and not pd.isna(suggestion['ai_p5'])) else None),
                'ai_p50': (float(suggestion['ai_p50']) if ('ai_p50' in suggestion and not pd.isna(suggestion['ai_p50'])) else None),
                'ai_p95': (float(suggestion['ai_p95']) if ('ai_p95' in suggestion and not pd.isna(suggestion['ai_p95'])) else None),
                'ai_lower': (float(suggestion['ai_lower']) if ('ai_lower' in suggestion and not pd.isna(suggestion['ai_lower'])) else None),
                'ai_upper': (float(suggestion['ai_upper']) if ('ai_upper' in suggestion and not pd.isna(suggestion['ai_upper'])) else None)
            })
 
        if debug_mode:
            try:
                with open(STATE_FILE, 'r') as f:
                    _st = json.load(f)
                diag['roster_config'] = _st.get('roster_config', {})
                diag['num_teams'] = _st.get('num_teams', DEFAULT_ROSTER_CONFIG['num_teams']) if 'DEFAULT_ROSTER_CONFIG' in globals() else _st.get('num_teams')
                diag['blend_alpha'] = float(alpha) if 'alpha' in locals() else None
                diag['current_round'] = int(current_round)
                diag['allowed_positions'] = sorted(list(allowed_positions)) if 'allowed_positions' in locals() else None
            except Exception:
                pass
            return jsonify({'diag': diag, 'suggestions': formatted_suggestions})
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
    
    drafted_names = []
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            our_team = state.get('our_team', [])
            opponent_team = state.get('opponent_team', [])
            drafted_by_others = state.get('drafted_by_others', [])
            
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

            if isinstance(drafted_by_others, list):
                for player in drafted_by_others:
                    if isinstance(player, dict) and player.get('name'):
                        drafted_names.append(player['name'])

    available_df = df[~df['name'].isin(drafted_names)]
    
    # Name search among available players
    matches = available_df[available_df['name'].str.contains(query, case=False, na=False)]
    if matches.empty:
        return jsonify([])
    top_matches = matches.head(20).copy()

    # Try to enrich with AI predictions and quantiles
    try:
        if MODEL_ADAPTER_AVAILABLE and is_model_available():
            ai_results = predict_players(top_matches)
            if ai_results is not None and not ai_results.empty:
                ai_results = ai_results.copy()
                # Prefer median as point estimate if available
                if 'ai_p50' in ai_results.columns:
                    ai_results['ai_point'] = ai_results['ai_p50']
                else:
                    ai_results['ai_point'] = ai_results['ai_prediction']

                formatted = []
                for _, row in ai_results.iterrows():
                    ai_p5_val = row.get('ai_p5', None)
                    ai_p50_val = row.get('ai_p50', None)
                    ai_p95_val = row.get('ai_p95', None)
                    ai_lower_val = row.get('ai_lower', None)
                    ai_upper_val = row.get('ai_upper', None)
                    ai_point_val = row.get('ai_point', None)
                    formatted.append({
                        'name': row.get('name'),
                        'position': row.get('position'),
                        'adp_rank': row.get('adp_rank', None),
                        'projected_points': (
                            float(row.get('projected_points'))
                            if (row.get('projected_points') is not None and not pd.isna(row.get('projected_points')))
                            else 0.0
                        ),
                        'bye_week': row.get('bye_week', 'Unknown'),
                        'team': row.get('team', ''),
                        'is_rookie': bool(row.get('is_rookie', False)),
                        'injury_status': row.get('injury_status', None),
                        'injury_description': row.get('injury_description', ""),
                        'injury_icon': row.get('injury_icon', ""),
                        # AI fields
                        'ai_score': (float(ai_point_val) if (ai_point_val is not None and not pd.isna(ai_point_val)) else 0.0),
                        'ai_p5': (float(ai_p5_val) if (ai_p5_val is not None and not pd.isna(ai_p5_val)) else None),
                        'ai_p50': (float(ai_p50_val) if (ai_p50_val is not None and not pd.isna(ai_p50_val)) else None),
                        'ai_p95': (float(ai_p95_val) if (ai_p95_val is not None and not pd.isna(ai_p95_val)) else None),
                        'ai_lower': (float(ai_lower_val) if (ai_lower_val is not None and not pd.isna(ai_lower_val)) else None),
                        'ai_upper': (float(ai_upper_val) if (ai_upper_val is not None and not pd.isna(ai_upper_val)) else None),
                    })
                return jsonify(formatted)
    except Exception as e:
        print(f"⚠️ AI enrichment in /search failed: {e}")

    # Fallback: return basic info with AI fields absent/defaulted
    fallback = []
    for _, row in top_matches.iterrows():
        fallback.append({
            'name': row.get('name'),
            'position': row.get('position'),
            'adp_rank': row.get('adp_rank', None),
            'projected_points': (
                float(row.get('projected_points'))
                if (row.get('projected_points') is not None and not pd.isna(row.get('projected_points')))
                else 0.0
            ),
            'bye_week': row.get('bye_week', 'Unknown'),
            'team': row.get('team', ''),
            'is_rookie': bool(row.get('is_rookie', False)),
            'injury_status': row.get('injury_status', None),
            'injury_description': row.get('injury_description', ""),
            'injury_icon': row.get('injury_icon', ""),
            # Maintain consistent AI keys for frontend; no model -> 0.0/None
            'ai_score': 0.0,
            'ai_p5': None,
            'ai_p50': None,
            'ai_p95': None,
            'ai_lower': None,
            'ai_upper': None,
        })
    return jsonify(fallback)

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
        # Use roster config slot mapping
        try:
            with open(STATE_FILE, 'r') as f:
                st = json.load(f)
            roster_config = st.get('roster_config', DEFAULT_ROSTER_CONFIG)
        except Exception:
            roster_config = DEFAULT_ROSTER_CONFIG
        assigned_slot = assign_player_to_lineup(our_team, player_dict, roster_config)
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
        player_name = data.get('player')
        
        if not player_name:
            return jsonify({'success': False, 'error': 'No player specified'})
        
        player_data = None
        player_matches = df[df['name'].str.lower() == str(player_name).lower()]
        if not player_matches.empty:
            player_data = player_matches.iloc[0].to_dict()
        else:
            rookie_matches = rookie_df[rookie_df['name'].str.lower() == str(player_name).lower()]
            if not rookie_matches.empty:
                player_data = rookie_matches.iloc[0].to_dict()
         
        if not player_data:
            return jsonify({'success': False, 'error': 'Player not found'})
        
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
        else:
            # Initialize state using roster-aware builders when file doesn't exist
            roster_config = DEFAULT_ROSTER_CONFIG
            state = {
                'available_df': df.to_json(orient='records'),
                'our_team': build_team_from_config(roster_config),
                'drafted_by_others': [],
                'team_needs': build_position_targets_from_config(roster_config),
                'roster_config': roster_config,
                'num_teams': roster_config.get('num_teams', 10)
            }
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)

        if not isinstance(state.get('drafted_by_others'), list):
            state['drafted_by_others'] = []
        
        already_drafted = any(p.get('name') == player_name for p in state['drafted_by_others'] if isinstance(p, dict))
        if not already_drafted:
            state['drafted_by_others'].append(player_data)
        
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        
        # Convert NaN to None for JSON safety
        cleaned_player = {k: (None if pd.isna(v) else v) for k, v in player_data.items()}
        
        return jsonify({'success': True, 'player': cleaned_player})
         
    except Exception as e:
        print(f"Mark taken error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset', methods=['POST'])
def reset_draft():
    """Reset the entire draft"""
    try:
        # Clear AI cache first
        clear_ai_cache()

        # Preserve roster_config/num_teams and rebuild team from config
        try:
            with open(STATE_FILE, 'r') as f:
                prev = json.load(f)
        except Exception:
            prev = {}
        roster_config = prev.get('roster_config', DEFAULT_ROSTER_CONFIG)
        num_teams = prev.get('num_teams', DEFAULT_ROSTER_CONFIG['num_teams'])
        draft_strategy = prev.get('draft_strategy', 'balanced')

        initial_state = {
            'available_df': df.to_json(orient='records'),
            'our_team': build_team_from_config(roster_config),
            'drafted_by_others': [],
            'team_needs': build_position_targets_from_config(roster_config),
            'roster_config': roster_config,
            'num_teams': num_teams,
            'draft_strategy': draft_strategy
        }

        with open(STATE_FILE, 'w') as f:
            json.dump(initial_state, f)
        try:
            team_keys = [k for k in initial_state['our_team'].keys() if k != 'Bench']
            bench_len = len(initial_state['our_team'].get('Bench', []))
            print(f"🔄 Reset draft with roster_config: {roster_config}")
            print(f"🧩 Team slots after reset: {team_keys}, Bench={bench_len}")
        except Exception:
            pass

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
        
        # Check starting lineup positions dynamically based on our_team keys
        for slot, assigned in our_team.items():
            if slot == 'Bench':
                continue
            if isinstance(assigned, dict) and assigned.get('name') == player_name:
                player_data = assigned
                our_team[slot] = None
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
                available_df = available_df.sort_values('adp_rank', ascending=True, na_position='last')
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

def assign_player_to_lineup(our_team, player_dict, roster_config=None):
    """Assign player to the first available appropriate position based on roster_config."""
    position = player_dict['position']
    slot_map = build_position_slot_map(roster_config or DEFAULT_ROSTER_CONFIG)
    try:
        _cfg = roster_config or DEFAULT_ROSTER_CONFIG
        _cfg_short = {k: _cfg.get(k) for k in ['QB','RB','WR','TE','FLEX','K','DST','Bench','use_positions']}
        print(f"assign_player_to_lineup: position={position}, roster_cfg={_cfg_short}")
        print(f"assign_player_to_lineup: candidate_slots={slot_map.get(position, [])}")
    except Exception:
        pass
    for slot in slot_map.get(position, []):
        # Only assign to slots that actually exist in our_team to avoid
        # reintroducing removed positions (e.g., FLEX/TE) via assignment.
        if slot in our_team and our_team.get(slot) is None:
            our_team[slot] = player_dict
            return slot
    
    # If no starting slot available, put on bench
    for i, bench_slot in enumerate(our_team['Bench']):
        if bench_slot is None:
            our_team['Bench'][i] = player_dict
            return f'Bench {i+1}'
    
    # If bench is full, return None (shouldn't happen in normal draft)
    return None

# League settings endpoint for local dev
@app.route('/league_settings', methods=['GET', 'POST'])
def league_settings_local():
    try:
        if request.method == 'GET':
            try:
                with open(STATE_FILE, 'r') as f:
                    st = json.load(f)
            except Exception:
                st = {}
            cfg = st.get('roster_config', DEFAULT_ROSTER_CONFIG)
            num_teams = st.get('num_teams', DEFAULT_ROSTER_CONFIG['num_teams'])
            return jsonify({'success': True, 'config': cfg, 'num_teams': num_teams})
        # POST
        payload = request.get_json() or {}
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
        except Exception:
            state = {}
        new_cfg = {**state.get('roster_config', DEFAULT_ROSTER_CONFIG), **(payload.get('roster_config') or {})}
        for key in ['QB','RB','WR','TE','FLEX','K','DST','Bench','STARTERS']:
            if key in new_cfg:
                try:
                    new_cfg[key] = int(new_cfg[key])
                except Exception:
                    pass
        new_cfg['use_positions'] = bool(new_cfg.get('use_positions', True))
        num_teams = int(payload.get('num_teams', state.get('num_teams', DEFAULT_ROSTER_CONFIG['num_teams'])))
        # Rebuild structures
        state['roster_config'] = new_cfg
        state['num_teams'] = num_teams
        state['our_team'] = build_team_from_config(new_cfg)
        state['team_needs'] = build_position_targets_from_config(new_cfg)
        # Ensure available_df exists
        if 'available_df' not in state:
            state['available_df'] = df.to_json(orient='records')
        if 'drafted_by_others' not in state:
            state['drafted_by_others'] = []
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        try:
            team_keys = [k for k in state['our_team'].keys() if k != 'Bench']
            bench_len = len(state['our_team'].get('Bench', []))
            print(f"🛠️ Updated roster_config via /league_settings: {new_cfg}, num_teams={num_teams}")
            print(f"🧩 Team slots: {team_keys}, Bench={bench_len}")
        except Exception:
            pass
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Strategy endpoints for local dev
@app.route('/strategy', methods=['GET', 'POST'])
def strategy_settings():
    try:
        if request.method == 'GET':
            try:
                with open(STATE_FILE, 'r') as f:
                    st = json.load(f)
            except Exception:
                st = {}
            strat = st.get('draft_strategy', 'balanced')
            return jsonify({'success': True, 'draft_strategy': strat, 'allowed': sorted(list(ALLOWED_STRATEGIES))})
        payload = request.get_json() or {}
        new_strat = str(payload.get('draft_strategy', 'balanced')).lower()
        if new_strat not in ALLOWED_STRATEGIES:
            return jsonify({'success': False, 'error': 'Invalid strategy'}), 400
        try:
            with open(STATE_FILE, 'r') as f:
                st = json.load(f)
        except Exception:
            st = {}
        st['draft_strategy'] = new_strat
        with open(STATE_FILE, 'w') as f:
            json.dump(st, f)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)