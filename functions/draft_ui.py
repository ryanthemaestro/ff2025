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

# Default roster configuration (can be changed from the UI)
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
    # When use_positions is False, we use generic STARTER slots
    'STARTERS': 8
}

def build_team_from_config(config: dict) -> dict:
    """Create an empty team structure based on roster config."""
    cfg = {**DEFAULT_ROSTER_CONFIG, **(config or {})}
    team: dict = {}
    if cfg.get('use_positions', True):
        # Positioned lineup
        if cfg['QB'] >= 1:
            team['QB'] = None
        for i in range(1, cfg['RB'] + 1):
            team[f'RB{i}'] = None
        for i in range(1, cfg['WR'] + 1):
            team[f'WR{i}'] = None
        if cfg['TE'] >= 1:
            team['TE'] = None
        # FLEX slots
        if cfg['FLEX'] == 1:
            team['FLEX'] = None
        elif cfg['FLEX'] > 1:
            for i in range(1, cfg['FLEX'] + 1):
                team[f'FLEX{i}'] = None
        # K and DST
        if cfg['K'] >= 1:
            team['K'] = None
        if cfg['DST'] >= 1:
            team['DST'] = None
    else:
        # Positionless lineup (best ball style): STARTER slots
        starters = max(1, int(cfg.get('STARTERS', 8)))
        for i in range(1, starters + 1):
            team[f'STARTER{i}'] = None

    # Bench
    team['Bench'] = [None] * int(cfg.get('Bench', 6))
    return team

def build_position_slot_map(config: dict) -> dict:
    """Return mapping of position -> list of allowed slots to place that position."""
    cfg = {**DEFAULT_ROSTER_CONFIG, **(config or {})}
    slot_map: dict = {
        'QB': [], 'RB': [], 'WR': [], 'TE': [], 'K': [], 'DST': []
    }
    # Dedicated slots
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
        # FLEX is eligible for RB/WR/TE
        flex_slots: list = []
        if cfg['FLEX'] == 1:
            flex_slots = ['FLEX']
        elif cfg['FLEX'] > 1:
            flex_slots = [f'FLEX{i}'] if cfg['FLEX'] == 1 else [f'FLEX{i}' for i in range(1, cfg['FLEX'] + 1)]
        for s in flex_slots:
            slot_map['RB'].append(s)
            slot_map['WR'].append(s)
            slot_map['TE'].append(s)
    # STARTER slots (positionless) are eligible for any position
    starter_slots: list = []
    if not cfg.get('use_positions', True):
        starter_slots = [f'STARTER{i}' for i in range(1, int(cfg.get('STARTERS', 8)) + 1)]
    if starter_slots:
        for p in slot_map.keys():
            slot_map[p].extend(starter_slots)
    return slot_map

def build_position_targets_from_config(config: dict) -> dict:
    """Return desired counts for positional targets based on roster config."""
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
    # If positionless, softly target typical distribution
    starters = int(cfg.get('STARTERS', 8))
    # Simple heuristic split
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
    roster_config = DEFAULT_ROSTER_CONFIG.copy()
    state = {
        'roster_config': roster_config,
        'num_teams': roster_config['num_teams'],
        'our_team': build_team_from_config(roster_config),
        'drafted_by_others': [],
        'team_needs': build_position_targets_from_config(roster_config)
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
        debug_mode = request.args.get('diag') == 'true'
        diag = {}
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
        
        # Estimate current round using configured number of teams
        total_picks = len(drafted_names)
        try:
            with open(STATE_FILE, 'r') as f:
                _st = json.load(f)
                num_teams = int(_st.get('num_teams', DEFAULT_ROSTER_CONFIG['num_teams']))
        except Exception:
            num_teams = DEFAULT_ROSTER_CONFIG['num_teams']
        current_round = (total_picks // max(1, num_teams)) + 1
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
            
            # Define position needs from roster configuration
            try:
                with open(STATE_FILE, 'r') as f:
                    _st = json.load(f)
                    position_targets = build_position_targets_from_config(_st.get('roster_config', {}))
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
            from scripts.proper_model_adapter import predict_players, is_model_available
            
            # Diagnostics for model presence and files
            try:
                diag['model_available'] = bool(is_model_available())
            except Exception as _:
                diag['model_available'] = False
            try:
                diag['models_dir'] = sorted(os.listdir('models')) if os.path.isdir('models') else []
                model_path = 'models/proper_fantasy_model.pkl'
                diag['model_file_exists'] = os.path.exists(model_path)
                if diag['model_file_exists']:
                    diag['model_file_size'] = os.path.getsize(model_path)
            except Exception:
                pass

            if diag.get('model_available', False):
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
                    diag['used'] = 'SCARCITY_FALLBACK_AI_NONE'
                    enhanced_suggestions = apply_simple_scarcity_boost(available_df, our_team, current_round)
                    enhanced_suggestions['boosted_score'] = (
                        enhanced_suggestions['projected_points'] * enhanced_suggestions['scarcity_boost']
                    )
                    enhanced_suggestions = enhanced_suggestions.sort_values('boosted_score', ascending=False)
            else:
                print("❌ Proper AI model not available, using scarcity-based sorting")
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
            # Determine replacement level per position from remaining players
            for pos, k in pos_replacement_index.items():
                pos_df = enhanced_suggestions[enhanced_suggestions['position'] == pos].sort_values('boosted_score', ascending=False)
                if len(pos_df) >= k:
                    rep = pos_df['boosted_score'].iloc[k - 1]
                elif len(pos_df) > 0:
                    # Fallback to last available at position
                    rep = pos_df['boosted_score'].iloc[-1]
                else:
                    rep = 0.0
                replacement_points[pos] = float(rep)
 
            # Compute VOR and blended final score
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
 
            # Rank by final score
            enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)
        except Exception as _:
            # If anything goes wrong, keep boosted_score sort
            pass

        # Round- and roster-aware position weighting (deprioritize early QBs, delay K/DST)
        try:
            # Reconstruct simple position counts from our_team
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

                # Baseline
                w = 1.0

                # Strongly delay K/DST until late
                if pos in ('K', 'DST'):
                    if rnd <= 12:
                        return 0.2
                    elif rnd <= 14:
                        return 0.5
                    else:
                        return 0.9

                # Deprioritize QB early in 1QB formats
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

                    # If we still don't have a QB by mid rounds, gently increase urgency
                    if have_qb == 0 and rnd >= 9:
                        w = max(w, 1.05)
                    if have_qb >= 1:
                        # If QB filled, lower priority for additional QBs
                        w = min(w, 0.6)
                    return w

                # RB/WR early priority until we have two each
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

                # TE: mild early deprioritization unless we reach mid rounds without one
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

            # Apply multipliers to final_score if present, else boosted_score
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
        except Exception as _:
            pass

        # Blend AI-driven score with ADP baseline; add early-round ADP anchoring
        try:
            # ADP baseline using projected_points × scarcity_boost
            adp_baseline = []
            for _, row in enhanced_suggestions.iterrows():
                proj = float(row.get('projected_points', 0.0) or 0.0)
                sc = float(row.get('scarcity_boost', 1.0) or 1.0)
                adp_baseline.append(proj * sc)

            enhanced_suggestions = enhanced_suggestions.copy()
            enhanced_suggestions['adp_baseline'] = adp_baseline

            # Dynamic blend weight by round (mix more ADP in early/mid rounds)
            if current_round == 1:
                alpha = 0.45  # Round 1: lean a bit toward ADP
            elif current_round == 2:
                alpha = 0.65  # Round 2: still mix more ADP than before
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

            # Early/mid-round ADP anchoring multiplier
            if current_round <= 7 and 'adp_rank' in enhanced_suggestions.columns:
                adp_series = enhanced_suggestions['adp_rank'].astype(float)
                adp_min = adp_series.min(skipna=True)
                adp_max = adp_series.max(skipna=True)
                denom = (adp_max - adp_min) if pd.notna(adp_max) and pd.notna(adp_min) and (adp_max - adp_min) > 0 else 1.0
                # Slightly stronger ADP anchor in early rounds
                if current_round == 1:
                    anchor_strength = 0.06
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
        except Exception as _:
            pass
 
        # Encourage divergence from ADP when AI strongly disagrees (Rounds 1-2)
        try:
            if current_round <= 2 and 'adp_rank' in enhanced_suggestions.columns:
                # Compute AI rank (higher ai_prediction -> better rank)
                if 'ai_prediction' in enhanced_suggestions.columns:
                    enhanced_suggestions = enhanced_suggestions.copy()
                    enhanced_suggestions['ai_rank'] = enhanced_suggestions['ai_prediction'].rank(ascending=False, method='min')
                    enhanced_suggestions['adp_rank_num'] = pd.to_numeric(enhanced_suggestions['adp_rank'], errors='coerce')
                    n = max(1.0, float(len(enhanced_suggestions)))
                    # Positive when AI ranks a player better than ADP
                    enhanced_suggestions['rank_delta_norm'] = (
                        (enhanced_suggestions['adp_rank_num'] - enhanced_suggestions['ai_rank']) / n
                    ).fillna(0.0)
                    # Base exploration strength
                    gamma = 0.50
                    base_boost = 1.0 + gamma * enhanced_suggestions['rank_delta_norm']
                    # Discrete extra bump when AI strongly prefers a player
                    delta = (enhanced_suggestions['adp_rank_num'] - enhanced_suggestions['ai_rank']).fillna(0.0)
                    extra = pd.Series(1.0, index=delta.index)
                    extra = extra.where(~(delta >= 8), 1.12)
                    extra = extra.where(~(delta >= 12), 1.18)
                    boost = (base_boost * extra)
                    # Clamp to safe range to avoid extreme jumps
                    boost = boost.clip(lower=0.88, upper=1.20)
                    enhanced_suggestions['final_score'] = enhanced_suggestions['final_score'].astype(float) * boost.astype(float)
                    enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)
        except Exception as _:
            pass

        # Enforce early-round QB deprioritization and non-QB gating
        try:
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

            if qb_penalty < 1.0:
                mask_qb = enhanced_suggestions['position'] == 'QB'
                enhanced_suggestions.loc[mask_qb, 'final_score'] = (
                    enhanced_suggestions.loc[mask_qb, 'final_score'].astype(float) * qb_penalty
                )
                enhanced_suggestions = enhanced_suggestions.sort_values('final_score', ascending=False)

            # For first two rounds, fill recommendations from non-QB first
            if current_round <= 2:
                non_qb = enhanced_suggestions[enhanced_suggestions['position'] != 'QB']
                qbs = enhanced_suggestions[enhanced_suggestions['position'] == 'QB']
                enhanced_suggestions = pd.concat([non_qb, qbs], ignore_index=True)
        except Exception as _:
            pass

        # Format for frontend
        formatted_suggestions = []
        for _, suggestion in enhanced_suggestions.head(8).iterrows():  # Top 8 picks
            # Use boosted score as the optimized score
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
        
        if debug_mode:
            # Include minimal diag bundle
            try:
                with open(STATE_FILE, 'r') as f:
                    _st = json.load(f)
                diag['roster_config'] = _st.get('roster_config', {})
                diag['num_teams'] = _st.get('num_teams', DEFAULT_ROSTER_CONFIG['num_teams'])
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
    
    # Load draft state to exclude drafted players
    drafted_names = []
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            our_team = state.get('our_team', [])
            opponent_team = state.get('opponent_team', [])
            drafted_by_others = state.get('drafted_by_others', [])
            
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

            if isinstance(drafted_by_others, list):
                for player in drafted_by_others:
                    if isinstance(player, dict) and player.get('name'):
                        drafted_names.append(player['name'])

    # Filter out drafted players
    available_df = df[~df['name'].isin(drafted_names)]
    
    # Simple name search
    matches = available_df[available_df['name'].str.contains(query, case=False, na=False)]
    result = clean_nan_for_json(matches.head(20))
    return jsonify(result.to_dict('records'))

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/draft', methods=['POST', 'OPTIONS'])
def draft():
    """Draft a player"""
    try:
        if request.method == 'OPTIONS':
            print("[OPTIONS] /draft preflight")
            return ('', 204)
        data = request.get_json(silent=True) or {}
        print(f"[POST] /draft payload: {data}")
        player_name = data.get('player')
        
        if not player_name:
            return make_response(jsonify({'success': False, 'error': 'No player specified'}), 200)
        
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
            return make_response(jsonify({'success': False, 'error': 'Player not found'}), 200)
        
        # Load current state
        state = load_state()
        
        # Try to add to starting lineup first (dynamic by roster config)
        our_team = state['our_team']
        position = player_data['position']
        roster_config = state.get('roster_config', DEFAULT_ROSTER_CONFIG)
        slot_map = build_position_slot_map(roster_config)
        candidate_slots = slot_map.get(position, [])
        placed = False
        for slot in candidate_slots:
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
            return make_response(jsonify({'success': False, 'error': 'No available roster spots'}), 200)
        
        # Save state
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        print(f"[OK] Drafted: {player_name}")
        cleaned_player = clean_nan_for_json(player_data)
        return make_response(jsonify({'success': True, 'player': cleaned_player}), 200)
        
    except Exception as e:
        print(f"Draft error: {e}")
        return make_response(jsonify({'success': False, 'error': str(e)}), 200)

@app.route('/mark_taken', methods=['POST', 'OPTIONS'])
def mark_taken():
    """Mark a player as taken by another team"""
    try:
        if request.method == 'OPTIONS':
            print("[OPTIONS] /mark_taken preflight")
            return ('', 204)
        data = request.get_json(silent=True) or {}
        print(f"[POST] /mark_taken payload: {data}")
        player_name = data.get('player')
        
        if not player_name:
            return make_response(jsonify({'success': False, 'error': 'No player specified'}), 200)
        
        # Find player data
        player_data = None
        player_matches = df[df['name'].str.lower() == str(player_name).lower()]
        if not player_matches.empty:
            player_data = player_matches.iloc[0].to_dict()
        else:
            rookie_matches = rookie_df[rookie_df['name'].str.lower() == str(player_name).lower()]
            if not rookie_matches.empty:
                player_data = rookie_matches.iloc[0].to_dict()
        
        if not player_data:
            return make_response(jsonify({'success': False, 'error': 'Player not found'}), 200)
        
        # Load state dict and add to drafted_by_others
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
        else:
            state = init_state()
        if not isinstance(state.get('drafted_by_others'), list):
            state['drafted_by_others'] = []
        
        # Check if already marked
        already_drafted = any(p.get('name') == player_name for p in state['drafted_by_others'] if isinstance(p, dict))
        if not already_drafted:
            state['drafted_by_others'].append(player_data)
        
        # Save state
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        print(f"[OK] Marked taken: {player_name}")
        cleaned_player = clean_nan_for_json(player_data)
        return make_response(jsonify({'success': True, 'player': cleaned_player}), 200)
        
    except Exception as e:
        print(f"Mark taken error: {e}")
        return make_response(jsonify({'success': False, 'error': str(e)}), 200)

@app.route('/reset', methods=['POST', 'OPTIONS'])
def reset():
    """Reset the draft"""
    try:
        if request.method == 'OPTIONS':
            return ('', 204)
        init_state()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Reset error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/undo', methods=['POST', 'OPTIONS'])
def undo():
    """Undo last draft action"""
    try:
        if request.method == 'OPTIONS':
            return ('', 204)
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
        
        # If bench is empty, check starting lineup (dynamic order: reverse of keys excluding Bench)
        if last_player is None:
            start_slots = [k for k in our_team.keys() if k != 'Bench']
            for slot in reversed(start_slots):
                if our_team.get(slot) is not None:
                    last_player = our_team[slot]
                    our_team[slot] = None
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

# League settings endpoints
@app.route('/league_settings', methods=['GET', 'POST', 'OPTIONS'])
def league_settings():
    try:
        if request.method == 'OPTIONS':
            return ('', 204)
        if request.method == 'GET':
            st = load_state()
            # load_state returns dict in this file
            state = st if isinstance(st, dict) else {}
            config = state.get('roster_config', DEFAULT_ROSTER_CONFIG)
            return make_response(jsonify({'success': True, 'config': config, 'num_teams': state.get('num_teams', DEFAULT_ROSTER_CONFIG['num_teams'])}), 200)
        # POST: update config
        payload = request.get_json(silent=True) or {}
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        # Merge with defaults and coerce ints
        new_cfg = {**state.get('roster_config', DEFAULT_ROSTER_CONFIG), **(payload.get('roster_config') or {})}
        for key in ['QB','RB','WR','TE','FLEX','K','DST','Bench','STARTERS']:
            if key in new_cfg:
                try:
                    new_cfg[key] = int(new_cfg[key])
                except Exception:
                    pass
        use_positions = bool(new_cfg.get('use_positions', True))
        new_cfg['use_positions'] = use_positions
        num_teams = int(payload.get('num_teams', state.get('num_teams', DEFAULT_ROSTER_CONFIG['num_teams'])))
        # Rebuild team based on new config (reset team layout)
        state['roster_config'] = new_cfg
        state['num_teams'] = num_teams
        state['our_team'] = build_team_from_config(new_cfg)
        state['team_needs'] = build_position_targets_from_config(new_cfg)
        # Persist
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        return make_response(jsonify({'success': True}), 200)
    except Exception as e:
        return make_response(jsonify({'success': False, 'error': str(e)}), 200)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)