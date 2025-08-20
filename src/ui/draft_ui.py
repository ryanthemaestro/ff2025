#!/usr/bin/env python3
"""
Clean Fantasy Football Draft UI
Uses ADP rankings for player selection and basic team building
"""
import sys
import os

# Set environment for headless operation
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

sys.path.insert(0, '../..')
# Resolve absolute project base directory for reliable file/template paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import json

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Global variables
df = None
our_team = {
    'QB': None, 'RB1': None, 'RB2': None, 'WR1': None, 'WR2': None,
    'TE': None, 'FLEX': None, 'K': None, 'DST': None,
    'Bench': [None] * 6
}
drafted_by_others = []

def load_player_data():
    """Load and process ADP player data"""
    global df
    print("🏈 Loading FantasyPros ADP Rankings...")
    
    try:
        data_path = os.path.join(BASE_DIR, 'data', 'FantasyPros_2025_Overall_ADP_Rankings.csv')
        df = pd.read_csv(data_path, on_bad_lines='skip')
        
        # Clean up column names
        df = df.rename(columns={
            'Player': 'name',
            'Rank': 'adp_rank', 
            'POS': 'position',
            'Bye': 'bye_week',
            'Team': 'team'
        })
        
        # Clean position data
        df['position'] = df['position'].str.replace(r'\d+$', '', regex=True)
        df['adp_rank'] = pd.to_numeric(df['adp_rank'], errors='coerce')
        
        # Calculate projected points based on ADP
        def calc_projected_points(row):
            adp = row['adp_rank']
            pos = row['position']
            if pd.isna(adp):
                return 100
            
            base_scores = {'QB': 350, 'RB': 280, 'WR': 260, 'TE': 180, 'K': 140, 'DST': 130}
            base = base_scores.get(pos, 200)
            decay_rate = 0.8
            projected = max(50, base - (adp * decay_rate))
            return round(projected, 1)
        
        df['projected_points'] = df.apply(calc_projected_points, axis=1)
        df = df.sort_values('adp_rank', ascending=True)
        
        print(f"✅ Loaded {len(df)} players")
        print(f"   Positions: {df['position'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        df = pd.DataFrame()

# Eagerly load data at import time (useful for WSGI/gunicorn)
try:
    load_player_data()
except Exception as _e:
    print(f"⚠️ Deferred data load due to error: {_e}")

def clean_for_json(data):
    """Clean data for JSON serialization"""
    if isinstance(data, pd.DataFrame):
        return data.where(pd.notna(data), None)
    return data

def get_available_players():
    """Get players not yet drafted"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Get names of drafted players
    drafted_names = []
    
    # From our team
    for pos, player in our_team.items():
        if pos != 'Bench' and isinstance(player, dict) and player.get('name'):
            drafted_names.append(player['name'])
    
    # From bench
    if 'Bench' in our_team:
        for bench_player in our_team['Bench']:
            if isinstance(bench_player, dict) and bench_player.get('name'):
                drafted_names.append(bench_player['name'])
    
    # From others
    for player in drafted_by_others:
        if isinstance(player, dict) and player.get('name'):
            drafted_names.append(player['name'])
    
    # Filter out drafted players
    available = df[~df['name'].isin(drafted_names)]
    return available.copy()

def assign_player_to_lineup(player_dict):
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
    
    return None

@app.route('/')
def index():
    """Main page"""
    available_players = get_available_players()
    top_available = available_players.head(50)
    cleaned_available = clean_for_json(top_available)
    
    context = {
        'available_players': cleaned_available.to_dict('records'),
        'our_team': our_team,
        'team_with_byes': our_team,
        'drafted_by_others': drafted_by_others,
        'team_needs': calculate_team_needs()
    }
    
    try:
        response = make_response(render_template('index.html', **context))
        response.headers['Cache-Control'] = 'no-cache'
        return response
    except Exception as e:
        return f"""
        <html>
        <head><title>Fantasy Football Draft UI</title></head>
        <body>
            <h1>🏈 Fantasy Football Draft UI</h1>
            <p>✅ Server running! {len(available_players)} players available.</p>
            <p>Template error: {e}</p>
            <h2>Top Available Players</h2>
            <ul>
                {''.join([f'<li>{p["name"]} - {p["position"]} (ADP: {p["adp_rank"]})</li>' 
                         for _, p in top_available.head(10).iterrows()])}
            </ul>
            <p><a href="/api/suggest">View API</a></p>
        </body>
        </html>
        """

@app.route('/suggest')
def suggest():
    """Player suggestions API"""
    available_players = get_available_players()
    position_filter = request.args.get('position')
    
    if position_filter in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
        filtered = available_players[available_players['position'] == position_filter].head(20)
    else:
        # Return top ADP players
        filtered = available_players.head(20)
    
    cleaned = clean_for_json(filtered)
    return jsonify(cleaned.to_dict('records'))

@app.route('/search')
def search():
    """Search for players by name (case-insensitive substring)"""
    query = (request.args.get('q') or '').strip()
    available_players = get_available_players()
    if query:
        mask = available_players['name'].str.contains(query, case=False, na=False)
        results = available_players[mask].head(50)
    else:
        results = available_players.head(50)
    cleaned = clean_for_json(results)
    return jsonify(cleaned.to_dict('records'))

@app.route('/draft', methods=['POST'])
def draft_player():
    """Draft a player to our team"""
    try:
        data = request.get_json()
        player_name = data.get('player')
        
        if not player_name:
            return jsonify({'success': False, 'message': 'No player specified'})
        
        available_players = get_available_players()
        player_row = available_players[available_players['name'] == player_name]
        
        if player_row.empty:
            return jsonify({'success': False, 'message': 'Player not found'})
        
        player_dict = player_row.iloc[0].to_dict()
        # Convert pandas types to native Python types
        for key, value in player_dict.items():
            if pd.isna(value):
                player_dict[key] = None
            elif hasattr(value, 'item'):
                player_dict[key] = value.item()
        
        assigned_slot = assign_player_to_lineup(player_dict)
        if assigned_slot is None:
            return jsonify({'success': False, 'message': 'No roster spots available'})
        
        return jsonify({
            'success': True,
            'message': f'Drafted {player_name} to {assigned_slot}',
            'assigned_slot': assigned_slot
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/mark_taken', methods=['POST'])
def mark_taken():
    """Mark a player as taken by another team"""
    try:
        data = request.get_json()
        player_name = data.get('player')
        
        if not player_name:
            return jsonify({'success': False, 'message': 'No player specified'})
        
        available_players = get_available_players()
        player_row = available_players[available_players['name'] == player_name]
        
        if player_row.empty:
            return jsonify({'success': False, 'message': 'Player not found'})
        
        player_dict = player_row.iloc[0].to_dict()
        # Convert pandas types
        for key, value in player_dict.items():
            if pd.isna(value):
                player_dict[key] = None
            elif hasattr(value, 'item'):
                player_dict[key] = value.item()
        
        drafted_by_others.append(player_dict)
        
        return jsonify({'success': True, 'message': f'Marked {player_name} as taken'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/players')
def api_players():
    """API endpoint for player data"""
    available_players = get_available_players()
    top_players = available_players.head(50)
    cleaned = clean_for_json(top_players)
    return jsonify(cleaned.to_dict('records'))

@app.route('/health')
def health():
    """Health check endpoint"""
    available_players = get_available_players()
    return jsonify({
        'status': 'healthy',
        'total_players': len(df) if df is not None else 0,
        'available_players': len(available_players),
        'our_team_filled': sum(1 for p in our_team.values() if p is not None and p != [None]*6)
    })

def calculate_team_needs():
    """Calculate what positions we still need"""
    needs = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'K': 1, 'DST': 1}
    
    for position, player in our_team.items():
        if position != 'Bench' and isinstance(player, dict) and player.get('position'):
            pos = player['position']
            if pos in needs:
                needs[pos] = max(0, needs[pos] - 1)
    
    return needs

if __name__ == '__main__':
    load_player_data()
    
    print("🚀 Starting Clean Fantasy Football Draft UI...")
    print("📍 Server available at: http://localhost:5000")
    print("🔧 Press Ctrl+C to stop")
    print("")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False
    ) 