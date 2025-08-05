import requests
import pandas as pd
import json
from datetime import datetime

def fetch_injury_data():
    """Fetch current injury data from nflverse using nfl_data_py library"""
    try:
        # Try to import and use nfl_data_py (official nflverse library)
        print("📊 Loading nfl_data_py library...")
        import nfl_data_py as nfl
        
        # Get current season
        current_year = datetime.now().year
        
        # Try current season first, then previous season if current isn't available
        for year in [current_year, current_year - 1]:
            try:
                print(f"📊 Fetching {year} NFL injury data from nflverse...")
                injury_df = nfl.import_injuries([year])
                
                if not injury_df.empty:
                    print(f"✅ Successfully loaded {len(injury_df)} injury records from nflverse for {year}")
                    # Filter to most recent week/date
                    if 'date' in injury_df.columns:
                        latest_date = injury_df['date'].max()
                        injury_df = injury_df[injury_df['date'] == latest_date]
                        print(f"📅 Filtered to latest date: {latest_date} ({len(injury_df)} records)")
                    
                    # Standardize column names
                    if 'player_display_name' in injury_df.columns:
                        injury_df['full_name'] = injury_df['player_display_name']
                    elif 'full_name' not in injury_df.columns and 'player_name' in injury_df.columns:
                        injury_df['full_name'] = injury_df['player_name']
                    
                    if 'injury_status' in injury_df.columns:
                        injury_df['report_status'] = injury_df['injury_status']
                    
                    if 'injury_type' in injury_df.columns:
                        injury_df['report_primary_injury'] = injury_df['injury_type']
                    elif 'injury' in injury_df.columns:
                        injury_df['report_primary_injury'] = injury_df['injury']
                    
                    return injury_df
                    
            except Exception as e:
                print(f"❌ Failed to get {year} data: {e}")
                continue
        
        print("❌ nfl_data_py returned no injury data")
        
    except ImportError:
        print("❌ nfl_data_py not installed. Installing...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nfl_data_py"])
            print("✅ Installed nfl_data_py, retrying...")
            return fetch_injury_data()  # Retry after installation
        except Exception as e:
            print(f"❌ Failed to install nfl_data_py: {e}")
    
    except Exception as e:
        print(f"❌ nfl_data_py error: {e}")
    
    # Fallback: Try direct nflverse GitHub API
    try:
        print("📊 Trying direct nflverse GitHub API...")
        urls = [
            "https://github.com/nflverse/nflverse-data/releases/latest/download/injuries.csv",
            "https://github.com/nflverse/nfldata/releases/latest/download/injuries.csv"
        ]
        
        for url in urls:
            try:
                print(f"📊 Fetching from: {url}")
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                
                from io import StringIO
                injury_df = pd.read_csv(StringIO(response.text))
                
                if not injury_df.empty:
                    print(f"✅ Successfully loaded {len(injury_df)} injury records from GitHub")
                    
                    # Standardize column names for consistency
                    if 'player_display_name' in injury_df.columns:
                        injury_df['full_name'] = injury_df['player_display_name']
                    elif 'player_name' in injury_df.columns and 'full_name' not in injury_df.columns:
                        injury_df['full_name'] = injury_df['player_name']
                    
                    if 'injury_status' in injury_df.columns:
                        injury_df['report_status'] = injury_df['injury_status']
                    
                    if 'injury_type' in injury_df.columns:
                        injury_df['report_primary_injury'] = injury_df['injury_type']
                    elif 'injury' in injury_df.columns:
                        injury_df['report_primary_injury'] = injury_df['injury']
                    
                    return injury_df
                    
            except Exception as e:
                print(f"❌ Failed {url}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ GitHub API error: {e}")
    
    # Final fallback: Return empty DataFrame with expected structure
    print("⚠️ Could not fetch real injury data, returning empty DataFrame")
    return pd.DataFrame(columns=['full_name', 'report_status', 'report_primary_injury'])

def normalize_name_for_matching(name):
    """Normalize player names for matching between datasets"""
    if pd.isna(name):
        return ""
    
    # Remove common suffixes and prefixes
    name = str(name).strip()
    name = name.replace("Jr.", "").replace("Sr.", "").replace("III", "").replace("II", "")
    
    # Handle common name variations
    name_mappings = {
        "Kenneth Walker": "Ken Walker",
        "Joshua Palmer": "Josh Palmer",
        "Jonathan Taylor": "Jonathan Taylor",
        "Joseph Mixon": "Joe Mixon",
        "Christopher Olave": "Chris Olave",
        "Matthew Stafford": "Matt Stafford",
        "Joshua Jacobs": "Josh Jacobs",
        "Nicholas Chubb": "Nick Chubb"
    }
    
    if name in name_mappings:
        name = name_mappings[name]
    
    # Convert to lowercase for matching
    return name.lower().strip()

def add_injury_status_to_dataframe(player_df, injury_df):
    """Add injury status information to player DataFrame"""
    if injury_df.empty:
        print("⚠️ No injury data available, adding empty injury columns")
        player_df['injury_status'] = None
        player_df['injury_description'] = ""
        player_df['injury_icon'] = ""
        return player_df
    
    print(f"🔍 Matching {len(player_df)} players against {len(injury_df)} injury records...")
    
    # Create normalized name columns for matching
    player_df['norm_name'] = player_df['name'].apply(normalize_name_for_matching)
    injury_df['norm_name'] = injury_df['full_name'].apply(normalize_name_for_matching)
    
    # Initialize injury columns
    player_df['injury_status'] = None
    player_df['injury_description'] = ""
    player_df['injury_icon'] = ""
    
    # Map injury status to icons and descriptions (no icons per user request)
    status_mapping = {
        'Out': {'icon': '', 'desc': 'Out'},
        'Doubtful': {'icon': '', 'desc': 'Doubtful'}, 
        'Questionable': {'icon': '', 'desc': 'Questionable'},
        'Probable': {'icon': '', 'desc': 'Probable'},
        'PUP': {'icon': '', 'desc': 'PUP'},
        'IR': {'icon': '', 'desc': 'Injured Reserve'},
        'DNP': {'icon': '', 'desc': 'Did Not Practice'}
    }
    
    matched_count = 0
    
    # Match players with injury data
    for idx, player in player_df.iterrows():
        player_norm = player['norm_name']
        
        # Find matching injury record
        injury_matches = injury_df[injury_df['norm_name'] == player_norm]
        
        if not injury_matches.empty:
            injury_record = injury_matches.iloc[0]  # Take first match
            status = injury_record.get('report_status', '')
            injury_type = injury_record.get('report_primary_injury', '')
            
            if status and status in status_mapping:
                player_df.at[idx, 'injury_status'] = status
                player_df.at[idx, 'injury_icon'] = status_mapping[status]['icon']
                player_df.at[idx, 'injury_description'] = f"{status_mapping[status]['desc']}: {injury_type}" if injury_type else status_mapping[status]['desc']
                matched_count += 1
    
    print(f"✅ Found injury status for {matched_count} players")
    
    # Clean up temporary columns
    player_df.drop('norm_name', axis=1, inplace=True)
    
    return player_df

def get_current_injury_data():
    """Main function to get current injury data"""
    try:
        return fetch_injury_data()
    except Exception as e:
        print(f"❌ Error in get_current_injury_data: {e}")
        return pd.DataFrame(columns=['full_name', 'report_status', 'report_primary_injury'])

if __name__ == "__main__":
    print("🏥 Testing injury data fetching...")
    
    injury_data = get_current_injury_data()
    
    if not injury_data.empty:
        print(f"\n📊 Sample injury data ({len(injury_data)} total records):")
        print(injury_data.head())
        
        if 'report_status' in injury_data.columns:
            print(f"\n📈 Injury status distribution:")
            print(injury_data['report_status'].value_counts())
    else:
        print("❌ No injury data retrieved") 