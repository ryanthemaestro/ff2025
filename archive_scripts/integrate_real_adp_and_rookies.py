#!/usr/bin/env python3
"""
Integrate Real ADP and Rookie Data from FantasyPros and NFLverse
Enhanced name matching for different formats

REAL NFL DATA ONLY
"""

import pandas as pd
import nfl_data_py as nfl
import json
import re
from difflib import SequenceMatcher

class ADPRookieIntegrator:
    
    def __init__(self):
        self.adp_matches = 0
        self.rookie_matches = 0
        
    def normalize_name(self, name):
        """Normalize a name for comparison"""
        if pd.isna(name) or not name:
            return ""
        
        name = str(name).strip()
        # Remove quotes
        name = name.replace('"', '').replace("'", "")
        # Convert to uppercase
        name = name.upper()
        # Remove Jr., Sr., III, etc.
        name = re.sub(r'\s+(JR\.?|SR\.?|III|IV|V)$', '', name)
        return name

    def convert_full_name_to_abbreviated(self, full_name):
        """Convert 'Ja'Marr Chase' to 'J.CHASE' format"""
        if pd.isna(full_name) or not full_name:
            return ""
            
        # Clean the name
        name = str(full_name).strip().replace('"', '')
        
        # Handle apostrophes (Ja'Marr -> J)
        name = name.replace("'", "")
        
        # Split into parts
        parts = name.split()
        if len(parts) < 2:
            return name.upper()
        
        # Take first letter of first name + last name
        first_initial = parts[0][0].upper()
        last_name = parts[-1].upper()
        
        # Remove Jr., Sr., etc. from last name
        last_name = re.sub(r'\s+(JR\.?|SR\.?|III|IV|V)$', '', last_name)
        
        return f"{first_initial}.{last_name}"

    def _names_similar(self, name1, name2):
        """Enhanced name similarity matching"""
        # Handle NaN values
        if pd.isna(name1) or pd.isna(name2):
            return False

        # Convert to string in case of float values
        name1 = str(name1).strip()
        name2 = str(name2).strip()
        
        if not name1 or not name2:
            return False

        # Normalize both names
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return True
        
        # Try converting ADP name to abbreviated format
        if len(name1.split()) > 1:  # name1 is likely full name from ADP
            abbreviated = self.convert_full_name_to_abbreviated(name1)
            if abbreviated == norm2:
                return True
        
        if len(name2.split()) > 1:  # name2 is likely full name
            abbreviated = self.convert_full_name_to_abbreviated(name2)
            if abbreviated == norm1:
                return True
        
        # Split and compare parts
        parts1 = norm1.replace('.', ' ').split()
        parts2 = norm2.replace('.', ' ').split()
        
        # Check if last names match and first initial matches
        if len(parts1) >= 2 and len(parts2) >= 2:
            if parts1[-1] == parts2[-1] and parts1[0][0] == parts2[0][0]:
                return True
        
        # Fuzzy matching for similar names
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity > 0.85

    def match_players_with_adp(self, df, adp_df):
        """Match players with ADP data using enhanced name matching"""
        print("🎯 MATCHING PLAYERS WITH ADP DATA")
        print("=" * 45)
        
        matches = 0
        total_adp = len(adp_df)
        
        for _, adp_row in adp_df.iterrows():
            adp_player_name = adp_row.get('Player', '')
            
            # Skip invalid names
            if pd.isna(adp_player_name) or not adp_player_name:
                continue
                
            adp_rank = adp_row.get('Rank', '')
            position = adp_row.get('POS', '')
            bye_week = adp_row.get('Bye', '')
            avg_adp = adp_row.get('AVG', '')
            
            # Find matching player in our dataset
            match_found = False
            for idx, player_row in df.iterrows():
                player_name = player_row.get('name', '')
                
                if self._names_similar(adp_player_name, player_name):
                    # Update player with ADP data
                    df.loc[idx, 'adp'] = adp_rank
                    df.loc[idx, 'adp_rank'] = adp_rank
                    df.loc[idx, 'adp_avg'] = avg_adp
                    df.loc[idx, 'consensus_bye_week'] = bye_week
                    
                    # Calculate ADP tier
                    try:
                        rank_num = int(adp_rank)
                        if rank_num <= 30:
                            tier = "Elite (1-30)"
                        elif rank_num <= 60:
                            tier = "High-End (31-60)"
                        elif rank_num <= 100:
                            tier = "Mid-Tier (61-100)"
                        elif rank_num <= 150:
                            tier = "Late-Round (101-150)"
                        else:
                            tier = "Deep League (151+)"
                        df.loc[idx, 'adp_tier'] = tier
                    except:
                        df.loc[idx, 'adp_tier'] = "Unknown"
                    
                    matches += 1
                    match_found = True
                    
                    print(f"  ✅ {adp_player_name} -> {player_name} (ADP: {adp_rank})")
                    break
            
            if not match_found:
                # Try abbreviated format
                abbreviated = self.convert_full_name_to_abbreviated(adp_player_name)
                for idx, player_row in df.iterrows():
                    player_name = player_row.get('name', '')
                    if player_name.upper() == abbreviated:
                        df.loc[idx, 'adp'] = adp_rank
                        df.loc[idx, 'adp_rank'] = adp_rank
                        df.loc[idx, 'adp_avg'] = avg_adp
                        df.loc[idx, 'consensus_bye_week'] = bye_week
                        
                        try:
                            rank_num = int(adp_rank)
                            if rank_num <= 30:
                                tier = "Elite (1-30)"
                            elif rank_num <= 60:
                                tier = "High-End (31-60)"
                            elif rank_num <= 100:
                                tier = "Mid-Tier (61-100)"
                            elif rank_num <= 150:
                                tier = "Late-Round (101-150)"
                            else:
                                tier = "Deep League (151+)"
                            df.loc[idx, 'adp_tier'] = tier
                        except:
                            df.loc[idx, 'adp_tier'] = "Unknown"
                        
                        matches += 1
                        print(f"  ✅ {adp_player_name} -> {player_name} (Abbreviated: {abbreviated}, ADP: {adp_rank})")
                        break
        
        print(f"\n📊 ADP MATCHING RESULTS:")
        print(f"   ✅ {matches} players matched with ADP data")
        print(f"   ❌ {total_adp - matches} ADP players not found")
        print(f"   📈 Match rate: {matches/total_adp*100:.1f}%")
        
        return matches

    def integrate_adp_and_rookies(self):
        """Main integration function"""
        print("🏈 INTEGRATING REAL ADP AND ROOKIE DATA")
        print("=" * 55)
        
        try:
            # Load main dataset
            df = pd.read_csv('data/fantasy_metrics_2024.csv')
            print(f"📊 Loaded {len(df)} players from main dataset")
            
            # Load ADP data with better error handling
            print("\n📈 LOADING ADP DATA FROM FANTASYPROS...")
            adp_df = pd.read_csv('data/FantasyPros_2025_Overall_ADP_Rankings.csv',
                                on_bad_lines='skip',
                                quoting=1)
            print(f"📊 Loaded {len(adp_df)} players from FantasyPros ADP CSV")
            
            # Show sample data formats
            print(f"\n🔍 SAMPLE NAME FORMATS:")
            print(f"   ADP CSV: {adp_df['Player'].iloc[1:4].tolist()}")
            print(f"   Our data: {df['name'].iloc[1:4].tolist()}")
            
            # Match ADP data
            adp_matches = self.match_players_with_adp(df, adp_df)
            
            # Load and process rookies (from nflverse draft data)
            print(f"\n🆕 LOADING 2024 ROOKIE DATA...")
            try:
                draft_picks = nfl.import_draft_picks([2024])
                print(f"📊 Loaded {len(draft_picks)} 2024 NFL draft picks")
                
                rookie_matches = 0
                for _, pick in draft_picks.iterrows():
                    draft_name = pick.get('pfr_player_name', pick.get('player_name', ''))
                    if pd.isna(draft_name):
                        continue
                    
                    # Try to match with our dataset
                    for idx, player_row in df.iterrows():
                        player_name = player_row.get('name', '')
                        if self._names_similar(draft_name, player_name):
                            df.loc[idx, 'is_rookie'] = True
                            df.loc[idx, 'draft_round'] = pick.get('round', '')
                            df.loc[idx, 'draft_pick'] = pick.get('pick', '')
                            rookie_matches += 1
                            print(f"  🆕 Rookie: {draft_name} -> {player_name}")
                            break
                
                print(f"📊 Rookie matching: {rookie_matches} rookies identified")
                
            except Exception as e:
                print(f"⚠️ Rookie data loading failed: {e}")
                rookie_matches = 0
            
            # Calculate ADP distribution
            adp_players = df[df['adp'].notna()]
            tier_counts = adp_players['adp_tier'].value_counts()
            
            # Save updated dataset
            df.to_csv('data/fantasy_metrics_2024.csv', index=False)
            print(f"\n✅ INTEGRATION COMPLETE!")
            print(f"   📈 ADP integrated: {adp_matches} players")
            print(f"   🆕 Rookies identified: {rookie_matches} players")
            print(f"   💾 Saved updated data to fantasy_metrics_2024.csv")
            
            if len(tier_counts) > 0:
                print(f"\n📊 ADP TIER DISTRIBUTION:")
                for tier, count in tier_counts.items():
                    print(f"   {tier}: {count} players")
            
            return True
            
        except Exception as e:
            print(f"❌ Integration failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    integrator = ADPRookieIntegrator()
    integrator.integrate_adp_and_rookies() 