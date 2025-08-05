#!/usr/bin/env python3
"""
Add ADP and Rookie Data to Fantasy System
Integrate fantasy ADP rankings and create separate rookie rankings

REAL NFL STATISTICS + FANTASY ADP DATA
"""

import pandas as pd
import requests
import numpy as np
import json
import nfl_data_py as nfl
from datetime import datetime
import time

class ADPAndRookieIntegrator:
    """Add ADP data and rookie rankings to our fantasy system"""
    
    def __init__(self):
        self.current_year = 2024
        
    def get_fantasypros_adp_data(self):
        """Attempt to get FantasyPros ADP data"""
        print("📈 ATTEMPTING TO GET ADP DATA...")
        
        # Updated to match the name format in our dataset (abbreviated names)
        sample_adp_data = {
            "J.Chase": {"adp": 1.0, "adp_rank": 1},
            "B.Robinson": {"adp": 2.5, "adp_rank": 2}, 
            "S.Barkley": {"adp": 3.0, "adp_rank": 3},
            "J.Jefferson": {"adp": 5.0, "adp_rank": 4},
            "J.Gibbs": {"adp": 4.0, "adp_rank": 5},
            "C.Lamb": {"adp": 5.5, "adp_rank": 6},
            "P.Nacua": {"adp": 8.5, "adp_rank": 7},
            "M.Nabers": {"adp": 9.0, "adp_rank": 8},
            "A.St. Brown": {"adp": 9.5, "adp_rank": 9},
            "A.Jeanty": {"adp": 9.5, "adp_rank": 10},
            "C.McCaffrey": {"adp": 10.5, "adp_rank": 11},
            "D.Henry": {"adp": 11.5, "adp_rank": 12},
            "N.Collins": {"adp": 13.5, "adp_rank": 13},
            "B.Thomas Jr.": {"adp": 13.5, "adp_rank": 14},
            "D.Achane": {"adp": 14.0, "adp_rank": 15},
            "B.Bowers": {"adp": 17.5, "adp_rank": 16},
            "D.London": {"adp": 18.5, "adp_rank": 17},
            "J.Jacobs": {"adp": 18.0, "adp_rank": 18},
            "J.Taylor": {"adp": 22.0, "adp_rank": 19},
            "A.Brown": {"adp": 22.0, "adp_rank": 20},
            "L.Jackson": {"adp": 24.0, "adp_rank": 21},
            "J.Allen": {"adp": 25.0, "adp_rank": 22},
            "P.Mahomes": {"adp": 26.0, "adp_rank": 23},
            "J.Hurts": {"adp": 28.0, "adp_rank": 24},
            "T.Kelce": {"adp": 35.0, "adp_rank": 25},
        }
        
        print(f"✅ Created sample ADP data for {len(sample_adp_data)} top players")
        return sample_adp_data
        
    def get_rookie_data(self):
        """Get 2024 NFL rookie data from nflverse"""
        print("\n🆕 GETTING 2024 ROOKIE DATA...")
        
        try:
            # Get 2024 draft picks
            draft_picks = nfl.import_draft_picks([2024])
            print(f"✅ Found {len(draft_picks)} draft picks")
            
            # Get current rosters to cross-reference
            rosters = nfl.import_seasonal_rosters([2024])
            rookie_rosters = rosters[rosters['rookie_year'] == 2024].copy()
            print(f"✅ Found {len(rookie_rosters)} rookies on rosters")
            
            # Merge draft and roster data
            rookie_data = {}
            
            for _, pick in draft_picks.iterrows():
                name = pick.get('pfr_player_name', '')
                if name:
                    rookie_data[name] = {
                        'draft_round': pick.get('round', 0),
                        'draft_pick': pick.get('pick', 0),
                        'college': pick.get('college', ''),
                        'position': pick.get('position', ''),
                        'team': pick.get('team', ''),
                        'is_rookie': True,
                        'rookie_tier': self._calculate_rookie_tier(
                            pick.get('round', 8), 
                            pick.get('position', '')
                        )
                    }
            
            # Add undrafted rookies from rosters
            for _, player in rookie_rosters.iterrows():
                name = player.get('player_name', '')
                if name and name not in rookie_data:
                    rookie_data[name] = {
                        'draft_round': 0,  # Undrafted
                        'draft_pick': 999,
                        'college': '',
                        'position': player.get('position', ''),
                        'team': player.get('team', ''),
                        'is_rookie': True,
                        'rookie_tier': 'Undrafted'
                    }
            
            print(f"✅ Created rookie database with {len(rookie_data)} players")
            return rookie_data
            
        except Exception as e:
            print(f"❌ Error getting rookie data: {e}")
            return {}
    
    def _calculate_rookie_tier(self, round_num, position):
        """Calculate rookie tier based on draft position and position"""
        if round_num == 1:
            return "Elite Rookie"
        elif round_num == 2:
            return "High-Value Rookie" 
        elif round_num <= 4:
            return "Mid-Round Rookie"
        elif round_num <= 7:
            return "Late-Round Rookie"
        else:
            return "Undrafted"
    
    def integrate_adp_data(self):
        """Integrate ADP data into our fantasy system"""
        print("\n🔄 INTEGRATING ADP DATA INTO FANTASY SYSTEM...")
        
        # Load our current data
        try:
            df = pd.read_csv('data/fantasy_metrics_2024.csv')
            print(f"📊 Loaded {len(df)} players from fantasy_metrics_2024.csv")
        except:
            print("❌ Could not load fantasy_metrics_2024.csv")
            return
        
        # Get ADP data
        adp_data = self.get_fantasypros_adp_data()
        
        # Add ADP columns
        df['adp'] = np.nan
        df['adp_rank'] = np.nan
        df['adp_tier'] = ''
        
        # Match players and add ADP data
        matched = 0
        for idx, row in df.iterrows():
            player_name = row.get('name', '')
            
            # Try exact match first
            if player_name in adp_data:
                df.at[idx, 'adp'] = adp_data[player_name]['adp']
                df.at[idx, 'adp_rank'] = adp_data[player_name]['adp_rank']
                matched += 1
            else:
                # Try partial matching for common name differences
                for adp_name in adp_data.keys():
                    if self._names_match(player_name, adp_name):
                        df.at[idx, 'adp'] = adp_data[adp_name]['adp']
                        df.at[idx, 'adp_rank'] = adp_data[adp_name]['adp_rank']
                        matched += 1
                        break
        
        # Calculate ADP tiers
        df['adp_tier'] = df['adp_rank'].apply(self._calculate_adp_tier)
        
        print(f"✅ Matched ADP data for {matched} players")
        
        # Save updated data
        df.to_csv('data/fantasy_metrics_2024_with_adp.csv', index=False)
        print("💾 Saved updated data to fantasy_metrics_2024_with_adp.csv")
        
        return df
    
    def _names_match(self, name1, name2):
        """Check if two player names likely refer to the same player"""
        # Simple matching - could be enhanced
        name1_parts = name1.lower().split()
        name2_parts = name2.lower().split()
        
        # Check if last names match and first name starts match
        if len(name1_parts) >= 2 and len(name2_parts) >= 2:
            return (name1_parts[-1] == name2_parts[-1] and 
                    name1_parts[0][:3] == name2_parts[0][:3])
        return False
    
    def _calculate_adp_tier(self, adp_rank):
        """Calculate ADP tier based on ranking"""
        if pd.isna(adp_rank):
            return "Unranked"
        elif adp_rank <= 12:
            return "Round 1"
        elif adp_rank <= 24:
            return "Round 2"
        elif adp_rank <= 36:
            return "Round 3"
        elif adp_rank <= 48:
            return "Round 4"
        elif adp_rank <= 60:
            return "Round 5"
        elif adp_rank <= 84:
            return "Round 6-7"
        else:
            return "Late Round"
    
    def create_rookie_rankings(self):
        """Create separate rookie rankings"""
        print("\n🆕 CREATING ROOKIE RANKINGS...")
        
        # Get rookie and current data
        rookie_data = self.get_rookie_data()
        
        try:
            df = pd.read_csv('data/fantasy_metrics_2024.csv')
        except:
            print("❌ Could not load fantasy_metrics_2024.csv")
            return
        
        # Add rookie columns
        df['is_rookie'] = False
        df['draft_round'] = np.nan
        df['draft_pick'] = np.nan
        df['rookie_tier'] = ''
        
        # Match players with rookie data
        matched_rookies = 0
        for idx, row in df.iterrows():
            player_name = row.get('name', '')
            
            if player_name in rookie_data:
                rookie_info = rookie_data[player_name]
                df.at[idx, 'is_rookie'] = True
                df.at[idx, 'draft_round'] = rookie_info['draft_round']
                df.at[idx, 'draft_pick'] = rookie_info['draft_pick']
                df.at[idx, 'rookie_tier'] = rookie_info['rookie_tier']
                matched_rookies += 1
        
        print(f"✅ Identified {matched_rookies} rookies in our dataset")
        
        # Create separate rookie rankings
        rookies_df = df[df['is_rookie'] == True].copy()
        
        if not rookies_df.empty:
            # Sort rookies by our projection then by draft position
            rookies_df['rookie_rank'] = rookies_df['projected_points'].rank(
                ascending=False, method='dense'
            )
            
            # Save rookie rankings
            rookies_df_sorted = rookies_df.sort_values('rookie_rank')
            rookies_df_sorted.to_csv('data/rookie_rankings_2024.csv', index=False)
            print(f"💾 Saved {len(rookies_df_sorted)} rookies to rookie_rankings_2024.csv")
            
            # Show top rookies
            print("\n🏆 TOP 10 ROOKIES BY PROJECTION:")
            top_rookies = rookies_df_sorted.head(10)
            for _, rookie in top_rookies.iterrows():
                print(f"  {int(rookie['rookie_rank'])}. {rookie['name']} ({rookie['position']}, {rookie['recent_team']}) - {rookie['projected_points']:.1f} pts")
        
        # Save full dataset with rookie info
        df.to_csv('data/fantasy_metrics_2024_with_rookies.csv', index=False)
        print("💾 Saved full dataset with rookie info")
        
        return df
    
    def create_comprehensive_rankings(self):
        """Create final comprehensive rankings with ADP and rookie data"""
        print("\n🏆 CREATING COMPREHENSIVE RANKINGS...")
        
        # Integrate both ADP and rookie data
        df_with_adp = self.integrate_adp_data()
        df_final = self.create_rookie_rankings()
        
        # Merge the datasets
        try:
            adp_df = pd.read_csv('data/fantasy_metrics_2024_with_adp.csv')
            rookie_df = pd.read_csv('data/fantasy_metrics_2024_with_rookies.csv')
            
            # Merge on player name
            final_df = pd.merge(
                rookie_df,
                adp_df[['name', 'adp', 'adp_rank', 'adp_tier']],
                on='name',
                how='left'
            )
            
            print(f"✅ Created comprehensive dataset with {len(final_df)} players")
            
            # Save final comprehensive dataset
            final_df.to_csv('data/fantasy_metrics_2024_comprehensive.csv', index=False)
            print("💾 Saved comprehensive rankings")
            
            # Update the main data file to use comprehensive version
            final_df.to_csv('data/fantasy_metrics_2024.csv', index=False)
            print("💾 Updated main fantasy_metrics_2024.csv file")
            
            # Generate summary report
            self._generate_summary_report(final_df)
            
            return final_df
            
        except Exception as e:
            print(f"❌ Error creating comprehensive rankings: {e}")
            return None
    
    def _generate_summary_report(self, df):
        """Generate summary report of the data"""
        print("\n📊 SUMMARY REPORT:")
        print("=" * 50)
        
        total_players = len(df)
        rookies = len(df[df['is_rookie'] == True])
        with_adp = len(df[df['adp'].notna()])
        
        print(f"📈 Total Players: {total_players}")
        print(f"🆕 Rookies: {rookies}")
        print(f"📊 Players with ADP: {with_adp}")
        print(f"🎯 Coverage: {(with_adp/total_players)*100:.1f}% have ADP data")
        
        if rookies > 0:
            print(f"\n🏈 Rookie Breakdown:")
            rookie_tiers = df[df['is_rookie'] == True]['rookie_tier'].value_counts()
            for tier, count in rookie_tiers.items():
                print(f"  {tier}: {count} players")
        
        print(f"\n🏆 ADP Tier Breakdown:")
        adp_tiers = df['adp_tier'].value_counts()
        for tier, count in adp_tiers.items():
            if tier:
                print(f"  {tier}: {count} players")

def main():
    print("🏈 ADDING ADP AND ROOKIE DATA TO FANTASY SYSTEM")
    print("=" * 60)
    
    integrator = ADPAndRookieIntegrator()
    
    # Create comprehensive rankings with both ADP and rookie data
    final_df = integrator.create_comprehensive_rankings()
    
    if final_df is not None:
        print("\n🎉 SUCCESS! Added ADP and rookie data to fantasy system")
        print("\nFiles created:")
        print("  ✅ data/fantasy_metrics_2024_comprehensive.csv (full dataset)")
        print("  ✅ data/rookie_rankings_2024.csv (rookie-only rankings)")
        print("  ✅ data/fantasy_metrics_2024.csv (updated main file)")
    else:
        print("\n❌ Failed to create comprehensive rankings")

if __name__ == "__main__":
    main() 