#!/usr/bin/env python3
"""
Add ADP Rankings to Kickers and Defense/Special Teams
- Assign realistic ADP values based on projected points
- Kickers typically drafted in rounds 12-16 (ADP 144-192)
- DST typically drafted in rounds 13-16 (ADP 156-192)
"""

import pandas as pd
import json

def add_k_dst_adp_rankings():
    print("🏈 ADDING ADP RANKINGS TO KICKERS AND DST")
    print("=" * 50)

    try:
        # Load main data files
        df = pd.read_csv('data/fantasy_metrics_2024.csv')
        with open('data/players.json', 'r') as f:
            players_data = json.load(f)
        
        print(f"📊 Loaded {len(df)} players from CSV")
        print(f"📱 Loaded {len(players_data)} players from JSON")
        
        # Separate K and DST players
        kickers_df = df[df['position'] == 'K'].copy()
        dst_df = df[df['position'] == 'DST'].copy()
        
        print(f"⚡ Found {len(kickers_df)} kickers")
        print(f"🛡️ Found {len(dst_df)} defenses")
        
        if len(kickers_df) == 0 and len(dst_df) == 0:
            print("❌ No K or DST players found!")
            return
        
        # STEP 1: Assign ADP rankings to Kickers (rounds 12-16, ADP 144-192)
        if len(kickers_df) > 0:
            # Sort kickers by projected points (descending)
            kickers_sorted = kickers_df.sort_values('projected_points', ascending=False)
            
            print("\n⚡ ASSIGNING KICKER ADP RANKINGS:")
            print("   Range: ADP 144-168 (rounds 12-14)")
            
            kicker_adp_start = 144  # Round 12 start
            kicker_adp_range = 24   # Spread across ~2 rounds
            
            for i, (idx, kicker) in enumerate(kickers_sorted.iterrows()):
                # Calculate ADP based on ranking within kickers
                adp_rank = kicker_adp_start + (i * kicker_adp_range // len(kickers_sorted))
                adp_tier = f"Kicker {i+1}"
                
                # Update DataFrame
                df.loc[df['name'] == kicker['name'], 'adp_rank'] = adp_rank
                df.loc[df['name'] == kicker['name'], 'adp_tier'] = adp_tier
                
                # Update JSON
                for player_id, player_data in players_data.items():
                    if player_data.get('name') == kicker['name'] and player_data.get('position') == 'K':
                        player_data['adp_rank'] = adp_rank
                        player_data['adp_tier'] = adp_tier
                        break
                
                print(f"  {i+1:2d}. {kicker['name']:<20} ADP: {adp_rank:3d} | {kicker['projected_points']:.1f} pts")
        
        # STEP 2: Assign ADP rankings to DST (rounds 13-16, ADP 156-192)
        if len(dst_df) > 0:
            # Sort DST by projected points (descending)
            dst_sorted = dst_df.sort_values('projected_points', ascending=False)
            
            print("\n🛡️ ASSIGNING DST ADP RANKINGS:")
            print("   Range: ADP 156-180 (rounds 13-15)")
            
            dst_adp_start = 156     # Round 13 start
            dst_adp_range = 24      # Spread across ~2 rounds
            
            for i, (idx, dst) in enumerate(dst_sorted.iterrows()):
                # Calculate ADP based on ranking within DST
                adp_rank = dst_adp_start + (i * dst_adp_range // len(dst_sorted))
                adp_tier = f"Defense {i+1}"
                
                # Update DataFrame
                df.loc[df['name'] == dst['name'], 'adp_rank'] = adp_rank
                df.loc[df['name'] == dst['name'], 'adp_tier'] = adp_tier
                
                # Update JSON
                for player_id, player_data in players_data.items():
                    if player_data.get('name') == dst['name'] and player_data.get('position') == 'DST':
                        player_data['adp_rank'] = adp_rank
                        player_data['adp_tier'] = adp_tier
                        break
                
                print(f"  {i+1:2d}. {dst['name']:<25} ADP: {adp_rank:3d} | {dst['projected_points']:.1f} pts")
        
        # Save updated files
        df.to_csv('data/fantasy_metrics_2024.csv', index=False)
        with open('data/players.json', 'w') as f:
            json.dump(players_data, f, indent=2)
        
        print(f"\n✅ SUCCESS!")
        print("✅ Updated data/fantasy_metrics_2024.csv")
        print("✅ Updated data/players.json")
        print("\n🎯 K and DST players now have realistic ADP rankings!")
        print("   - Kickers: ADP 144-168 (rounds 12-14)")
        print("   - Defenses: ADP 156-180 (rounds 13-15)")

    except Exception as e:
        print(f"❌ Error adding K/DST ADP rankings: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    add_k_dst_adp_rankings() 