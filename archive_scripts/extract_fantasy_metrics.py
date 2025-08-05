#!/usr/bin/env python3
"""
Extract fantasy-relevant metrics from NFLsavant play-by-play data
Converts raw play-by-play into useful features like target share, snap count, red zone usage
"""

import pandas as pd
import re
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class FantasyMetricsExtractor:
    def __init__(self, pbp_file):
        """Initialize with path to play-by-play CSV file"""
        print(f"🏈 Loading play-by-play data from {pbp_file}...")
        self.df = pd.read_csv(pbp_file)
        print(f"   📊 Loaded {len(self.df):,} plays from {self.df['SeasonYear'].iloc[0]} season")
        self.player_stats = defaultdict(lambda: defaultdict(int))
        self.team_stats = defaultdict(lambda: defaultdict(int))
        
    def extract_players_from_description(self, description):
        """Extract player names from play description"""
        if pd.isna(description):
            return []
        
        # Pattern: number-initial.lastname (e.g., "8-J.JACOBS", "14-S.DARNOLD")
        players = re.findall(r'(\d+)-([A-Z]\.[A-Z]+)', str(description))
        return [(num, name) for num, name in players]
    
    def is_red_zone_play(self, yard_line, yard_direction):
        """Determine if play is in red zone (within 20 yards of goal line)"""
        if pd.isna(yard_line) or pd.isna(yard_direction):
            return False
            
        try:
            yard_line = int(yard_line)
            # If attacking opponent's goal line and within 20 yards
            if yard_direction == 'OPP' and yard_line <= 20:
                return True
        except:
            pass
        return False
    
    def extract_target_data(self):
        """Extract passing targets and receptions"""
        print("\n🎯 Extracting target and reception data...")
        
        passing_plays = self.df[self.df['IsPass'] == 1].copy()
        target_count = 0
        reception_count = 0
        
        for _, play in passing_plays.iterrows():
            players = self.extract_players_from_description(play['Description'])
            team = play['OffenseTeam']
            is_red_zone = self.is_red_zone_play(play['YardLineFixed'], play['YardLineDirection'])
            
            if len(players) >= 2:  # QB + receiver
                qb_num, qb_name = players[0]
                receiver_num, receiver_name = players[1]
                
                # Count as target
                self.player_stats[receiver_name]['targets'] += 1
                self.player_stats[receiver_name]['team'] = team
                self.team_stats[team]['total_targets'] += 1
                target_count += 1
                
                if is_red_zone:
                    self.player_stats[receiver_name]['red_zone_targets'] += 1
                
                # Count as reception if not incomplete/interception
                if play['IsIncomplete'] == 0 and play['IsInterception'] == 0:
                    self.player_stats[receiver_name]['receptions'] += 1
                    self.player_stats[receiver_name]['receiving_yards'] += play['Yards']
                    reception_count += 1
                    
                    if is_red_zone:
                        self.player_stats[receiver_name]['red_zone_receptions'] += 1
                
                # Touchdown reception
                if play['IsTouchdown'] == 1 and play['IsIncomplete'] == 0:
                    self.player_stats[receiver_name]['receiving_tds'] += 1
        
        print(f"   📊 Found {target_count:,} targets and {reception_count:,} receptions")
    
    def extract_rushing_data(self):
        """Extract rushing attempts and yards"""
        print("\n🏃 Extracting rushing data...")
        
        rushing_plays = self.df[self.df['IsRush'] == 1].copy()
        carry_count = 0
        
        for _, play in rushing_plays.iterrows():
            players = self.extract_players_from_description(play['Description'])
            team = play['OffenseTeam']
            is_red_zone = self.is_red_zone_play(play['YardLineFixed'], play['YardLineDirection'])
            
            if players:  # At least one player (rusher)
                rusher_num, rusher_name = players[0]
                
                self.player_stats[rusher_name]['carries'] += 1
                self.player_stats[rusher_name]['rushing_yards'] += play['Yards']
                self.player_stats[rusher_name]['team'] = team
                carry_count += 1
                
                if is_red_zone:
                    self.player_stats[rusher_name]['red_zone_carries'] += 1
                
                # Rushing touchdown
                if play['IsTouchdown'] == 1:
                    self.player_stats[rusher_name]['rushing_tds'] += 1
        
        print(f"   📊 Found {carry_count:,} rushing attempts")
    
    def calculate_snap_estimates(self):
        """Estimate snap counts from play participation"""
        print("\n⚡ Estimating snap counts...")
        
        # Count offensive plays per team
        team_offensive_plays = self.df.groupby('OffenseTeam').size().to_dict()
        
        for team, plays in team_offensive_plays.items():
            self.team_stats[team]['offensive_plays'] = plays
        
        # Estimate snaps for players who had touches/targets
        snap_estimates = 0
        for player_name, stats in self.player_stats.items():
            team = stats['team']
            if team in team_offensive_plays:
                # Rough estimate: players with activity probably played 60-90% of snaps
                total_touches = stats['carries'] + stats['targets']
                team_plays = team_offensive_plays[team]
                
                if total_touches > 0:
                    # Estimate based on usage - more touches = more snaps
                    if total_touches >= 15:  # High usage
                        snap_estimate = int(team_plays * 0.85)
                    elif total_touches >= 8:  # Medium usage  
                        snap_estimate = int(team_plays * 0.65)
                    elif total_touches >= 3:  # Low usage
                        snap_estimate = int(team_plays * 0.40)
                    else:  # Very low usage
                        snap_estimate = int(team_plays * 0.20)
                    
                    stats['estimated_snaps'] = snap_estimate
                    snap_estimates += 1
        
        print(f"   📊 Generated snap estimates for {snap_estimates} players")
    
    def calculate_advanced_metrics(self):
        """Calculate advanced fantasy metrics"""
        print("\n📈 Calculating advanced fantasy metrics...")
        
        metrics_calculated = 0
        
        for player_name, stats in self.player_stats.items():
            team = stats['team']
            
            # Target share
            if stats['targets'] > 0 and team in self.team_stats:
                team_targets = self.team_stats[team]['total_targets']
                if team_targets > 0:
                    stats['target_share_pct'] = round((stats['targets'] / team_targets) * 100, 1)
            
            # Snap count percentage (estimated)
            if stats.get('estimated_snaps', 0) > 0 and team in self.team_stats:
                team_plays = self.team_stats[team]['offensive_plays']
                if team_plays > 0:
                    stats['snap_count_pct'] = round((stats['estimated_snaps'] / team_plays) * 100, 1)
            
            # Red zone usage
            total_rz_opportunities = stats.get('red_zone_targets', 0) + stats.get('red_zone_carries', 0)
            if total_rz_opportunities > 0:
                stats['red_zone_opportunities'] = total_rz_opportunities
            
            # Catch rate
            if stats['targets'] > 0:
                stats['catch_rate_pct'] = round((stats['receptions'] / stats['targets']) * 100, 1)
            
            # Yards per touch
            total_touches = stats['carries'] + stats['receptions']
            total_yards = stats['rushing_yards'] + stats['receiving_yards'] 
            if total_touches > 0:
                stats['yards_per_touch'] = round(total_yards / total_touches, 1)
            
            # Total fantasy points (standard scoring)
            receiving_pts = stats['receiving_yards'] * 0.1 + stats['receiving_tds'] * 6
            rushing_pts = stats['rushing_yards'] * 0.1 + stats['rushing_tds'] * 6
            ppr_pts = stats['receptions'] * 1  # PPR bonus
            
            stats['fantasy_points_standard'] = round(receiving_pts + rushing_pts, 1)
            stats['fantasy_points_ppr'] = round(receiving_pts + rushing_pts + ppr_pts, 1)
            
            metrics_calculated += 1
        
        print(f"   📊 Calculated advanced metrics for {metrics_calculated} players")
    
    def create_player_summary_df(self):
        """Create a clean DataFrame with player statistics"""
        print("\n📋 Creating player summary DataFrame...")
        
        player_data = []
        
        for player_name, stats in self.player_stats.items():
            # Only include players with meaningful activity
            if stats.get('targets', 0) > 0 or stats.get('carries', 0) > 0:
                player_data.append({
                    'player_name': player_name,
                    'team': stats.get('team', ''),
                    'targets': stats.get('targets', 0),
                    'receptions': stats.get('receptions', 0),
                    'receiving_yards': stats.get('receiving_yards', 0),
                    'receiving_tds': stats.get('receiving_tds', 0),
                    'carries': stats.get('carries', 0),
                    'rushing_yards': stats.get('rushing_yards', 0),
                    'rushing_tds': stats.get('rushing_tds', 0),
                    'estimated_snaps': stats.get('estimated_snaps', 0),
                    'target_share_pct': stats.get('target_share_pct', 0),
                    'snap_count_pct': stats.get('snap_count_pct', 0),
                    'catch_rate_pct': stats.get('catch_rate_pct', 0),
                    'red_zone_opportunities': stats.get('red_zone_opportunities', 0),
                    'yards_per_touch': stats.get('yards_per_touch', 0),
                    'fantasy_points_standard': stats.get('fantasy_points_standard', 0),
                    'fantasy_points_ppr': stats.get('fantasy_points_ppr', 0)
                })
        
        df = pd.DataFrame(player_data)
        df = df.sort_values('fantasy_points_ppr', ascending=False)
        
        print(f"   📊 Created summary for {len(df)} players with activity")
        return df
    
    def run_full_extraction(self):
        """Run the complete extraction process"""
        print("🚀 Starting full fantasy metrics extraction...")
        print("="*60)
        
        self.extract_target_data()
        self.extract_rushing_data()
        self.calculate_snap_estimates()
        self.calculate_advanced_metrics()
        
        player_df = self.create_player_summary_df()
        
        print("\n✅ EXTRACTION COMPLETE!")
        print("="*60)
        print(f"📊 Total players analyzed: {len(player_df)}")
        print(f"🎯 Players with targets: {len(player_df[player_df['targets'] > 0])}")
        print(f"🏃 Players with carries: {len(player_df[player_df['carries'] > 0])}")
        
        return player_df

def main():
    """Main function to demonstrate usage"""
    
    # Initialize extractor
    extractor = FantasyMetricsExtractor('data/pbp-2024.csv')
    
    # Run extraction
    player_df = extractor.run_full_extraction()
    
    # Show top performers
    print("\n🏆 TOP 10 PPR PERFORMERS:")
    print("="*50)
    top_players = player_df.head(10)[['player_name', 'team', 'targets', 'receptions', 
                                     'carries', 'target_share_pct', 'fantasy_points_ppr']]
    print(top_players.to_string(index=False))
    
    # Show top target shares
    print("\n🎯 TOP 10 TARGET SHARES:")
    print("="*50)
    target_leaders = player_df[player_df['targets'] >= 20].nlargest(10, 'target_share_pct')
    target_display = target_leaders[['player_name', 'team', 'targets', 'target_share_pct', 'catch_rate_pct']]
    print(target_display.to_string(index=False))
    
    # Save to CSV
    output_file = 'data/fantasy_metrics_2024.csv'
    player_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved detailed metrics to: {output_file}")
    
    return player_df

if __name__ == "__main__":
    player_df = main() 