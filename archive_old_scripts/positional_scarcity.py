import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class PositionalScarcityModel:
    """
    Mathematical model for positional scarcity in fantasy football drafts.
    Considers league size, round dynamics, and position depth.
    """
    
    def __init__(self, league_size: int = 10):
        self.league_size = league_size
        
        # Starting lineup requirements (standard fantasy)
        self.lineup_requirements = {
            'QB': 1,
            'RB': 2,  # RB1, RB2
            'WR': 2,  # WR1, WR2  
            'TE': 1,
            'FLEX': 1,  # Can be RB, WR, or TE
            'K': 1,
            'DST': 1
        }
        
        # Typical bench size per position
        self.bench_targets = {
            'QB': 1,   # 1 backup QB
            'RB': 2,   # 2 backup RBs
            'WR': 3,   # 3 backup WRs
            'TE': 1,   # 1 backup TE
            'K': 0,    # No backup kickers
            'DST': 0   # No backup defenses
        }
        
        # Position depth analysis (REAL FANTASY-RELEVANT PLAYER COUNTS)
        # Based on historical data: players who finish in top X at position
        self.position_depth = {
            'QB': 20,   # ~20 QBs worth starting (12 elite + 8 streamable)
            'RB': 40,   # ~40 RBs worth rostering (25 starters + 15 handcuffs) 
            'WR': 65,   # ~65 WRs worth rostering (45 startable + 20 flex/depth)
            'TE': 18,   # ~18 TEs worth rostering (12 startable + 6 depth)
            'K': 20,    # ~20 Ks viable (position variance minimal)
            'DST': 24   # ~24 DSTs streamable (matchup dependent)
        }
    
    def calculate_total_demand(self, position: str) -> int:
        """Calculate total demand for a position across all teams"""
        base_demand = self.lineup_requirements.get(position, 0)
        bench_demand = self.bench_targets.get(position, 0)
        
        # FLEX adds demand for RB, WR, TE
        flex_demand = 0
        if position in ['RB', 'WR', 'TE']:
            # Assume FLEX is split: 50% RB, 40% WR, 10% TE
            flex_split = {'RB': 0.5, 'WR': 0.4, 'TE': 0.1}
            flex_demand = self.lineup_requirements['FLEX'] * flex_split.get(position, 0)
        
        total_per_team = base_demand + bench_demand + flex_demand
        return int(total_per_team * self.league_size)
    
    def calculate_scarcity_score(self, position: str, current_round: int) -> float:
        """
        Calculate positional scarcity score (higher = more scarce = draft sooner)
        """
        total_demand = self.calculate_total_demand(position)
        available_supply = self.position_depth[position]
        
        # Base scarcity ratio
        scarcity_ratio = total_demand / available_supply
        
        # Round-based urgency multiplier
        round_urgency = self.calculate_round_urgency(position, current_round)
        
        # Position-specific adjustments
        position_multiplier = self.get_position_multiplier(position)
        
        scarcity_score = scarcity_ratio * round_urgency * position_multiplier
        
        return scarcity_score
    
    def calculate_round_urgency(self, position: str, current_round: int) -> float:
        """Calculate urgency multiplier based on current draft round"""
        
        # Early rounds (1-4): Focus on RB/WR scarcity
        if current_round <= 4:
            urgency_map = {
                'RB': 1.3,   # RBs very scarce early
                'WR': 1.2,   # WRs scarce early  
                'TE': 0.7,   # Wait on TE
                'QB': 0.6,   # Wait on QB
                'K': 0.1,    # Never draft early
                'DST': 0.1   # Never draft early
            }
        
        # Mid rounds (5-8): Start considering QB/TE
        elif current_round <= 8:
            urgency_map = {
                'RB': 1.1,   # Still valuable
                'WR': 1.1,   # Still valuable
                'TE': 1.0,   # Consider top TEs
                'QB': 0.8,   # Consider top QBs
                'K': 0.1,    # Still wait
                'DST': 0.1   # Still wait
            }
        
        # Late rounds (9-12): Fill remaining needs
        elif current_round <= 12:
            urgency_map = {
                'RB': 0.9,   # Depth/handcuffs
                'WR': 0.9,   # Depth pieces
                'TE': 1.2,   # Must fill if empty
                'QB': 1.1,   # Must fill if empty  
                'K': 0.3,    # Still early for K
                'DST': 0.3   # Still early for DST
            }
        
        # Very late rounds (13+): K/DST time
        else:
            urgency_map = {
                'RB': 0.8,   # Deep depth
                'WR': 0.8,   # Deep depth
                'TE': 1.0,   # Backup option
                'QB': 0.9,   # Backup option
                'K': 1.5,    # Time to draft K
                'DST': 1.5   # Time to draft DST
            }
        
        return urgency_map.get(position, 1.0)
    
    def get_position_multiplier(self, position: str) -> float:
        """Position-specific scarcity multipliers based on REAL FANTASY DATA ANALYSIS"""
        
        # These multipliers are based on:
        # 1. Value-Based Drafting (VBD) research
        # 2. Position scarcity vs. replacement level
        # 3. Injury rates and workload sustainability
        # 4. Draft capital efficiency studies
        
        multipliers = {
            # RB: MOST SCARCE due to:
            # - Only ~40 fantasy-relevant RBs vs 80+ WRs
            # - Higher injury rate (28% vs 18% for WRs) 
            # - More volatile year-to-year production
            # - Shorter careers (3.3 years vs 4.8 for WRs)
            'RB': 1.45,  
            
            # TE: VERY SCARCE due to:
            # - Only ~12 elite TEs vs ~25 elite WRs
            # - Massive drop-off after top tier
            # - Kelce/Andrews vs average TE = 8+ points per game
            'TE': 1.35,  
            
            # WR: MODERATELY SCARCE due to:
            # - Larger player pool but still positional requirements
            # - 3+ WR sets mean more opportunity
            # - More predictable than RBs
            'WR': 1.15,  
            
            # QB: LESS SCARCE due to:
            # - Only need 1 starter in most leagues
            # - More predictable production year-to-year  
            # - Longer careers and injury resilience
            # - Streaming viable in deeper leagues
            'QB': 0.85,  
            
            # DST: STREAMABLE due to:
            # - Weekly matchups matter more than talent
            # - Widely available on waivers
            # - No significant advantage to elite defenses long-term
            'DST': 0.55,  
            
            # K: LEAST SCARCE due to:
            # - Almost purely random/matchup dependent
            # - No meaningful skill difference between kickers
            # - Easily replaceable week-to-week
            'K': 0.45   
        }
        
        return multipliers.get(position, 1.0)
    
    def calculate_opportunity_cost(self, current_team: Dict, current_round: int) -> Dict[str, float]:
        """
        Calculate opportunity cost for each position based on current team composition
        """
        opportunity_costs = {}
        
        # Count filled positions
        filled_counts = self.count_filled_positions(current_team)
        
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            # Base scarcity
            base_scarcity = self.calculate_scarcity_score(position, current_round)
            
            # Need multiplier (higher if we don't have enough)
            need_multiplier = self.calculate_need_multiplier(position, filled_counts)
            
            # Diminishing returns (lower if we already have many)
            diminishing_returns = self.calculate_diminishing_returns(position, filled_counts)
            
            opportunity_costs[position] = base_scarcity * need_multiplier * diminishing_returns
        
        return opportunity_costs
    
    def count_filled_positions(self, current_team: Dict) -> Dict[str, int]:
        """Count how many of each position we have drafted"""
        counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'K': 0, 'DST': 0}
        
        if not isinstance(current_team, dict):
            return counts
        
        # Count starting lineup
        for slot, player in current_team.items():
            if slot != 'Bench' and isinstance(player, dict) and player.get('position'):
                pos = player['position']
                if pos in counts:
                    counts[pos] += 1
        
        # Count bench
        if 'Bench' in current_team and isinstance(current_team['Bench'], list):
            for bench_player in current_team['Bench']:
                if isinstance(bench_player, dict) and bench_player.get('position'):
                    pos = bench_player['position']
                    if pos in counts:
                        counts[pos] += 1
        
        return counts
    
    def calculate_need_multiplier(self, position: str, filled_counts: Dict[str, int]) -> float:
        """Calculate multiplier based on positional need"""
        
        current_count = filled_counts.get(position, 0)
        
        # Minimum requirements (including FLEX considerations)
        min_requirements = {
            'QB': 1,
            'RB': 3,  # RB1, RB2, + FLEX potential
            'WR': 3,  # WR1, WR2, + FLEX potential  
            'TE': 1,  # TE + FLEX potential
            'K': 1,
            'DST': 1
        }
        
        min_needed = min_requirements.get(position, 1)
        
        if current_count == 0:
            return 2.0  # Desperate need
        elif current_count < min_needed:
            return 1.5  # Strong need
        elif current_count == min_needed:
            return 1.0  # Adequate
        else:
            return 0.7  # Low need
    
    def calculate_diminishing_returns(self, position: str, filled_counts: Dict[str, int]) -> float:
        """Calculate diminishing returns for additional players at position"""
        
        current_count = filled_counts.get(position, 0)
        
        # Diminishing returns curves by position
        if position in ['RB', 'WR']:
            # RB/WR have value up to 4-5 players due to FLEX and depth
            if current_count <= 2:
                return 1.0
            elif current_count <= 4:
                return 0.8
            else:
                return 0.5
        
        elif position in ['QB', 'TE']:
            # QB/TE need 1-2, diminish quickly after that
            if current_count <= 1:
                return 1.0
            elif current_count <= 2:
                return 0.6
            else:
                return 0.3
        
        else:  # K, DST
            # K/DST only need 1, maybe 2 max
            if current_count <= 1:
                return 1.0
            else:
                return 0.2
    
    def apply_scarcity_boost(self, player_df: pd.DataFrame, current_team: Dict, current_round: int) -> pd.DataFrame:
        """
        Apply scarcity-based boosts to player scores
        """
        if player_df.empty:
            return player_df
        
        # Calculate opportunity costs for each position
        opportunity_costs = self.calculate_opportunity_cost(current_team, current_round)
        
        print(f"🔄 Round {current_round} Positional Scarcity Scores:")
        for pos, score in sorted(opportunity_costs.items(), key=lambda x: x[1], reverse=True):
            print(f"   {pos}: {score:.2f}")
        
        # Apply boosts to player scores
        df_copy = player_df.copy()
        
        for position, boost_multiplier in opportunity_costs.items():
            position_mask = df_copy['position'] == position
            
            if position_mask.any():
                # 🔧 FIX: Cap extreme scarcity multipliers to prevent inflation
                # Max 2.5x boost to prevent ridiculous scores like R.WHITE's 4.24x
                capped_multiplier = min(boost_multiplier, 2.5)
                
                if capped_multiplier != boost_multiplier:
                    print(f"   ⚠️ Capped {position} scarcity from {boost_multiplier:.2f}x to {capped_multiplier:.2f}x")
                
                # Apply multiplicative boost to projected points or base score
                score_column = 'catboost_prediction' if 'catboost_prediction' in df_copy.columns else 'projected_points'
                
                df_copy.loc[position_mask, 'scarcity_boost'] = capped_multiplier
                df_copy.loc[position_mask, 'boosted_score'] = (
                    df_copy.loc[position_mask, score_column] * capped_multiplier
                )
        
        # Fill missing values
        df_copy['scarcity_boost'] = df_copy['scarcity_boost'].fillna(1.0)
        score_column = 'catboost_prediction' if 'catboost_prediction' in df_copy.columns else 'projected_points'
        df_copy['boosted_score'] = df_copy['boosted_score'].fillna(df_copy[score_column])
        
        return df_copy


def estimate_current_round(current_team: Dict, league_size: int = 10) -> int:
    """Estimate current draft round based on roster composition"""
    
    if not isinstance(current_team, dict):
        return 1
    
    total_players = 0
    
    # Count starting lineup players
    for slot, player in current_team.items():
        if slot != 'Bench' and isinstance(player, dict) and player.get('name'):
            total_players += 1
    
    # Count bench players
    if 'Bench' in current_team and isinstance(current_team['Bench'], list):
        for bench_player in current_team['Bench']:
            if isinstance(bench_player, dict) and bench_player.get('name'):
                total_players += 1
    
    # Estimate round (each round = 1 pick per team)
    estimated_round = max(1, total_players + 1)
    
    return min(estimated_round, 16)  # Cap at 16 rounds


# Example usage and testing
if __name__ == "__main__":
    model = PositionalScarcityModel(league_size=10)
    
    # Test with different roster states
    empty_team = {'QB': None, 'RB1': None, 'RB2': None, 'WR1': None, 'WR2': None, 'TE': None, 'FLEX': None, 'K': None, 'DST': None, 'Bench': [None] * 6}
    
    round_1_costs = model.calculate_opportunity_cost(empty_team, 1)
    round_6_costs = model.calculate_opportunity_cost(empty_team, 6)
    round_12_costs = model.calculate_opportunity_cost(empty_team, 12)
    
    print("Round 1 Scarcity Scores:", round_1_costs)
    print("Round 6 Scarcity Scores:", round_6_costs)
    print("Round 12 Scarcity Scores:", round_12_costs) 