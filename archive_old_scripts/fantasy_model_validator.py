#!/usr/bin/env python3
"""
Fantasy Football Model Validator
Tests real-world effectiveness of our model's recommendations
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FantasyModelValidator:
    def __init__(self):
        self.validation_results = {}
        
    def load_validation_data(self):
        """Load historical data for validation"""
        print("🏈 FANTASY FOOTBALL MODEL VALIDATION")
        print("=" * 50)
        
        # Load our model's historical predictions if available
        try:
            # Load 2024 actual results
            actual_2024 = pd.read_csv('data/fantasy_metrics_2024.csv')
            print(f"✅ Loaded 2024 actual results: {len(actual_2024)} players")
            
            # Load our comprehensive dataset (what our model trains on)
            comprehensive_files = [
                'recency_weighted_dataset_20250803_154654.csv',
                'comprehensive_training_data_FIXED_20250803_135952.csv', 
                'clean_unified_dataset_20250803_133711.csv'
            ]
            
            model_data = None
            for file in comprehensive_files:
                try:
                    model_data = pd.read_csv(file)
                    print(f"✅ Loaded model training data: {len(model_data)} players from {file}")
                    break
                except FileNotFoundError:
                    continue
                    
            return actual_2024, model_data
            
        except Exception as e:
            print(f"❌ Error loading validation data: {e}")
            return None, None
    
    def test_draft_strategy_effectiveness(self, actual_2024, model_data):
        """Test if our top-ranked players actually performed well"""
        print("\n📊 TESTING DRAFT STRATEGY EFFECTIVENESS")
        print("-" * 40)
        
        if model_data is None:
            print("❌ No model data available for validation")
            return {}
            
        strategy_results = {}
        
        # Simulate our model's top picks by position
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for pos in positions:
            print(f"\n🎯 {pos} Position Analysis:")
            
            # Get our model's top 10 players at this position
            pos_players = model_data[model_data['position'] == pos].copy()
            if 'recency_weighted_fantasy_points' in pos_players.columns:
                top_model_picks = pos_players.nlargest(10, 'recency_weighted_fantasy_points')
            elif 'recency_weighted_score' in pos_players.columns:
                top_model_picks = pos_players.nlargest(10, 'recency_weighted_score')
            elif 'avg_fantasy_points_3yr' in pos_players.columns:
                top_model_picks = pos_players.nlargest(10, 'avg_fantasy_points_3yr')
            else:
                print(f"   ⚠️ No scoring column found for {pos}")
                print(f"   Available columns: {list(pos_players.columns)}")
                continue
            
            # Compare to actual 2024 performance
            actual_pos = actual_2024[actual_2024['position'] == pos].copy()
            
            model_performance = []
            for _, player in top_model_picks.iterrows():
                player_name = player['name']
                
                # Find actual performance (handle name variations)
                actual_match = actual_pos[actual_pos['name'].str.contains(
                    player_name.split()[0], case=False, na=False)]
                
                if not actual_match.empty:
                    actual_points = actual_match.iloc[0]['fantasy_points_ppr']
                    model_performance.append({
                        'name': player_name,
                        'predicted_score': player.get('recency_weighted_fantasy_points', player.get('recency_weighted_score', player.get('avg_fantasy_points_3yr', 0))),
                        'actual_points': actual_points
                    })
            
            if model_performance:
                avg_actual = np.mean([p['actual_points'] for p in model_performance])
                pos_avg_all = actual_pos['fantasy_points_ppr'].mean()
                
                outperformance = avg_actual - pos_avg_all
                print(f"   Our Top 10: {avg_actual:.1f} avg points")
                print(f"   Position Avg: {pos_avg_all:.1f} avg points")
                print(f"   Outperformance: {outperformance:+.1f} points ({outperformance/pos_avg_all*100:+.1f}%)")
                
                strategy_results[pos] = {
                    'our_top_10_avg': avg_actual,
                    'position_average': pos_avg_all,
                    'outperformance': outperformance,
                    'outperformance_pct': outperformance/pos_avg_all*100,
                    'sample_size': len(model_performance)
                }
        
        return strategy_results
    
    def test_vs_expert_consensus(self, actual_2024, model_data):
        """Compare our rankings to expert consensus (ADP)"""
        print("\n📊 TESTING VS EXPERT CONSENSUS (ADP)")
        print("-" * 40)
        
        try:
            # Load ADP data with error handling
            adp_data = pd.read_csv('data/adp.csv', on_bad_lines='skip')
            # Clean column names
            adp_data.columns = adp_data.columns.str.strip().str.replace('"', '')
            # Use Player column for name
            if 'Player' in adp_data.columns:
                adp_data['name'] = adp_data['Player']
            print(f"✅ Loaded ADP data: {len(adp_data)} players")
            
            consensus_results = {}
            
            # Compare top 50 players by ADP vs our model vs actual performance
            adp_top_50 = adp_data.head(50)
            
            if model_data is not None and 'recency_weighted_fantasy_points' in model_data.columns:
                model_top_50 = model_data.nlargest(50, 'recency_weighted_fantasy_points')
            elif model_data is not None and 'recency_weighted_score' in model_data.columns:
                model_top_50 = model_data.nlargest(50, 'recency_weighted_score')
            else:
                print("⚠️ No model rankings available")
                return {}
            
            # Calculate actual performance for both sets
            adp_actual_points = []
            model_actual_points = []
            
            for _, player in adp_top_50.iterrows():
                match = actual_2024[actual_2024['name'].str.contains(
                    player['name'].split()[0], case=False, na=False)]
                if not match.empty:
                    adp_actual_points.append(match.iloc[0]['fantasy_points_ppr'])
            
            for _, player in model_top_50.iterrows():
                match = actual_2024[actual_2024['name'].str.contains(
                    player['name'].split()[0], case=False, na=False)]
                if not match.empty:
                    model_actual_points.append(match.iloc[0]['fantasy_points_ppr'])
            
            if adp_actual_points and model_actual_points:
                adp_avg = np.mean(adp_actual_points)
                model_avg = np.mean(model_actual_points)
                
                print(f"ADP Top 50 Average: {adp_avg:.1f} fantasy points")
                print(f"Our Top 50 Average: {model_avg:.1f} fantasy points")
                print(f"Difference: {model_avg - adp_avg:+.1f} points ({(model_avg - adp_avg)/adp_avg*100:+.1f}%)")
                
                consensus_results = {
                    'adp_top_50_avg': adp_avg,
                    'model_top_50_avg': model_avg,
                    'difference': model_avg - adp_avg,
                    'difference_pct': (model_avg - adp_avg)/adp_avg*100,
                    'adp_sample_size': len(adp_actual_points),
                    'model_sample_size': len(model_actual_points)
                }
            
            return consensus_results
            
        except Exception as e:
            print(f"❌ Error comparing to consensus: {e}")
            return {}
    
    def test_weekly_consistency(self, actual_2024):
        """Test if our high-scoring players are actually consistent"""
        print("\n📊 TESTING WEEKLY CONSISTENCY")
        print("-" * 40)
        
        # This would require weekly data, which we might not have
        # For now, we can use games played and fantasy points to estimate consistency
        
        consistency_results = {}
        
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_players = actual_2024[actual_2024['position'] == pos].copy()
            
            if 'games' in pos_players.columns and len(pos_players) > 10:
                # Calculate consistency score (points per game vs total points)
                pos_players['ppg'] = pos_players['fantasy_points_ppr'] / pos_players['games'].clip(lower=1)
                pos_players['consistency'] = 1 / (1 + pos_players['ppg'].std() / pos_players['ppg'].mean())
                
                top_performers = pos_players.nlargest(10, 'fantasy_points_ppr')
                avg_consistency = top_performers['consistency'].mean()
                
                print(f"{pos}: Top 10 avg consistency score: {avg_consistency:.3f}")
                
                consistency_results[pos] = {
                    'top_10_consistency': avg_consistency,
                    'position_avg_consistency': pos_players['consistency'].mean()
                }
        
        return consistency_results
    
    def test_bust_rate(self, actual_2024, model_data):
        """Test how often our top picks 'bust' (perform below expectations)"""
        print("\n📊 TESTING BUST RATE")
        print("-" * 40)
        
        if model_data is None:
            print("❌ No model data for bust analysis")
            return {}
            
        bust_results = {}
        
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_model = model_data[model_data['position'] == pos].copy()
            pos_actual = actual_2024[actual_2024['position'] == pos].copy()
            
            if len(pos_model) > 10 and len(pos_actual) > 10:
                # Get our top 10 predictions
                if 'recency_weighted_fantasy_points' in pos_model.columns:
                    top_10_model = pos_model.nlargest(10, 'recency_weighted_fantasy_points')
                elif 'recency_weighted_score' in pos_model.columns:
                    top_10_model = pos_model.nlargest(10, 'recency_weighted_score')
                else:
                    continue
                
                # Check how many performed below position average
                pos_avg_actual = pos_actual['fantasy_points_ppr'].mean()
                busts = 0
                total_matched = 0
                
                for _, player in top_10_model.iterrows():
                    match = pos_actual[pos_actual['name'].str.contains(
                        player['name'].split()[0], case=False, na=False)]
                    
                    if not match.empty:
                        total_matched += 1
                        actual_points = match.iloc[0]['fantasy_points_ppr']
                        if actual_points < pos_avg_actual:
                            busts += 1
                
                if total_matched > 0:
                    bust_rate = busts / total_matched * 100
                    print(f"{pos}: {busts}/{total_matched} top picks busted ({bust_rate:.1f}%)")
                    
                    bust_results[pos] = {
                        'busts': busts,
                        'total_matched': total_matched,
                        'bust_rate_pct': bust_rate
                    }
        
        return bust_results
    
    def generate_model_report_card(self):
        """Generate overall model effectiveness report"""
        print("\n🎯 MODEL EFFECTIVENESS REPORT CARD")
        print("=" * 50)
        
        # Load data
        actual_2024, model_data = self.load_validation_data()
        
        if actual_2024 is None:
            print("❌ Cannot generate report without validation data")
            return
        
        # Run all tests
        strategy_results = self.test_draft_strategy_effectiveness(actual_2024, model_data)
        consensus_results = self.test_vs_expert_consensus(actual_2024, model_data)
        consistency_results = self.test_weekly_consistency(actual_2024)
        bust_results = self.test_bust_rate(actual_2024, model_data)
        
        # Calculate overall grades
        print("\n📋 FINAL REPORT CARD:")
        print("-" * 30)
        
        grades = []
        
        # Grade strategy effectiveness
        if strategy_results:
            avg_outperformance = np.mean([r['outperformance_pct'] for r in strategy_results.values()])
            if avg_outperformance > 10:
                strategy_grade = 'A'
            elif avg_outperformance > 5:
                strategy_grade = 'B'
            elif avg_outperformance > 0:
                strategy_grade = 'C'
            else:
                strategy_grade = 'D'
            
            print(f"Draft Strategy: {strategy_grade} ({avg_outperformance:+.1f}% vs position average)")
            grades.append(strategy_grade)
        
        # Grade vs consensus
        if consensus_results and 'difference_pct' in consensus_results:
            consensus_diff = consensus_results['difference_pct']
            if consensus_diff > 5:
                consensus_grade = 'A'
            elif consensus_diff > 2:
                consensus_grade = 'B'
            elif consensus_diff > -2:
                consensus_grade = 'C'
            else:
                consensus_grade = 'D'
            
            print(f"vs Expert Consensus: {consensus_grade} ({consensus_diff:+.1f}% vs ADP)")
            grades.append(consensus_grade)
        
        # Grade bust rate
        if bust_results:
            avg_bust_rate = np.mean([r['bust_rate_pct'] for r in bust_results.values()])
            if avg_bust_rate < 20:
                bust_grade = 'A'
            elif avg_bust_rate < 30:
                bust_grade = 'B'
            elif avg_bust_rate < 40:
                bust_grade = 'C'
            else:
                bust_grade = 'D'
            
            print(f"Bust Avoidance: {bust_grade} ({avg_bust_rate:.1f}% bust rate)")
            grades.append(bust_grade)
        
        # Overall grade
        if grades:
            grade_values = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
            avg_grade_value = np.mean([grade_values[g] for g in grades])
            
            if avg_grade_value >= 3.5:
                overall = 'A'
            elif avg_grade_value >= 2.5:
                overall = 'B'
            elif avg_grade_value >= 1.5:
                overall = 'C'
            else:
                overall = 'D'
            
            print(f"\n🏆 OVERALL MODEL GRADE: {overall}")
            
            if overall in ['A', 'B']:
                print("✅ Model is performing well for fantasy football!")
            elif overall == 'C':
                print("⚠️ Model is mediocre - consider improvements")
            else:
                print("❌ Model needs significant work")
        
        # Save results
        self.validation_results = {
            'strategy_results': strategy_results,
            'consensus_results': consensus_results,
            'consistency_results': consistency_results,
            'bust_results': bust_results,
            'overall_grade': overall if grades else 'N/A'
        }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_validation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
        return self.validation_results

if __name__ == "__main__":
    validator = FantasyModelValidator()
    validator.generate_model_report_card() 