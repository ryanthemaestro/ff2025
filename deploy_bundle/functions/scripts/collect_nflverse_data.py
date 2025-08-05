#!/usr/bin/env python3
"""
Clean NFLverse Data Collector
Downloads real NFL statistics from 2022, 2023, 2024 for AI model training
"""

import pandas as pd
import nfl_data_py as nfl
import os
from datetime import datetime

class NFLverseDataCollector:
    """Collect clean NFL data for fantasy football AI model training"""
    
    def __init__(self):
        self.years = [2022, 2023, 2024]
        self.data_dir = 'data/nflverse'
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
    def collect_seasonal_stats(self):
        """Collect seasonal player statistics for each year"""
        print("📊 Collecting NFLverse seasonal statistics...")
        
        for year in self.years:
            print(f"\n🏈 Fetching {year} season data...")
            
            try:
                # Get regular season stats
                print(f"   • Regular season stats...")
                reg_stats = nfl.import_seasonal_data([year], s_type='REG')
                
                # Get playoff stats if available
                print(f"   • Playoff stats...")
                try:
                    playoff_stats = nfl.import_seasonal_data([year], s_type='POST')
                    # Combine regular season and playoff stats
                    all_stats = pd.concat([reg_stats, playoff_stats], ignore_index=True)
                except:
                    print(f"     (No playoff data for {year})")
                    all_stats = reg_stats
                
                # Save to file
                filename = f"{self.data_dir}/seasonal_stats_{year}.csv"
                all_stats.to_csv(filename, index=False)
                print(f"   ✅ Saved {len(all_stats)} records to {filename}")
                
                # Show data summary
                positions = all_stats['position'].value_counts()
                print(f"   📈 Positions: {positions.to_dict()}")
                
            except Exception as e:
                print(f"   ❌ Error fetching {year} data: {e}")
                
    def collect_weekly_stats(self):
        """Collect weekly player statistics for consistency tracking"""
        print("\n📅 Collecting NFLverse weekly statistics...")
        
        for year in self.years:
            print(f"\n🏈 Fetching {year} weekly data...")
            
            try:
                # Get weekly stats
                weekly_stats = nfl.import_weekly_data([year], downcast=True)
                
                # Save to file
                filename = f"{self.data_dir}/weekly_stats_{year}.csv"
                weekly_stats.to_csv(filename, index=False)
                print(f"   ✅ Saved {len(weekly_stats)} weekly records to {filename}")
                
            except Exception as e:
                print(f"   ❌ Error fetching {year} weekly data: {e}")
                
    def collect_player_info(self):
        """Collect player roster information"""
        print("\n👥 Collecting NFLverse player rosters...")
        
        for year in self.years:
            print(f"\n🏈 Fetching {year} rosters...")
            
            try:
                # Get roster data
                rosters = nfl.import_rosters([year])
                
                # Save to file
                filename = f"{self.data_dir}/rosters_{year}.csv"
                rosters.to_csv(filename, index=False)
                print(f"   ✅ Saved {len(rosters)} roster records to {filename}")
                
                # Show team summary
                teams = rosters['team'].value_counts()
                print(f"   📈 Teams: {len(teams)} teams with roster data")
                
            except Exception as e:
                print(f"   ❌ Error fetching {year} roster data: {e}")
                
    def create_training_dataset(self):
        """Combine all data into a clean training dataset"""
        print("\n🔧 Creating combined training dataset...")
        
        all_seasonal = []
        all_weekly = []
        all_rosters = []
        
        # Load all seasonal data
        for year in self.years:
            seasonal_file = f"{self.data_dir}/seasonal_stats_{year}.csv"
            weekly_file = f"{self.data_dir}/weekly_stats_{year}.csv"
            roster_file = f"{self.data_dir}/rosters_{year}.csv"
            
            if os.path.exists(seasonal_file):
                seasonal_data = pd.read_csv(seasonal_file)
                seasonal_data['data_year'] = year
                all_seasonal.append(seasonal_data)
                print(f"   ✅ Loaded {len(seasonal_data)} seasonal records from {year}")
                
            if os.path.exists(weekly_file):
                weekly_data = pd.read_csv(weekly_file)
                weekly_data['data_year'] = year
                all_weekly.append(weekly_data)
                print(f"   ✅ Loaded {len(weekly_data)} weekly records from {year}")
                
            if os.path.exists(roster_file):
                roster_data = pd.read_csv(roster_file)
                roster_data['data_year'] = year
                all_rosters.append(roster_data)
                print(f"   ✅ Loaded {len(roster_data)} roster records from {year}")
        
        # Combine all data
        if all_seasonal:
            combined_seasonal = pd.concat(all_seasonal, ignore_index=True)
            combined_seasonal.to_csv(f"{self.data_dir}/combined_seasonal_2022_2024.csv", index=False)
            print(f"   📊 Combined seasonal dataset: {len(combined_seasonal)} records")
            
        if all_weekly:
            combined_weekly = pd.concat(all_weekly, ignore_index=True)
            combined_weekly.to_csv(f"{self.data_dir}/combined_weekly_2022_2024.csv", index=False)
            print(f"   📅 Combined weekly dataset: {len(combined_weekly)} records")
            
        if all_rosters:
            combined_rosters = pd.concat(all_rosters, ignore_index=True)
            combined_rosters.to_csv(f"{self.data_dir}/combined_rosters_2022_2024.csv", index=False)
            print(f"   👥 Combined roster dataset: {len(combined_rosters)} records")
            
    def summarize_data(self):
        """Print summary of collected data"""
        print("\n📋 DATA COLLECTION SUMMARY")
        print("=" * 50)
        
        data_files = [
            "combined_seasonal_2022_2024.csv",
            "combined_weekly_2022_2024.csv", 
            "combined_rosters_2022_2024.csv"
        ]
        
        for filename in data_files:
            filepath = f"{self.data_dir}/{filename}"
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                print(f"✅ {filename}")
                print(f"   Records: {len(df):,}")
                if 'position' in df.columns:
                    print(f"   Positions: {df['position'].value_counts().to_dict()}")
                if 'data_year' in df.columns:
                    print(f"   Years: {sorted(df['data_year'].unique())}")
                print()
        
    def collect_all(self):
        """Run complete data collection process"""
        print("🚀 STARTING COMPLETE NFLVERSE DATA COLLECTION")
        print("=" * 60)
        
        # Collect all data types
        self.collect_seasonal_stats()
        self.collect_weekly_stats()
        self.collect_player_info()
        
        # Create combined datasets
        self.create_training_dataset()
        
        # Show summary
        self.summarize_data()
        
        print("\n🎉 NFLverse data collection complete!")
        print(f"📁 Data saved to: {self.data_dir}/")

def main():
    """Main execution function"""
    collector = NFLverseDataCollector()
    collector.collect_all()

if __name__ == "__main__":
    main() 