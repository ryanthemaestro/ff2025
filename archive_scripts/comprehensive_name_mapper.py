#!/usr/bin/env python3
"""
Comprehensive Name Mapper using FantasyPros ADP data as source of truth
Creates flexible search that works with both abbreviated and full names
"""

import pandas as pd
import re
from difflib import SequenceMatcher

class ComprehensiveNameMapper:
    def __init__(self):
        self.adp_data = None
        self.name_mappings = {}
        self.reverse_mappings = {}
        self.load_adp_data()
        self.create_mappings()
    
    def load_adp_data(self):
        """Load FantasyPros ADP data as source of truth"""
        try:
            # Use error_bad_lines=False to skip malformed lines
            self.adp_data = pd.read_csv('data/FantasyPros_2025_Overall_ADP_Rankings.csv', 
                                       on_bad_lines='skip', engine='python')
            print(f"✅ Loaded {len(self.adp_data)} players from ADP rankings")
        except FileNotFoundError:
            print("❌ Could not find FantasyPros ADP rankings file")
            self.adp_data = pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading ADP data: {e}")
            self.adp_data = pd.DataFrame()
    
    def normalize_name(self, name):
        """Normalize name for comparison"""
        if pd.isna(name) or not name:
            return ""
        
        # Remove extra spaces, quotes, periods
        name = str(name).strip().replace('"', '').replace("'", "")
        # Convert to uppercase for consistent comparison
        return name.upper()
    
    def create_abbreviated_name(self, full_name):
        """Convert full name to abbreviated format (e.g., 'Ja'Marr Chase' -> 'J.CHASE')"""
        if pd.isna(full_name) or not full_name:
            return ""
        
        # Clean the name
        name = str(full_name).strip().replace('"', '').replace("'", "")
        
        # Split into parts
        parts = name.split()
        if len(parts) < 2:
            return name.upper()
        
        # Take first letter of first name + last name
        first_initial = parts[0][0].upper()
        last_name = parts[-1].upper()
        
        # Remove common suffixes
        last_name = re.sub(r'\s+(JR\.?|SR\.?|III|IV|V)$', '', last_name)
        
        return f"{first_initial}.{last_name}"
    
    def create_mappings(self):
        """Create bidirectional mappings between abbreviated and full names"""
        if self.adp_data.empty:
            return
        
        for _, row in self.adp_data.iterrows():
            full_name = row['Player']
            abbreviated = self.create_abbreviated_name(full_name)
            
            # Store mappings (both directions)
            self.name_mappings[abbreviated] = full_name
            self.reverse_mappings[self.normalize_name(full_name)] = full_name
            
            # Also store without period for flexibility
            abbreviated_no_period = abbreviated.replace('.', '')
            self.name_mappings[abbreviated_no_period] = full_name
        
        print(f"✅ Created {len(self.name_mappings)} name mappings")
    
    def get_full_name(self, name):
        """Convert any name format to full name"""
        if pd.isna(name) or not name:
            return name
        
        name_clean = str(name).strip()
        name_upper = self.normalize_name(name_clean)
        
        # Direct mapping lookup
        if name_upper in self.name_mappings:
            return self.name_mappings[name_upper]
        
        # Check reverse mapping (already full name)
        if name_upper in self.reverse_mappings:
            return self.reverse_mappings[name_upper]
        
        # Fuzzy matching for variations
        best_match = self.find_fuzzy_match(name_clean)
        if best_match:
            return best_match
        
        # Return original if no match found
        return name_clean
    
    def find_fuzzy_match(self, name, threshold=0.8):
        """Find best fuzzy match for a name"""
        name_normalized = self.normalize_name(name)
        best_score = 0
        best_match = None
        
        # Check against all full names in ADP data
        for _, row in self.adp_data.iterrows():
            full_name = row['Player']
            full_normalized = self.normalize_name(full_name)
            abbreviated = self.create_abbreviated_name(full_name)
            
            # Check similarity with full name
            score1 = SequenceMatcher(None, name_normalized, full_normalized).ratio()
            # Check similarity with abbreviated name
            score2 = SequenceMatcher(None, name_normalized, abbreviated).ratio()
            
            max_score = max(score1, score2)
            if max_score > best_score and max_score >= threshold:
                best_score = max_score
                best_match = full_name
        
        return best_match
    
    def search_players(self, query, limit=10):
        """Search for players by name (flexible matching)"""
        if not query or self.adp_data.empty:
            return []
        
        query_normalized = self.normalize_name(query)
        matches = []
        
        for _, row in self.adp_data.iterrows():
            full_name = row['Player']
            abbreviated = self.create_abbreviated_name(full_name)
            
            # Check if query matches full name or abbreviated name
            if (query_normalized in self.normalize_name(full_name) or 
                query_normalized in abbreviated or
                self.normalize_name(full_name).startswith(query_normalized) or
                abbreviated.startswith(query_normalized)):
                
                matches.append({
                    'name': full_name,
                    'abbreviated': abbreviated,
                    'team': row['Team'],
                    'position': row['POS'],
                    'bye_week': row['Bye'],
                    'adp_rank': row['Rank']
                })
        
        return matches[:limit]
    
    def get_player_info(self, name):
        """Get comprehensive player info from ADP data"""
        full_name = self.get_full_name(name)
        
        # Find in ADP data
        match = self.adp_data[self.adp_data['Player'].str.upper() == full_name.upper()]
        if not match.empty:
            row = match.iloc[0]
            return {
                'name': row['Player'],
                'team': row['Team'],
                'position': row['POS'],
                'bye_week': row['Bye'],
                'adp_rank': row['Rank'],
                'adp_avg': row['AVG']
            }
        
        return None

# Global instance for easy access
name_mapper = ComprehensiveNameMapper()

def get_full_name(name):
    """Convenience function to get full name"""
    return name_mapper.get_full_name(name)

def search_players(query, limit=10):
    """Convenience function to search players"""
    return name_mapper.search_players(query, limit)

def get_player_info(name):
    """Convenience function to get player info"""
    return name_mapper.get_player_info(name)

if __name__ == "__main__":
    # Test the mapper
    test_names = ["J.Chase", "S.Barkley", "Ja'Marr Chase", "Bijan Robinson"]
    
    print("Testing name mappings:")
    for name in test_names:
        full_name = get_full_name(name)
        print(f"  {name} -> {full_name}")
    
    print("\nTesting search:")
    results = search_players("Chase")
    for result in results:
        print(f"  {result['name']} ({result['position']}) - ADP #{result['adp_rank']}") 