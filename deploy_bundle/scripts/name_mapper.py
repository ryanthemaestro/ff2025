#!/usr/bin/env python3
"""
Name mapping utility to match players between different data sources
Converts NFLsavant "J.SMITH" format to FantasyPros "Josh Allen" format
"""

# Common NFLsavant -> FantasyPros name mappings
# This is a starter set - can be expanded as needed
NAME_MAPPINGS = {
    # Top players that should definitely match
    'J.SMITH': 'Jaylen Smith',  # This might be wrong but let's see
    'S.BARKLEY': 'Saquon Barkley',
    'J.CHASE': 'Ja\'Marr Chase', 
    'J.GIBBS': 'Jahmyr Gibbs',
    'J.JACOBS': 'Josh Jacobs',
    'D.MOORE': 'DJ Moore',
    'J.JEFFERSON': 'Justin Jefferson',
    'D.HENRY': 'Derrick Henry',
    'B.ROBINSON': 'Bijan Robinson',  # ATL rookie RB
    'BROBINSON': 'Bijan Robinson',  # ATL rookie RB (no period)
    'BRIAN ROBINSON JR.': 'Brian Robinson Jr.',  # WAS veteran RB
    'A.ST. BROWN': 'Amon-Ra St. Brown',
    'A.ST': 'Amon-Ra St. Brown',
    'T.HILL': 'Tyreek Hill',
    'C.LAMB': 'CeeDee Lamb',
    'M.EVANS': 'Mike Evans',
    'D.ADAMS': 'Davante Adams',
    'A.COOPER': 'Amari Cooper',
    'K.ALLEN': 'Keenan Allen',
    'T.KELCE': 'Travis Kelce',
    'G.KITTLE': 'George Kittle',
    'M.ANDREWS': 'Mark Andrews',
    'J.ALLEN': 'Josh Allen',
    'L.JACKSON': 'Lamar Jackson',
    'P.MAHOMES': 'Patrick Mahomes II',
    'J.BURROW': 'Joe Burrow',
    'J.HURTS': 'Jalen Hurts',
    'D.PRESCOTT': 'Dak Prescott',
    'T.TAGOVAILOA': 'Tua Tagovailoa',
    'C.STROUD': 'C.J. Stroud',
    'A.RICHARDSON': 'Anthony Richardson',
    'J.DANIELS': 'Jayden Daniels',
    'C.WILLIAMS': 'Caleb Williams',
    'B.YOUNG': 'Bryce Young',
    
    # More RBs
    'C.MCCAFFREY': 'Christian McCaffrey',
    'A.EKELER': 'Austin Ekeler',
    'N.CHUBB': 'Nick Chubb',
    'J.TAYLOR': 'Jonathan Taylor',
    'A.JONES': 'Aaron Jones',
    'D.COOK': 'Dalvin Cook',
    'T.POLLARD': 'Tony Pollard',
    'R.STEVENSON': 'Rhamondre Stevenson',
    'I.PACHECO': 'Isiah Pacheco',
    'K.WALKER': 'Kenneth Walker III',
    'B.HALL': 'Breece Hall',
    'J.CONNER': 'James Conner',
    'T.ETIENNE': 'Travis Etienne Jr.',
    'N.HARRIS': 'Najee Harris',
    'R.WHITE': 'Rachaad White',
    'D.MONTGOMERY': 'David Montgomery',
    
    # More WRs
    'T.DIGGS': 'Stefon Diggs',
    'A.BROWN': 'A.J. Brown',
    'M.PITTMAN': 'Michael Pittman Jr.',
    'C.RIDLEY': 'Calvin Ridley',
    'G.WILSON': 'Garrett Wilson',
    'C.OLAVE': 'Chris Olave',
    'J.WADDLE': 'Jaylen Waddle',
    'D.JOHNSON': 'Diontae Johnson',
    'T.MCLAURIN': 'Terry McLaurin',
    'C.SUTTON': 'Courtland Sutton',
    'M.BROWN': 'Marquise Brown',
    'B.AIYUK': 'Brandon Aiyuk',
    'D.METCALF': 'DK Metcalf',
    'J.JEUDY': 'Jerry Jeudy',
    'A.THIELEN': 'Adam Thielen',
    'K.GOLLADAY': 'Kenny Golladay',
    
    # More TEs
    'D.WALLER': 'Darren Waller',
    'T.HOCKENSON': 'T.J. Hockenson',
    'E.ENGRAM': 'Evan Engram',
    'D.GOEDERT': 'Dallas Goedert',
    'P.FREIERMUTH': 'Pat Freiermuth',
    'T.MCBRIDE': 'Trey McBride',
    'C.KMET': 'Cole Kmet',
    'I.SMITH': 'Irv Smith Jr.'
}

def nflsavant_to_fantasypros(nflsavant_name):
    """Convert NFLsavant format name to FantasyPros format"""
    import pandas as pd
    if not nflsavant_name or pd.isna(nflsavant_name):
        return ''
    
    nflsavant_name = str(nflsavant_name).strip().upper()
    
    # Direct lookup first
    if nflsavant_name in NAME_MAPPINGS:
        return NAME_MAPPINGS[nflsavant_name]
    
    # Try some basic transformations for names not in our mapping
    if '.' in nflsavant_name:
        parts = nflsavant_name.split('.')
        if len(parts) == 2:
            initial = parts[0]
            last_name = parts[1]
            
            # Common first name expansions
            first_name_guesses = {
                'J': ['Josh', 'Justin', 'James', 'Jalen', 'Jordan', 'Jaylen'],
                'A': ['Austin', 'Aaron', 'Anthony', 'Adam', 'Antonio'],
                'C': ['Chris', 'Calvin', 'Cameron', 'Christian'],
                'D': ['Dalvin', 'Derrick', 'DJ', 'David', 'Dak'],
                'T': ['Tyler', 'Travis', 'Tyreek', 'Tony', 'Tua'],
                'M': ['Mike', 'Michael', 'Mark', 'Marquise'],
                'B': ['Brandon', 'Brian', 'Breece', 'Baker'],
                'S': ['Saquon', 'Stefon', 'Sam'],
                'R': ['Rhamondre', 'Russell', 'Rachaad'],
                'K': ['Kenneth', 'Keenan', 'Kyle']
            }
            
            if initial in first_name_guesses:
                # Return the first guess with last name
                return f"{first_name_guesses[initial][0]} {last_name.title()}"
    
    # If no match found, return original
    return nflsavant_name

def create_name_variants(full_name):
    """Create variants of a full name for matching"""
    import pandas as pd
    if not full_name or pd.isna(full_name):
        return []
    
    full_name = str(full_name).strip()
    variants = [full_name.upper()]
    
    # Add variants without suffixes
    name_clean = full_name.replace(' Jr.', '').replace(' Jr', '').replace(' Sr.', '').replace(' Sr', '')
    name_clean = name_clean.replace(' III', '').replace(' II', '').replace(' IV', '')
    variants.append(name_clean.upper())
    
    # Add variants with first name initial
    parts = full_name.split()
    if len(parts) >= 2:
        first_initial = parts[0][0] if parts[0] else ''
        last_name = parts[-1]
        if first_initial and last_name:
            variants.append(f"{first_initial}.{last_name.upper()}")
    
    return list(set(variants))  # Remove duplicates

def find_best_name_match(nflsavant_name, fantasypros_names):
    """Find the best match for an NFLsavant name in a list of FantasyPros names"""
    if not nflsavant_name:
        return None
    
    nflsavant_name = str(nflsavant_name).strip()
    
    # Try direct mapping first
    mapped_name = nflsavant_to_fantasypros(nflsavant_name)
    for fp_name in fantasypros_names:
        fp_variants = create_name_variants(fp_name)
        if mapped_name.upper() in [v.upper() for v in fp_variants]:
            return fp_name
    
    # Try matching variants
    nfl_variants = create_name_variants(nflsavant_name)
    for fp_name in fantasypros_names:
        fp_variants = create_name_variants(fp_name)
        for nfl_var in nfl_variants:
            for fp_var in fp_variants:
                if nfl_var.upper() == fp_var.upper():
                    return fp_name
    
    return None

if __name__ == "__main__":
    # Test the mapping
    import pandas as pd
    
    test_names = ['J.SMITH', 'S.BARKLEY', 'J.CHASE', 'UNKNOWN.PLAYER']
    print("Testing name conversions:")
    for name in test_names:
        converted = nflsavant_to_fantasypros(name)
        print(f"  {name} -> {converted}") 