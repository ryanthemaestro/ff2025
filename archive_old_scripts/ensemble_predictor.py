#!/usr/bin/env python3
"""
Ensemble Prediction System
"""


def ensemble_predict(player_data, position, models, adp_rank=None):
    """
    Ensemble prediction combining multiple approaches
    """
    predictions = {}
    
    # 1. Position-specific CatBoost prediction (70% weight)
    if position in models and models[position]:
        try:
            model_data = models[position]
            model = model_data['model']
            features = model_data['features']
            
            # Prepare features for this player
            player_features = []
            for feature in features:
                player_features.append(player_data.get(feature, 0))
            
            catboost_pred = model.predict([player_features])[0]
            predictions['catboost'] = max(0, catboost_pred)  # No negative predictions
            
        except Exception as e:
            print(f"Warning: CatBoost prediction failed for {position}: {e}")
            predictions['catboost'] = 0
    else:
        predictions['catboost'] = 0
    
    # 2. ADP consensus (15% weight)
    if adp_rank and adp_rank > 0:
        # Convert ADP rank to point expectation (rough heuristic)
        if adp_rank <= 12:  # Round 1
            adp_points = 250 - (adp_rank * 15)
        elif adp_rank <= 24:  # Round 2
            adp_points = 200 - ((adp_rank - 12) * 10)
        elif adp_rank <= 36:  # Round 3
            adp_points = 150 - ((adp_rank - 24) * 8)
        else:
            adp_points = max(50, 120 - (adp_rank - 36) * 2)
        
        predictions['adp_consensus'] = max(0, adp_points)
    else:
        predictions['adp_consensus'] = 100  # Default neutral
    
    # 3. Volume projection (10% weight)
    volume_score = 0
    if position == 'QB':
        volume_score = player_data.get('passing_yards', 0) * 0.04 + player_data.get('passing_tds', 0) * 4
    elif position == 'RB':
        volume_score = player_data.get('carries', 0) * 0.1 + player_data.get('targets', 0) * 0.5
    elif position in ['WR', 'TE']:
        volume_score = player_data.get('targets', 0) * 0.8 + player_data.get('receiving_yards', 0) * 0.1
    
    predictions['volume_projection'] = max(0, volume_score)
    
    # 4. SOS adjustment (5% weight)
    sos_score = player_data.get('sos_score', 3.0)
    sos_adjustment = 100 * (sos_score / 3.0)  # Neutral is 3.0
    predictions['sos_adjustment'] = sos_adjustment
    
    # Combine with ensemble weights
    ensemble_weights = {
        'catboost': 0.70,
        'adp_consensus': 0.15,
        'volume_projection': 0.10,
        'sos_adjustment': 0.05
    }
    
    final_prediction = 0
    for component, weight in ensemble_weights.items():
        if component in predictions:
            final_prediction += predictions[component] * weight
    
    return max(0, final_prediction), predictions
