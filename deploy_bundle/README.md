# Fantasy Football Draft UI

A smart fantasy football draft assistant with AI-powered player recommendations.

## Features

- 🤖 **AI-Powered Recommendations**: Uses a leak-free CatBoost model trained on NFLverse data
- 📊 **Position Scarcity Analysis**: Boosts recommendations based on team needs
- 🆕 **Rookie Integration**: Separate rookie rankings with proper bye week mapping
- 🏥 **Injury Tracking**: Real-time injury status from NFLverse
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

- **Backend**: Python Flask with Netlify Functions
- **Frontend**: HTML/CSS/JavaScript with jQuery
- **AI Model**: CatBoost (scikit-learn compatible)
- **Data Sources**: FantasyPros ADP Rankings, NFLverse historical data

## Deployment

This app is deployed on Netlify using serverless functions.

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Flask app:
   ```bash
   python scripts/draft_ui.py
   ```

3. Open http://localhost:5000

### Netlify Deployment

The app is configured for automatic deployment on Netlify with:
- Serverless functions in `/functions/`
- Static assets in `/static/` 
- Build configuration in `netlify.toml`

## Data Sources

- **Player Rankings**: FantasyPros 2025 Overall ADP Rankings
- **Rookie Data**: FantasyPros 2025 Rookie Rankings  
- **Historical Stats**: NFLverse data (2022-2024)
- **Injury Data**: NFLverse injury reports

## AI Model

The AI model was trained on historical player performance data with:
- **No Data Leakage**: Uses historical stats to predict future performance
- **Realistic Predictions**: R² = 0.279 (realistic, not overfitted)
- **Position-Aware**: Different models for QB, RB, WR, TE
- **Scarcity Integration**: Recommendations boosted by positional needs 