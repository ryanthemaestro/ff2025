#!/bin/bash
set -e

echo "🏈 Fantasy Football Draft UI Startup"
echo "===================================="

# Navigate to project directory
cd /home/nar/ff2025

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Set environment variables to prevent graphics issues
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
export DISPLAY=""

# Install packages from requirements to ensure all deps (incl. joblib) are present
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "🚀 Starting Flask server..."
echo "📍 Once started, open: http://localhost:5000"
echo "📝 Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
python scripts/draft_ui.py 