#!/bin/bash

# Set environment to disable graphics
export MPLBACKEND=Agg
export DISPLAY=""

# Navigate to project directory
cd /home/nar/ff2025

# Ensure virtual environment exists
if [ ! -d "venv" ]; then
  echo "❌ Virtual environment not found. Creating one..."
  python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install all dependencies from requirements.txt to guarantee joblib is present
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Start the Flask app and filter out graphics errors
python scripts/draft_ui.py 2>&1 | grep -v "ERROR:.*angle_platform_impl\|ERROR:.*gl_display\|ERROR:.*gl_ozone_egl\|ERR: Display\.cpp\|ERROR:.*viz_main_impl"