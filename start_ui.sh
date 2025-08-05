#!/bin/bash

# Set environment to disable graphics
export MPLBACKEND=Agg
export DISPLAY=""

# Navigate to project directory
cd /home/nar/ff2025

# Activate virtual environment
source venv/bin/activate

echo "🚀 Starting Fantasy Football Draft UI..."
echo "📍 Server will be available at: http://localhost:5000"
echo "🔧 Filtering out graphics errors from Cursor IDE..."
echo ""

# Start the Flask app and filter out graphics errors
python scripts/draft_ui.py 2>&1 | grep -v "ERROR:.*angle_platform_impl\|ERROR:.*gl_display\|ERROR:.*gl_ozone_egl\|ERR: Display\.cpp\|ERROR:.*viz_main_impl" 