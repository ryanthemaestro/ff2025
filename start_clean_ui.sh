#!/bin/bash
set -e

echo "🏈 Fantasy Football Draft Assistant"
echo "=================================="
echo ""

cd /home/nar/ff2025
source venv/bin/activate

echo "🚀 Starting clean draft UI..."
echo "📍 Open: http://localhost:5000"
echo "🔧 Press Ctrl+C to stop"
echo ""

python src/ui/draft_ui.py 