#!/bin/bash
# Script to update requirements.txt with current packages

echo "🔄 Updating requirements.txt..."

# Activate virtual environment
source venv/bin/activate

# Update requirements.txt
pip freeze > requirements.txt

echo "✅ requirements.txt updated!"
echo "📦 Current packages:"
pip list --format=freeze
