#!/bin/bash
# Activate the virtual environment for GENAI projects
echo "Activating GENAI virtual environment..."

# Check if venv exists, if not try .venv
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: No virtual environment found!"
    echo "Please run: source setup_venv.sh"
    return 1
fi

echo "Virtual environment activated! You can now run your AI projects."
echo "To deactivate, run: deactivate"
