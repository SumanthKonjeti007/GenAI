#!/bin/bash
# Script to create a new virtual environment for GENAI projects

echo "Removing old virtual environment if it exists..."
rm -rf venv .venv

echo "Creating new virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Virtual environment created and activated!"
echo "To activate in the future, run: source venv/bin/activate"
echo "To deactivate, run: deactivate"
echo ""
echo "Installing requirements..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    echo "Requirements installed!"
else
    echo "No requirements.txt found. Skipping package installation."
fi


