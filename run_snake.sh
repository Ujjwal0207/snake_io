#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

echo "======================================"
echo "Starting Snake AI (Visual Training)"
echo "======================================"

# Activate virtual environment if it exists (checks both venv and .venv)
if [ -f "venv/bin/activate" ]; then
    echo "-> Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "-> Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "-> No virtual environment found, using system Python..."
fi

# Check if requirements are installed (rudimentary check by trying to import pygame)
python -c "import pygame" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "-> Dependencies missing! Installing them now..."
    pip install -r requirements.txt
fi

# Run the main training file
python main.py
