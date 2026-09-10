#! /bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment in 'venv'..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate
echo "Installing requirements..."
pip install -r requirements.txt
echo "Starting bot"

# Initialize settings.json from settings.json.example if missing
if [ ! -f settings.json ] && [ -f settings.json.example ]; then
    cp settings.json.example settings.json
fi

# Determine whether to run in the background from settings.json (fallback to RUN_IN_BACKGROUND env)
BG_SETTING=$(python3 -c "import json; print(str(json.load(open('settings.json')).get('run_in_background', False)).lower())" 2>/dev/null || echo "false")

if [ "$RUN_IN_BACKGROUND" = "true" ] || [ "$BG_SETTING" = "true" ]; then
    nohup python3 run_bot.py > output.log 2>&1 &
    echo "Bot is running in the background. Logs are in output.log"
    echo "To stop the bot, run ./stop.sh"
else
    python3 run_bot.py
fi
