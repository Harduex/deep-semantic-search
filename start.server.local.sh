#!/bin/bash

# Load environment variables from .env file
if [[ -f .env ]]; then
    export $(cat .env | xargs)
fi

# Create a virtual environment and install the server dependencies
cd ./server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python ./src/api.py
