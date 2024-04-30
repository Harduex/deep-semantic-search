#!/bin/bash

# Load environment variables from .env file
if [[ -f .env ]]; then
    export $(cat .env | xargs)
fi

# Create a virtual environment and install the dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Ollama API and pull the model
docker compose up -d
docker exec -it ollama-api ollama pull $OLLAMA_LLM_MODEL
echo "Model $OLLAMA_LLM_MODEL pulled successfully!"

# Start the application
python ./src/app.py
