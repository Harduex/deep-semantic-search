#!/bin/bash

# Load environment variables from .env file
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# Create a virtual environment and install the dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Ollama API and pull the model
docker compose up -d
docker exec -it ollama-api ollama pull $OLLAMA_LLM_MODEL
echo "Model $OLLAMA_LLM_MODEL pulled successfully!"

# Start the api
python ./src/api.py
