#!/bin/bash

# Load environment variables from .env file
if [[ -f .env ]]; then
    export $(cat .env | xargs)
fi

# Build and start the containers
docker-compose up -d --build
docker exec -it ollama-api ollama pull $OLLAMA_LLM_MODEL
echo "Model $OLLAMA_LLM_MODEL pulled successfully!"
