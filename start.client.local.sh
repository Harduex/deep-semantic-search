#!/bin/bash

# Load environment variables from .env file
if [[ -f .env ]]; then
    export $(cat .env | xargs)
fi

# Start the client
cd ./client
nvm use
yarn
yarn dev
