#!/bin/bash
# Run the Docker container with data and model directories mounted for local access

docker run \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/model:/app/model" \
  housing-model
