#!/bin/bash

source ~/.zshrc

# Stop script on first error
set -e

# Define the directories
PROJECT_DIR="/path/to/your/project"
OUTPUT_DIR="/path/to/output"

# Print a message to indicate the start of the process
echo "Starting script execution..."

# Activate virtual environment (if needed)
# source /path/to/venv/bin/activate

# 
echo "Pipe Video"
python3 tools/pipe_video.py

echo "Perform Inference"
bell python3 inference.py

# Print completion message
echo "All tasks completed successfully!"
