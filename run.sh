#!/bin/bash
# Giano Computer Vision Runner Script

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the project directory
cd "$SCRIPT_DIR"

# Run the computer vision main module
.venv/bin/python -m src.computer_vision.main "$@"