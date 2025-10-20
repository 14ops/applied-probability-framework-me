#!/bin/bash

# Install Black if not already installed
sudo pip install black

# Find all Python files and format them with Black
find src/python -name "*.py" -exec python3.11 -m black {} \;

echo "Python code formatting complete!"

