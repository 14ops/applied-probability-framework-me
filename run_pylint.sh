#!/bin/bash

# Install pylint if not already installed
sudo pip install pylint

# Run pylint on all Python files in src/python
find src/python -name "*.py" -exec pylint {} \;

echo "Pylint analysis complete!"

