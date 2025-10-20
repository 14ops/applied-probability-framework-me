#!/bin/bash

# AI Dashboard Startup Script
# Starts both the Python backend API and React frontend

echo "🚀 Starting AI Agent Performance Dashboard..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed or not in PATH"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cd /workspace/backend/python
pip install flask flask-cors numpy sqlite3

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd /workspace/frontend/dashboard
npm install

# Start the Python API in the background
echo "🐍 Starting Python API server..."
cd /workspace/backend/python/src
python3 ai_api.py &
API_PID=$!

# Wait a moment for the API to start
sleep 3

# Start the React development server
echo "⚛️ Starting React development server..."
cd /workspace/frontend/dashboard
npm run dev &
FRONTEND_PID=$!

echo "✅ Dashboard started successfully!"
echo "🌐 Frontend: http://localhost:5173"
echo "🔌 API: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $API_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait