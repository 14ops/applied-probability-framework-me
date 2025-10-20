#!/bin/bash

echo "🤖 Starting AI Improvement Visualizer..."
echo ""
echo "📊 This dashboard shows all AI agents and their improvements:"
echo "   • 10+ AI agents (DRL, Strategy, Multi-Agent, etc.)"
echo "   • Real-time performance metrics"
echo "   • Improvement tracking and analytics"
echo "   • Interactive charts and visualizations"
echo ""

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Try different ports
PORTS=(3000 3001 3002 8080 8081)

for PORT in "${PORTS[@]}"; do
    echo "🚀 Trying to start server on port $PORT..."
    
    # Check if port is available
    if ! nc -z localhost $PORT 2>/dev/null; then
        echo "✅ Port $PORT is available, starting server..."
        PORT=$PORT npm run dev &
        SERVER_PID=$!
        
        # Wait a moment for server to start
        sleep 3
        
        # Check if server started successfully
        if kill -0 $SERVER_PID 2>/dev/null; then
            echo ""
            echo "🎉 AI Improvement Visualizer is running!"
            echo "🌐 Open your browser and navigate to: http://localhost:$PORT"
            echo ""
            echo "📋 Features available:"
            echo "   • AI Overview: See all AI agents and their status"
            echo "   • Improvements: Track all improvements over time"
            echo "   • Performance: Analytics and charts"
            echo ""
            echo "💡 Click on any AI card to filter other views"
            echo "🔄 Press Ctrl+C to stop the server"
            echo ""
            
            # Keep the script running
            wait $SERVER_PID
            exit 0
        else
            echo "❌ Failed to start on port $PORT"
        fi
    else
        echo "⚠️  Port $PORT is already in use"
    fi
done

echo "❌ Could not find an available port. Please check your system."
echo "💡 You can manually run: PORT=<your-port> npm run dev"