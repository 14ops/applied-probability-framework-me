# AI Agent Performance Visualization Dashboard

A comprehensive visualization system for monitoring AI agent performance, improvements, and metrics in real-time.

## Features

### 🤖 AI Agent Management
- **Multiple Agent Types**: Deep Reinforcement Learning, Monte Carlo Tree Search, Bayesian Probability, Adversarial Training, and Ensemble Learning
- **Real-time Status Tracking**: Monitor active, training, and inactive agents
- **Performance Metrics**: Win rate, average payout, total games, and improvement tracking
- **Configuration Management**: View and update agent parameters

### 📊 Interactive Visualizations
- **Performance Over Time**: Line charts showing win rate and payout trends
- **Improvement Comparison**: Bar charts comparing agent improvements
- **Status Distribution**: Pie charts showing agent status breakdown
- **Performance Scatter Plot**: Win rate vs average payout correlation
- **Detailed Metrics**: Comprehensive view of agent-specific parameters

### 🔄 Real-time Updates
- **Live Data Simulation**: Simulate agent improvements and performance updates
- **Auto-refresh**: Automatic data updates from the backend API
- **Manual Controls**: Trigger simulations and refresh data on demand

## Architecture

### Frontend (React + Chart.js)
- **Location**: `/frontend/dashboard/src/components/AIVisualization.jsx`
- **Dependencies**: React, Chart.js, react-chartjs-2
- **Features**: Interactive charts, real-time updates, responsive design

### Backend (Python + Flask)
- **Location**: `/backend/python/src/`
- **API Server**: `ai_api.py` - RESTful API endpoints
- **Data Management**: `ai_agent_tracker.py` - SQLite database operations
- **Features**: Agent tracking, performance history, simulation capabilities

### Database (SQLite)
- **Agents Table**: Agent configurations and metadata
- **Performance Table**: Historical performance data
- **Real-time Updates**: Live performance tracking

## Quick Start

### Option 1: Automated Startup
```bash
cd /workspace
./start_ai_dashboard.sh
```

### Option 2: Manual Startup

#### Backend (Terminal 1)
```bash
cd /workspace/backend/python
pip install -r requirements.txt
cd src
python3 ai_api.py
```

#### Frontend (Terminal 2)
```bash
cd /workspace/frontend/dashboard
npm install
npm run dev
```

### Access the Dashboard
- **Frontend**: http://localhost:5173
- **API**: http://localhost:5001

## API Endpoints

### Agent Management
- `GET /api/agents` - Get all agents
- `GET /api/agents/<id>` - Get specific agent
- `PUT /api/agents/<id>/status` - Update agent status

### Performance Data
- `GET /api/agents/<id>/performance` - Get performance history
- `GET /api/metrics/improvement` - Get improvement metrics
- `GET /api/metrics/status-distribution` - Get status distribution

### Simulation
- `POST /api/agents/<id>/simulate` - Simulate agent improvement
- `POST /api/agents/simulate-all` - Simulate all agents

## Dashboard Features

### Agent Cards
- **Performance Overview**: Win rate, average payout, total games, improvement
- **Status Indicators**: Visual status badges (active, training, inactive)
- **Interactive Selection**: Click to view detailed metrics
- **Simulation Controls**: Individual agent improvement simulation

### Charts and Visualizations
1. **Performance Over Time**: Line chart showing win rate and payout trends
2. **Improvement Comparison**: Bar chart comparing agent improvements
3. **Status Distribution**: Pie chart showing agent status breakdown
4. **Performance Scatter Plot**: Win rate vs average payout correlation

### Controls
- **View Modes**: Overview, detailed view, comparison
- **Time Ranges**: 24 hours, 7 days, 30 days
- **Simulation Controls**: Simulate individual agents or all agents
- **Refresh Controls**: Manual data refresh

## Data Structure

### Agent Object
```json
{
  "agent": {
    "agent_id": "drl_agent",
    "name": "Deep Reinforcement Learning Agent",
    "agent_type": "DRL",
    "status": "active",
    "created_at": "2025-01-01T00:00:00Z",
    "last_updated": "2025-01-14T10:30:00Z",
    "configuration": { ... }
  },
  "performance": {
    "win_rate": 0.73,
    "avg_payout": 1.45,
    "total_games": 1250,
    "improvement": 0.12,
    "last_update": "2025-01-14T10:30:00Z"
  },
  "metrics": { ... }
}
```

### Performance History
```json
{
  "id": 1,
  "agent_id": "drl_agent",
  "win_rate": 0.73,
  "avg_payout": 1.45,
  "total_games": 1250,
  "improvement": 0.12,
  "timestamp": "2025-01-14T10:30:00Z",
  "metrics": { ... }
}
```

## Customization

### Adding New Agent Types
1. Update `ai_agent_tracker.py` with new agent configuration
2. Add agent to `initialize_default_agents()` method
3. Update frontend visualization if needed

### Modifying Visualizations
1. Edit `AIVisualization.jsx` component
2. Update chart configurations in `getPerformanceChartData()`, `getImprovementChartData()`, etc.
3. Add new chart types using Chart.js

### Database Schema Changes
1. Modify SQLite schema in `init_database()` method
2. Update data access methods accordingly
3. Handle migration for existing data

## Troubleshooting

### Common Issues
1. **API Connection Error**: Ensure Python API is running on port 5001
2. **Chart Not Rendering**: Check Chart.js dependencies are installed
3. **Data Not Loading**: Verify database initialization and API endpoints

### Debug Mode
- **Backend**: Set `debug=True` in `ai_api.py`
- **Frontend**: Check browser console for errors
- **Database**: Check SQLite file permissions and data integrity

## Future Enhancements

### Planned Features
- **Real-time WebSocket Updates**: Live performance streaming
- **Advanced Analytics**: Machine learning insights and predictions
- **Agent Comparison Tools**: Side-by-side performance analysis
- **Export Capabilities**: Data export in various formats
- **Alert System**: Performance threshold notifications
- **Historical Analysis**: Long-term trend analysis and forecasting

### Technical Improvements
- **Database Optimization**: Indexing and query optimization
- **Caching Layer**: Redis for improved performance
- **Authentication**: User management and access control
- **API Rate Limiting**: Request throttling and security
- **Monitoring**: Application performance monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Applied Probability and Automation Framework for High-RTP Games research project.