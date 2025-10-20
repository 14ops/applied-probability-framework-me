# Backend Integration Guide

This guide explains how to connect the AI Improvement Visualizer with your existing Python backend.

## Quick Setup

The visualizer is designed to work with your existing Python AI framework. It can:
- Display real AI agents from your Python backend
- Show actual improvement history 
- Visualize real performance metrics
- Send commands back to control AI training

## Python Backend API Endpoints

Add these endpoints to your Python backend to enable full integration:

### 1. Get AI Agents
```python
@app.route('/api/agents', methods=['GET'])
def get_agents():
    agents = []
    
    # DRL Agent
    if hasattr(your_system, 'drl_agent'):
        agents.append({
            'id': 'drl-agent',
            'name': 'Deep RL Agent',
            'type': 'DRLAgent',
            'status': 'training' if your_system.drl_agent.is_training else 'active',
            'description': 'Q-learning based agent with neural network',
            'accuracy': your_system.drl_agent.get_accuracy(),
            'win_rate': your_system.drl_agent.get_win_rate(),
            'avg_profit': your_system.drl_agent.get_avg_profit(),
            'efficiency': your_system.drl_agent.get_efficiency(),
            'improvement_count': your_system.drl_agent.improvement_count,
            'last_updated': your_system.drl_agent.last_updated.isoformat(),
            'version': your_system.drl_agent.version
        })
    
    # Add other agents similarly...
    # Multi-Agent System
    # Strategy Agents (Takeshi, Lelouch, etc.)
    # Adversarial Agent
    # Behavioral Agent
    
    return jsonify(agents)
```

### 2. Get Improvements History
```python
@app.route('/api/improvements', methods=['GET'])
def get_improvements():
    improvements = []
    
    # Get from your improvement tracking system
    for imp in your_system.improvement_history:
        improvements.append({
            'id': imp.id,
            'agent_id': imp.agent_id,
            'timestamp': imp.timestamp.isoformat(),
            'type': imp.type,  # 'performance', 'algorithm', 'strategy', 'optimization'
            'description': imp.description,
            'impact': imp.impact,  # 'high', 'medium', 'low'
            'before_value': imp.before_value,
            'after_value': imp.after_value,
            'improvement_percentage': imp.improvement_percentage
        })
    
    return jsonify(improvements)
```

### 3. Get Performance Data
```python
@app.route('/api/performance', methods=['GET'])
def get_performance():
    performance_data = []
    
    # Get historical performance data
    for record in your_system.performance_history:
        performance_data.append({
            'timestamp': record.timestamp.isoformat(),
            'agent_id': record.agent_id,
            'accuracy': record.accuracy,
            'win_rate': record.win_rate,
            'profit': record.profit,
            'efficiency': record.efficiency
        })
    
    return jsonify(performance_data)
```

### 4. System Status
```python
@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'total_agents': len(your_system.agents),
        'active_agents': len([a for a in your_system.agents if a.is_active]),
        'training_agents': len([a for a in your_system.agents if a.is_training]),
        'total_improvements': your_system.total_improvements,
        'system_health': 'good'  # or calculate based on your metrics
    })
```

### 5. Command Interface
```python
@app.route('/api/command', methods=['POST'])
def handle_command():
    data = request.json
    command = data.get('command')
    params = data.get('params', {})
    
    if command == 'start_training':
        ai_id = params.get('ai_id')
        agent = your_system.get_agent(ai_id)
        if agent:
            agent.start_training()
            return jsonify({'success': True})
    
    elif command == 'stop_training':
        ai_id = params.get('ai_id')
        agent = your_system.get_agent(ai_id)
        if agent:
            agent.stop_training()
            return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'Unknown command'})
```

## Configuration

Update the backend URL in `app/data/dataIntegration.ts`:

```typescript
private backendUrl: string = 'http://localhost:8000' // Your Python server URL
```

## Testing Integration

1. **Start your Python backend** with the API endpoints
2. **Start the visualizer**: `./start.sh`
3. **Check browser console** for any connection errors
4. **Verify data** appears correctly in the dashboard

## Fallback Behavior

The visualizer gracefully falls back to mock data if:
- Python backend is not running
- API endpoints return errors
- Network connectivity issues

This ensures the dashboard always works for demonstration purposes.

## Real-time Updates

For real-time updates, consider adding WebSocket support:

```python
# In your Python backend
from flask_socketio import SocketIO, emit

socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    
def broadcast_improvement(improvement_data):
    socketio.emit('new_improvement', improvement_data)
    
def broadcast_performance_update(performance_data):
    socketio.emit('performance_update', performance_data)
```

## Data Mapping

The integration layer automatically maps your Python data structures to the frontend format:

- **Agent Types**: Maps Python class names to display types
- **Status Values**: Converts Python status to UI-friendly states  
- **Metrics**: Normalizes performance metrics for visualization
- **Timestamps**: Handles datetime conversion

## Troubleshooting

### Common Issues

1. **CORS Errors**: Add CORS headers to your Python backend
2. **Port Conflicts**: Update the backend URL in dataIntegration.ts
3. **Data Format**: Ensure your API returns JSON in the expected format
4. **Authentication**: Add auth headers if your backend requires them

### Debug Mode

Enable debug logging by adding to your component:

```typescript
// In your React component
useEffect(() => {
  console.log('Fetching data from backend...')
  DataIntegration.fetchAIAgents().then(data => {
    console.log('Received AI data:', data)
  })
}, [])
```

## Next Steps

1. **Implement the API endpoints** in your Python backend
2. **Test the integration** with real data
3. **Customize the visualization** based on your specific metrics
4. **Add authentication** if needed for production use
5. **Set up real-time updates** for live monitoring

The visualizer is designed to be flexible and work with various AI frameworks. Adapt the integration layer as needed for your specific implementation.