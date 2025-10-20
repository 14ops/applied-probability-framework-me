"""
Flask API for AI Agent Data
Provides REST endpoints for AI agent performance data
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
import sys
from datetime import datetime, timedelta

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_agent_tracker import tracker

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get all AI agents"""
    try:
        agents = tracker.get_all_agent_summaries()
        return jsonify({
            "success": True,
            "data": agents,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/agents/<agent_id>', methods=['GET'])
def get_agent(agent_id):
    """Get specific agent by ID"""
    try:
        summary = tracker.get_agent_summary(agent_id)
        if not summary:
            return jsonify({
                "success": False,
                "error": "Agent not found",
                "timestamp": datetime.now().isoformat()
            }), 404
        
        return jsonify({
            "success": True,
            "data": summary,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/agents/<agent_id>/performance', methods=['GET'])
def get_agent_performance(agent_id):
    """Get performance history for an agent"""
    try:
        days = request.args.get('days', 7, type=int)
        performance_data = tracker.get_agent_performance(agent_id, days)
        
        return jsonify({
            "success": True,
            "data": performance_data,
            "agent_id": agent_id,
            "days": days,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/agents/<agent_id>/status', methods=['PUT'])
def update_agent_status(agent_id):
    """Update agent status"""
    try:
        data = request.get_json()
        if not data or 'status' not in data:
            return jsonify({
                "success": False,
                "error": "Status is required",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        success = tracker.update_agent_status(agent_id, data['status'])
        if not success:
            return jsonify({
                "success": False,
                "error": "Failed to update agent status",
                "timestamp": datetime.now().isoformat()
            }), 500
        
        return jsonify({
            "success": True,
            "message": f"Agent {agent_id} status updated to {data['status']}",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/agents/<agent_id>/simulate', methods=['POST'])
def simulate_agent_improvement(agent_id):
    """Simulate agent improvement"""
    try:
        tracker.simulate_agent_improvement(agent_id)
        return jsonify({
            "success": True,
            "message": f"Simulated improvement for agent {agent_id}",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/agents/simulate-all', methods=['POST'])
def simulate_all_agents():
    """Simulate improvements for all active agents"""
    try:
        from ai_agent_tracker import simulate_agent_updates
        simulate_agent_updates()
        return jsonify({
            "success": True,
            "message": "Simulated improvements for all active agents",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/metrics/improvement', methods=['GET'])
def get_improvement_metrics():
    """Get improvement metrics for all agents"""
    try:
        agents = tracker.get_all_agent_summaries()
        metrics = []
        
        for agent_summary in agents:
            agent = agent_summary["agent"]
            perf = agent_summary["performance"]
            
            metrics.append({
                "agent_id": agent["agent_id"],
                "agent_name": agent["name"],
                "agent_type": agent["agent_type"],
                "win_rate": perf["win_rate"],
                "avg_payout": perf["avg_payout"],
                "improvement": perf["improvement"],
                "total_games": perf["total_games"],
                "status": agent["status"]
            })
        
        # Sort by improvement
        metrics.sort(key=lambda x: x["improvement"], reverse=True)
        
        return jsonify({
            "success": True,
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/metrics/status-distribution', methods=['GET'])
def get_status_distribution():
    """Get status distribution of agents"""
    try:
        agents = tracker.get_all_agents()
        distribution = {}
        
        for agent in agents:
            status = agent["status"]
            distribution[status] = distribution.get(status, 0) + 1
        
        return jsonify({
            "success": True,
            "data": distribution,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "success": True,
        "message": "AI Agent API is running",
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    print("Starting AI Agent API...")
    print("Available endpoints:")
    print("- GET /api/agents - Get all agents")
    print("- GET /api/agents/<id> - Get specific agent")
    print("- GET /api/agents/<id>/performance - Get agent performance history")
    print("- PUT /api/agents/<id>/status - Update agent status")
    print("- POST /api/agents/<id>/simulate - Simulate agent improvement")
    print("- POST /api/agents/simulate-all - Simulate all agents")
    print("- GET /api/metrics/improvement - Get improvement metrics")
    print("- GET /api/metrics/status-distribution - Get status distribution")
    print("- GET /api/health - Health check")
    
    app.run(host='0.0.0.0', port=5001, debug=True)