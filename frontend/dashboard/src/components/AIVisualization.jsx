import { useState, useEffect } from 'react';
import { Line, Bar, Scatter, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const AIVisualization = () => {
  const [aiAgents, setAIAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [timeRange, setTimeRange] = useState('7d');
  const [viewMode, setViewMode] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [performanceHistory, setPerformanceHistory] = useState({});

  // API base URL
  const API_BASE_URL = 'http://localhost:5001/api';

  // Fetch AI agents data
  const fetchAIAgents = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/agents`);
      const data = await response.json();
      
      if (data.success) {
        setAIAgents(data.data);
        if (data.data.length > 0) {
          setSelectedAgent(data.data[0]);
        }
      } else {
        setError(data.error || 'Failed to fetch agents');
      }
    } catch (err) {
      setError('Failed to connect to AI agent API');
      console.error('Error fetching agents:', err);
    } finally {
      setLoading(false);
    }
  };

  // Fetch performance history for an agent
  const fetchPerformanceHistory = async (agentId) => {
    try {
      const days = timeRange === '24h' ? 1 : timeRange === '7d' ? 7 : 30;
      const response = await fetch(`${API_BASE_URL}/agents/${agentId}/performance?days=${days}`);
      const data = await response.json();
      
      if (data.success) {
        setPerformanceHistory(prev => ({
          ...prev,
          [agentId]: data.data
        }));
      }
    } catch (err) {
      console.error('Error fetching performance history:', err);
    }
  };

  // Simulate agent improvement
  const simulateImprovement = async (agentId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/agents/${agentId}/simulate`, {
        method: 'POST'
      });
      const data = await response.json();
      
      if (data.success) {
        // Refresh the agents data
        await fetchAIAgents();
        // Refresh performance history
        await fetchPerformanceHistory(agentId);
      }
    } catch (err) {
      console.error('Error simulating improvement:', err);
    }
  };

  // Simulate all agents
  const simulateAllAgents = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/agents/simulate-all`, {
        method: 'POST'
      });
      const data = await response.json();
      
      if (data.success) {
        // Refresh the agents data
        await fetchAIAgents();
        // Refresh performance history for selected agent
        if (selectedAgent) {
          await fetchPerformanceHistory(selectedAgent.agent.agent_id);
        }
      }
    } catch (err) {
      console.error('Error simulating all agents:', err);
    }
  };

  useEffect(() => {
    fetchAIAgents();
  }, []);

  useEffect(() => {
    if (selectedAgent) {
      fetchPerformanceHistory(selectedAgent.agent.agent_id);
    }
  }, [selectedAgent, timeRange]);

  const getPerformanceChartData = (agent) => {
    if (!agent || !performanceHistory[agent.agent.agent_id]) {
      return { datasets: [] };
    }

    const history = performanceHistory[agent.agent.agent_id];
    const winRateData = history.map(entry => ({ 
      x: entry.timestamp.split('T')[0], 
      y: entry.win_rate 
    }));
    
    const payoutData = history.map(entry => ({ 
      x: entry.timestamp.split('T')[0], 
      y: entry.avg_payout 
    }));

    return {
      datasets: [
        {
          label: 'Win Rate',
          data: winRateData,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          tension: 0.4,
          yAxisID: 'y'
        },
        {
          label: 'Average Payout',
          data: payoutData,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          tension: 0.4,
          yAxisID: 'y1'
        }
      ]
    };
  };

  const getImprovementChartData = () => {
    const agents = aiAgents.map(agent => ({
      name: agent.agent.name,
      improvement: agent.performance.improvement,
      winRate: agent.performance.win_rate,
      avgPayout: agent.performance.avg_payout
    }));

    return {
      labels: agents.map(a => a.name.split(' ')[0]),
      datasets: [
        {
          label: 'Improvement Rate',
          data: agents.map(a => a.improvement),
          backgroundColor: [
            'rgba(59, 130, 246, 0.8)',
            'rgba(16, 185, 129, 0.8)',
            'rgba(245, 158, 11, 0.8)',
            'rgba(239, 68, 68, 0.8)',
            'rgba(139, 92, 246, 0.8)'
          ],
          borderColor: [
            'rgba(59, 130, 246, 1)',
            'rgba(16, 185, 129, 1)',
            'rgba(245, 158, 11, 1)',
            'rgba(239, 68, 68, 1)',
            'rgba(139, 92, 246, 1)'
          ],
          borderWidth: 2
        }
      ]
    };
  };

  const getStatusDistribution = () => {
    const statusCounts = aiAgents.reduce((acc, agent) => {
      acc[agent.agent.status] = (acc[agent.agent.status] || 0) + 1;
      return acc;
    }, {});

    return {
      labels: Object.keys(statusCounts).map(status => 
        status.charAt(0).toUpperCase() + status.slice(1)
      ),
      datasets: [{
        data: Object.values(statusCounts),
        backgroundColor: [
          'rgba(16, 185, 129, 0.8)',
          'rgba(245, 158, 11, 0.8)',
          'rgba(239, 68, 68, 0.8)'
        ],
        borderColor: [
          'rgba(16, 185, 129, 1)',
          'rgba(245, 158, 11, 1)',
          'rgba(239, 68, 68, 1)'
        ],
        borderWidth: 2
      }]
    };
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#e2e8f0'
        }
      },
      title: {
        display: true,
        color: '#e2e8f0'
      }
    },
    scales: {
      x: {
        ticks: { color: '#94a3b8' },
        grid: { color: 'rgba(148, 163, 184, 0.1)' }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        ticks: { color: '#94a3b8' },
        grid: { color: 'rgba(148, 163, 184, 0.1)' }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        ticks: { color: '#94a3b8' },
        grid: { drawOnChartArea: false }
      }
    }
  };

  const pieOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#e2e8f0'
        }
      }
    }
  };

  if (loading) {
    return (
      <div style={{
        minHeight: '100vh',
        padding: '2rem',
        background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
        color: '#e2e8f0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{
            width: '50px',
            height: '50px',
            border: '3px solid #374151',
            borderTop: '3px solid #3b82f6',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 1rem'
          }}></div>
          <p>Loading AI agents data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        minHeight: '100vh',
        padding: '2rem',
        background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
        color: '#e2e8f0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <div style={{ textAlign: 'center' }}>
          <h2 style={{ color: '#ef4444', marginBottom: '1rem' }}>Error Loading Data</h2>
          <p style={{ color: '#94a3b8', marginBottom: '1rem' }}>{error}</p>
          <button
            onClick={fetchAIAgents}
            style={{
              padding: '0.75rem 1.5rem',
              borderRadius: '0.5rem',
              border: 'none',
              background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
              color: '#e2e8f0',
              cursor: 'pointer',
              fontWeight: '600'
            }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      minHeight: '100vh',
      padding: '2rem',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      color: '#e2e8f0'
    }}>
      <div style={{ maxWidth: '1600px', margin: '0 auto' }}>
        <header style={{
          marginBottom: '2rem',
          textAlign: 'center'
        }}>
          <h1 style={{
            fontSize: '2.5rem',
            fontWeight: '700',
            marginBottom: '0.5rem',
            background: 'linear-gradient(135deg, #60a5fa, #a78bfa)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            AI Agents Performance Dashboard
          </h1>
          <p style={{
            color: '#94a3b8',
            fontSize: '1.1rem',
            marginBottom: '1rem'
          }}>
            Real-time monitoring of AI improvements and performance metrics
          </p>
          
          {/* Action buttons */}
          <div style={{
            display: 'flex',
            gap: '1rem',
            justifyContent: 'center',
            marginBottom: '1rem'
          }}>
            <button
              onClick={simulateAllAgents}
              style={{
                padding: '0.5rem 1rem',
                borderRadius: '0.5rem',
                border: 'none',
                background: 'linear-gradient(135deg, #10b981, #059669)',
                color: '#e2e8f0',
                cursor: 'pointer',
                fontWeight: '600',
                fontSize: '0.875rem'
              }}
            >
              Simulate All Agents
            </button>
            <button
              onClick={fetchAIAgents}
              style={{
                padding: '0.5rem 1rem',
                borderRadius: '0.5rem',
                border: 'none',
                background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                color: '#e2e8f0',
                cursor: 'pointer',
                fontWeight: '600',
                fontSize: '0.875rem'
              }}
            >
              Refresh Data
            </button>
          </div>
        </header>

        {/* Control Panel */}
        <div style={{
          display: 'flex',
          gap: '1rem',
          marginBottom: '2rem',
          flexWrap: 'wrap',
          justifyContent: 'center'
        }}>
          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value)}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '0.5rem',
              border: '1px solid #374151',
              background: '#1f2937',
              color: '#e2e8f0'
            }}
          >
            <option value="overview">Overview</option>
            <option value="detailed">Detailed View</option>
            <option value="comparison">Comparison</option>
          </select>
          
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '0.5rem',
              border: '1px solid #374151',
              background: '#1f2937',
              color: '#e2e8f0'
            }}
          >
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
        </div>

        {/* Agent Cards */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '1.5rem',
          marginBottom: '2rem'
        }}>
          {aiAgents.map(agent => (
            <div
              key={agent.agent.agent_id}
              onClick={() => setSelectedAgent(agent)}
              style={{
                background: selectedAgent?.agent.agent_id === agent.agent.agent_id 
                  ? 'linear-gradient(135deg, #1e40af, #3730a3)' 
                  : 'linear-gradient(135deg, #1f2937, #374151)',
                borderRadius: '1rem',
                padding: '1.5rem',
                cursor: 'pointer',
                border: selectedAgent?.agent.agent_id === agent.agent.agent_id 
                  ? '2px solid #3b82f6' 
                  : '1px solid #374151',
                transition: 'all 0.3s ease'
              }}
            >
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '1rem'
              }}>
                <h3 style={{
                  fontSize: '1.25rem',
                  fontWeight: '600',
                  margin: 0
                }}>
                  {agent.agent.name}
                </h3>
                <span style={{
                  padding: '0.25rem 0.75rem',
                  borderRadius: '1rem',
                  fontSize: '0.875rem',
                  fontWeight: '500',
                  background: agent.agent.status === 'active' 
                    ? 'rgba(16, 185, 129, 0.2)' 
                    : 'rgba(245, 158, 11, 0.2)',
                  color: agent.agent.status === 'active' 
                    ? '#10b981' 
                    : '#f59e0b'
                }}>
                  {agent.agent.status}
                </span>
              </div>
              
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '1rem'
              }}>
                <div>
                  <p style={{ color: '#94a3b8', fontSize: '0.875rem', margin: '0 0 0.25rem 0' }}>
                    Win Rate
                  </p>
                  <p style={{ fontSize: '1.5rem', fontWeight: '700', margin: 0 }}>
                    {(agent.performance.win_rate * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p style={{ color: '#94a3b8', fontSize: '0.875rem', margin: '0 0 0.25rem 0' }}>
                    Avg Payout
                  </p>
                  <p style={{ fontSize: '1.5rem', fontWeight: '700', margin: 0 }}>
                    {agent.performance.avg_payout.toFixed(2)}x
                  </p>
                </div>
                <div>
                  <p style={{ color: '#94a3b8', fontSize: '0.875rem', margin: '0 0 0.25rem 0' }}>
                    Total Games
                  </p>
                  <p style={{ fontSize: '1.25rem', fontWeight: '600', margin: 0 }}>
                    {agent.performance.total_games.toLocaleString()}
                  </p>
                </div>
                <div>
                  <p style={{ color: '#94a3b8', fontSize: '0.875rem', margin: '0 0 0.25rem 0' }}>
                    Improvement
                  </p>
                  <p style={{ 
                    fontSize: '1.25rem', 
                    fontWeight: '600', 
                    margin: 0,
                    color: agent.performance.improvement > 0.1 ? '#10b981' : '#f59e0b'
                  }}>
                    +{(agent.performance.improvement * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              
              {/* Action button for individual agent */}
              <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    simulateImprovement(agent.agent.agent_id);
                  }}
                  style={{
                    padding: '0.5rem 1rem',
                    borderRadius: '0.5rem',
                    border: 'none',
                    background: 'rgba(59, 130, 246, 0.2)',
                    color: '#3b82f6',
                    cursor: 'pointer',
                    fontWeight: '600',
                    fontSize: '0.875rem',
                    border: '1px solid #3b82f6'
                  }}
                >
                  Simulate Improvement
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Charts Section */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
          gap: '2rem',
          marginBottom: '2rem'
        }}>
          {/* Performance Over Time */}
          {selectedAgent && (
            <div style={{
              background: 'linear-gradient(135deg, #1f2937, #374151)',
              borderRadius: '1rem',
              padding: '1.5rem',
              border: '1px solid #374151'
            }}>
              <h3 style={{
                fontSize: '1.25rem',
                fontWeight: '600',
                marginBottom: '1rem',
                textAlign: 'center'
              }}>
                {selectedAgent.agent.name} - Performance Over Time
              </h3>
              <div style={{ height: '300px' }}>
                <Line data={getPerformanceChartData(selectedAgent)} options={chartOptions} />
              </div>
            </div>
          )}

          {/* Improvement Comparison */}
          <div style={{
            background: 'linear-gradient(135deg, #1f2937, #374151)',
            borderRadius: '1rem',
            padding: '1.5rem',
            border: '1px solid #374151'
          }}>
            <h3 style={{
              fontSize: '1.25rem',
              fontWeight: '600',
              marginBottom: '1rem',
              textAlign: 'center'
            }}>
              Agent Improvement Comparison
            </h3>
            <div style={{ height: '300px' }}>
              <Bar data={getImprovementChartData()} options={chartOptions} />
            </div>
          </div>

          {/* Status Distribution */}
          <div style={{
            background: 'linear-gradient(135deg, #1f2937, #374151)',
            borderRadius: '1rem',
            padding: '1.5rem',
            border: '1px solid #374151'
          }}>
            <h3 style={{
              fontSize: '1.25rem',
              fontWeight: '600',
              marginBottom: '1rem',
              textAlign: 'center'
            }}>
              Agent Status Distribution
            </h3>
            <div style={{ height: '300px' }}>
              <Pie data={getStatusDistribution()} options={pieOptions} />
            </div>
          </div>

          {/* Performance Scatter Plot */}
          <div style={{
            background: 'linear-gradient(135deg, #1f2937, #374151)',
            borderRadius: '1rem',
            padding: '1.5rem',
            border: '1px solid #374151'
          }}>
            <h3 style={{
              fontSize: '1.25rem',
              fontWeight: '600',
              marginBottom: '1rem',
              textAlign: 'center'
            }}>
              Win Rate vs Average Payout
            </h3>
            <div style={{ height: '300px' }}>
              <Scatter 
                data={{
                  datasets: [{
                    label: 'AI Agents',
                    data: aiAgents.map(agent => ({
                      x: agent.performance.win_rate,
                      y: agent.performance.avg_payout
                    })),
                    backgroundColor: 'rgba(59, 130, 246, 0.6)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    pointRadius: 8
                  }]
                }}
                options={{
                  responsive: true,
                  plugins: {
                    legend: {
                      labels: { color: '#e2e8f0' }
                    }
                  },
                  scales: {
                    x: {
                      title: {
                        display: true,
                        text: 'Win Rate',
                        color: '#e2e8f0'
                      },
                      ticks: { color: '#94a3b8' },
                      grid: { color: 'rgba(148, 163, 184, 0.1)' }
                    },
                    y: {
                      title: {
                        display: true,
                        text: 'Average Payout',
                        color: '#e2e8f0'
                      },
                      ticks: { color: '#94a3b8' },
                      grid: { color: 'rgba(148, 163, 184, 0.1)' }
                    }
                  }
                }}
              />
            </div>
          </div>
        </div>

        {/* Detailed Metrics Table */}
        {selectedAgent && viewMode === 'detailed' && (
          <div style={{
            background: 'linear-gradient(135deg, #1f2937, #374151)',
            borderRadius: '1rem',
            padding: '1.5rem',
            border: '1px solid #374151'
          }}>
            <h3 style={{
              fontSize: '1.25rem',
              fontWeight: '600',
              marginBottom: '1rem',
              textAlign: 'center'
            }}>
              {selectedAgent.agent.name} - Detailed Metrics
            </h3>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '1rem'
            }}>
              {Object.entries(selectedAgent.metrics).map(([key, value]) => (
                <div key={key} style={{
                  background: 'rgba(0, 0, 0, 0.2)',
                  borderRadius: '0.5rem',
                  padding: '1rem',
                  textAlign: 'center'
                }}>
                  <p style={{
                    color: '#94a3b8',
                    fontSize: '0.875rem',
                    margin: '0 0 0.5rem 0',
                    textTransform: 'capitalize'
                  }}>
                    {key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ').trim()}
                  </p>
                  <p style={{
                    fontSize: '1.25rem',
                    fontWeight: '600',
                    margin: 0
                  }}>
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AIVisualization;