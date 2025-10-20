// AI Agent Data Service
// Manages AI agent data, performance tracking, and improvement metrics

class AIAgentService {
  constructor() {
    this.agents = new Map();
    this.performanceHistory = new Map();
    this.improvementMetrics = new Map();
    this.initializeDefaultAgents();
  }

  initializeDefaultAgents() {
    const defaultAgents = [
      {
        id: 'drl_agent',
        name: 'Deep Reinforcement Learning Agent',
        type: 'DRL',
        description: 'Uses Q-learning and neural networks to optimize decision making',
        status: 'active',
        createdAt: new Date('2025-01-01'),
        lastUpdated: new Date(),
        performance: {
          winRate: 0.73,
          avgPayout: 1.45,
          totalGames: 1250,
          improvement: 0.12,
          lastUpdate: new Date()
        },
        metrics: {
          learningRate: 0.001,
          epsilon: 0.15,
          qTableSize: 10000,
          episodes: 5000,
          memorySize: 100000,
          batchSize: 32
        },
        configuration: {
          stateSpace: 'board_state',
          actionSpace: 'cell_selection',
          rewardFunction: 'payout_based',
          explorationStrategy: 'epsilon_greedy'
        }
      },
      {
        id: 'monte_carlo_agent',
        name: 'Monte Carlo Tree Search Agent',
        type: 'MCTS',
        description: 'Uses tree search with random sampling for optimal move selection',
        status: 'active',
        createdAt: new Date('2025-01-02'),
        lastUpdated: new Date(),
        performance: {
          winRate: 0.68,
          avgPayout: 1.38,
          totalGames: 980,
          improvement: 0.08,
          lastUpdate: new Date()
        },
        metrics: {
          simulations: 1000,
          explorationConstant: 1.4,
          maxDepth: 20,
          iterations: 2500,
          timeLimit: 1000,
          selectionPolicy: 'UCB1'
        },
        configuration: {
          selectionPolicy: 'UCB1',
          expansionStrategy: 'random',
          simulationStrategy: 'random_rollout',
          backpropagationStrategy: 'average'
        }
      },
      {
        id: 'bayesian_agent',
        name: 'Bayesian Probability Agent',
        type: 'Bayesian',
        description: 'Uses Bayesian inference to update probability estimates',
        status: 'active',
        createdAt: new Date('2025-01-03'),
        lastUpdated: new Date(),
        performance: {
          winRate: 0.71,
          avgPayout: 1.42,
          totalGames: 1100,
          improvement: 0.15,
          lastUpdate: new Date()
        },
        metrics: {
          priorStrength: 0.1,
          confidenceLevel: 0.95,
          sampleSize: 500,
          iterations: 3000,
          convergenceThreshold: 0.001,
          updateFrequency: 10
        },
        configuration: {
          priorDistribution: 'beta',
          likelihoodFunction: 'binomial',
          updateRule: 'bayesian',
          confidenceInterval: 0.95
        }
      },
      {
        id: 'adversarial_agent',
        name: 'Adversarial Training Agent',
        type: 'Adversarial',
        description: 'Uses generative adversarial networks for robust decision making',
        status: 'training',
        createdAt: new Date('2025-01-04'),
        lastUpdated: new Date(),
        performance: {
          winRate: 0.65,
          avgPayout: 1.35,
          totalGames: 750,
          improvement: 0.20,
          lastUpdate: new Date()
        },
        metrics: {
          generatorLearningRate: 0.0002,
          discriminatorLearningRate: 0.0001,
          epochs: 100,
          batchSize: 32,
          discriminatorSteps: 1,
          generatorSteps: 1
        },
        configuration: {
          architecture: 'DCGAN',
          lossFunction: 'binary_crossentropy',
          optimizer: 'Adam',
          regularization: 'L2'
        }
      },
      {
        id: 'ensemble_agent',
        name: 'Ensemble Learning Agent',
        type: 'Ensemble',
        description: 'Combines multiple models for improved prediction accuracy',
        status: 'active',
        createdAt: new Date('2025-01-05'),
        lastUpdated: new Date(),
        performance: {
          winRate: 0.75,
          avgPayout: 1.48,
          totalGames: 1500,
          improvement: 0.18,
          lastUpdate: new Date()
        },
        metrics: {
          numModels: 5,
          votingMethod: 'weighted',
          confidenceThreshold: 0.8,
          retrainingFrequency: 100,
          diversityMeasure: 0.7,
          consensusThreshold: 0.6
        },
        configuration: {
          baseModels: ['DRL', 'MCTS', 'Bayesian', 'RandomForest', 'SVM'],
          aggregationMethod: 'weighted_average',
          weightUpdateStrategy: 'performance_based',
          diversityStrategy: 'bagging'
        }
      }
    ];

    defaultAgents.forEach(agent => {
      this.agents.set(agent.id, agent);
      this.initializePerformanceHistory(agent.id);
    });
  }

  initializePerformanceHistory(agentId) {
    const agent = this.agents.get(agentId);
    if (!agent) return;

    const history = [];
    const startDate = new Date(agent.createdAt);
    const now = new Date();
    const daysDiff = Math.ceil((now - startDate) / (1000 * 60 * 60 * 24));

    // Generate historical performance data
    for (let i = 0; i < Math.min(daysDiff, 30); i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      // Simulate gradual improvement with some randomness
      const baseWinRate = 0.5 + (i * 0.01) + (Math.random() - 0.5) * 0.05;
      const basePayout = 1.2 + (i * 0.01) + (Math.random() - 0.5) * 0.1;
      
      history.push({
        date: date.toISOString().split('T')[0],
        winRate: Math.max(0.1, Math.min(0.95, baseWinRate)),
        avgPayout: Math.max(1.0, Math.min(2.0, basePayout)),
        totalGames: Math.floor(50 + i * 10 + Math.random() * 20),
        improvement: Math.max(0, (Math.random() - 0.3) * 0.1)
      });
    }

    this.performanceHistory.set(agentId, history);
  }

  // Get all agents
  getAllAgents() {
    return Array.from(this.agents.values());
  }

  // Get specific agent
  getAgent(agentId) {
    return this.agents.get(agentId);
  }

  // Get agent performance history
  getAgentPerformanceHistory(agentId, timeRange = '7d') {
    const history = this.performanceHistory.get(agentId) || [];
    const now = new Date();
    const days = timeRange === '24h' ? 1 : timeRange === '7d' ? 7 : 30;
    const cutoffDate = new Date(now.getTime() - (days * 24 * 60 * 60 * 1000));
    
    return history.filter(entry => new Date(entry.date) >= cutoffDate);
  }

  // Update agent performance
  updateAgentPerformance(agentId, performanceData) {
    const agent = this.agents.get(agentId);
    if (!agent) return false;

    const oldPerformance = { ...agent.performance };
    agent.performance = {
      ...agent.performance,
      ...performanceData,
      lastUpdate: new Date()
    };

    // Calculate improvement
    const winRateImprovement = agent.performance.winRate - oldPerformance.winRate;
    const payoutImprovement = agent.performance.avgPayout - oldPerformance.avgPayout;
    agent.performance.improvement = (winRateImprovement + payoutImprovement) / 2;

    // Update performance history
    const today = new Date().toISOString().split('T')[0];
    const history = this.performanceHistory.get(agentId) || [];
    const todayEntry = history.find(entry => entry.date === today);
    
    if (todayEntry) {
      todayEntry.winRate = agent.performance.winRate;
      todayEntry.avgPayout = agent.performance.avgPayout;
      todayEntry.totalGames = agent.performance.totalGames;
      todayEntry.improvement = agent.performance.improvement;
    } else {
      history.push({
        date: today,
        winRate: agent.performance.winRate,
        avgPayout: agent.performance.avgPayout,
        totalGames: agent.performance.totalGames,
        improvement: agent.performance.improvement
      });
    }

    this.performanceHistory.set(agentId, history);
    agent.lastUpdated = new Date();
    
    return true;
  }

  // Get improvement metrics for all agents
  getImprovementMetrics() {
    const metrics = [];
    
    this.agents.forEach((agent, agentId) => {
      const history = this.performanceHistory.get(agentId) || [];
      if (history.length < 2) return;

      const recent = history.slice(-7); // Last 7 days
      const older = history.slice(-14, -7); // Previous 7 days
      
      if (recent.length === 0 || older.length === 0) return;

      const recentAvgWinRate = recent.reduce((sum, entry) => sum + entry.winRate, 0) / recent.length;
      const olderAvgWinRate = older.reduce((sum, entry) => sum + entry.winRate, 0) / older.length;
      const recentAvgPayout = recent.reduce((sum, entry) => sum + entry.avgPayout, 0) / recent.length;
      const olderAvgPayout = older.reduce((sum, entry) => sum + entry.avgPayout, 0) / older.length;

      metrics.push({
        agentId,
        agentName: agent.name,
        winRateImprovement: recentAvgWinRate - olderAvgWinRate,
        payoutImprovement: recentAvgPayout - olderAvgPayout,
        overallImprovement: ((recentAvgWinRate - olderAvgWinRate) + (recentAvgPayout - olderAvgPayout)) / 2,
        currentWinRate: recentAvgWinRate,
        currentAvgPayout: recentAvgPayout,
        totalGames: recent.reduce((sum, entry) => sum + entry.totalGames, 0)
      });
    });

    return metrics.sort((a, b) => b.overallImprovement - a.overallImprovement);
  }

  // Get performance comparison data
  getPerformanceComparison() {
    const agents = this.getAllAgents();
    return agents.map(agent => ({
      id: agent.id,
      name: agent.name,
      type: agent.type,
      winRate: agent.performance.winRate,
      avgPayout: agent.performance.avgPayout,
      totalGames: agent.performance.totalGames,
      improvement: agent.performance.improvement,
      status: agent.status
    }));
  }

  // Get status distribution
  getStatusDistribution() {
    const agents = this.getAllAgents();
    const distribution = {};
    
    agents.forEach(agent => {
      distribution[agent.status] = (distribution[agent.status] || 0) + 1;
    });

    return distribution;
  }

  // Add new agent
  addAgent(agentData) {
    const agentId = agentData.id || `agent_${Date.now()}`;
    const agent = {
      id: agentId,
      createdAt: new Date(),
      lastUpdated: new Date(),
      performance: {
        winRate: 0.5,
        avgPayout: 1.0,
        totalGames: 0,
        improvement: 0,
        lastUpdate: new Date()
      },
      ...agentData
    };

    this.agents.set(agentId, agent);
    this.initializePerformanceHistory(agentId);
    return agentId;
  }

  // Remove agent
  removeAgent(agentId) {
    this.agents.delete(agentId);
    this.performanceHistory.delete(agentId);
    this.improvementMetrics.delete(agentId);
  }

  // Update agent configuration
  updateAgentConfiguration(agentId, configuration) {
    const agent = this.agents.get(agentId);
    if (!agent) return false;

    agent.configuration = { ...agent.configuration, ...configuration };
    agent.lastUpdated = new Date();
    return true;
  }

  // Get agent metrics for specific time range
  getAgentMetrics(agentId, timeRange = '7d') {
    const history = this.getAgentPerformanceHistory(agentId, timeRange);
    if (history.length === 0) return null;

    const winRates = history.map(entry => entry.winRate);
    const payouts = history.map(entry => entry.avgPayout);
    const improvements = history.map(entry => entry.improvement);

    return {
      avgWinRate: winRates.reduce((sum, rate) => sum + rate, 0) / winRates.length,
      avgPayout: payouts.reduce((sum, payout) => sum + payout, 0) / payouts.length,
      avgImprovement: improvements.reduce((sum, imp) => sum + imp, 0) / improvements.length,
      maxWinRate: Math.max(...winRates),
      minWinRate: Math.min(...winRates),
      maxPayout: Math.max(...payouts),
      minPayout: Math.min(...payouts),
      totalGames: history.reduce((sum, entry) => sum + entry.totalGames, 0),
      volatility: this.calculateVolatility(winRates)
    };
  }

  // Calculate volatility (standard deviation)
  calculateVolatility(values) {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  // Export agent data
  exportAgentData(agentId) {
    const agent = this.agents.get(agentId);
    const history = this.performanceHistory.get(agentId);
    
    if (!agent) return null;

    return {
      agent,
      performanceHistory: history,
      metrics: this.getAgentMetrics(agentId),
      exportedAt: new Date().toISOString()
    };
  }

  // Import agent data
  importAgentData(data) {
    try {
      if (data.agent) {
        this.agents.set(data.agent.id, data.agent);
      }
      if (data.performanceHistory) {
        this.performanceHistory.set(data.agent.id, data.performanceHistory);
      }
      return true;
    } catch (error) {
      console.error('Error importing agent data:', error);
      return false;
    }
  }
}

// Create singleton instance
const aiAgentService = new AIAgentService();

export default aiAgentService;