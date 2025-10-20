export interface AIAgent {
  id: string
  name: string
  type: 'DRL' | 'Strategy' | 'Multi-Agent' | 'Adversarial' | 'Behavioral'
  status: 'active' | 'training' | 'idle' | 'optimizing'
  description: string
  performance: {
    accuracy: number
    winRate: number
    avgProfit: number
    efficiency: number
  }
  improvements: number
  lastUpdated: string
  color: string
  version: string
}

export interface Improvement {
  id: string
  aiId: string
  timestamp: string
  type: 'performance' | 'algorithm' | 'strategy' | 'optimization'
  description: string
  impact: 'high' | 'medium' | 'low'
  metrics: {
    before: number
    after: number
    improvement: number
  }
}

export interface PerformanceMetric {
  timestamp: string
  aiId: string
  accuracy: number
  winRate: number
  profit: number
  efficiency: number
}

export const aiData: AIAgent[] = [
  {
    id: 'drl-agent',
    name: 'Deep RL Agent',
    type: 'DRL',
    status: 'training',
    description: 'Q-learning based agent with neural network approximation',
    performance: {
      accuracy: 87.5,
      winRate: 72.3,
      avgProfit: 156.8,
      efficiency: 91.2
    },
    improvements: 23,
    lastUpdated: '2024-10-20T10:30:00Z',
    color: '#3b82f6',
    version: 'v2.1.4'
  },
  {
    id: 'takeshi-strategy',
    name: 'Takeshi (Aggressive)',
    type: 'Strategy',
    status: 'active',
    description: 'High-risk, high-reward aggressive trading strategy',
    performance: {
      accuracy: 68.9,
      winRate: 65.1,
      avgProfit: 203.4,
      efficiency: 78.6
    },
    improvements: 15,
    lastUpdated: '2024-10-20T09:15:00Z',
    color: '#ef4444',
    version: 'v1.8.2'
  },
  {
    id: 'lelouch-strategy',
    name: 'Lelouch (Mastermind)',
    type: 'Strategy',
    status: 'active',
    description: 'Calculated strategic approach with game theory principles',
    performance: {
      accuracy: 92.1,
      winRate: 78.9,
      avgProfit: 189.7,
      efficiency: 94.3
    },
    improvements: 31,
    lastUpdated: '2024-10-20T11:45:00Z',
    color: '#8b5cf6',
    version: 'v2.3.1'
  },
  {
    id: 'kazuya-strategy',
    name: 'Kazuya (Conservative)',
    type: 'Strategy',
    status: 'active',
    description: 'Risk-averse strategy focused on capital preservation',
    performance: {
      accuracy: 89.3,
      winRate: 71.2,
      avgProfit: 134.5,
      efficiency: 88.7
    },
    improvements: 18,
    lastUpdated: '2024-10-20T08:20:00Z',
    color: '#10b981',
    version: 'v1.9.5'
  },
  {
    id: 'senku-strategy',
    name: 'Senku (Analytical)',
    type: 'Strategy',
    status: 'optimizing',
    description: 'Data-driven scientific approach with statistical analysis',
    performance: {
      accuracy: 94.7,
      winRate: 81.4,
      avgProfit: 167.2,
      efficiency: 96.1
    },
    improvements: 27,
    lastUpdated: '2024-10-20T12:10:00Z',
    color: '#06b6d4',
    version: 'v2.0.8'
  },
  {
    id: 'multi-agent-core',
    name: 'Multi-Agent System',
    type: 'Multi-Agent',
    status: 'active',
    description: 'Coordinated system of 10 collaborative agents',
    performance: {
      accuracy: 85.2,
      winRate: 74.8,
      avgProfit: 178.9,
      efficiency: 87.4
    },
    improvements: 42,
    lastUpdated: '2024-10-20T13:30:00Z',
    color: '#f59e0b',
    version: 'v3.1.2'
  },
  {
    id: 'adversarial-agent',
    name: 'Adversarial Agent',
    type: 'Adversarial',
    status: 'training',
    description: 'Agent trained to detect and counter adversarial attacks',
    performance: {
      accuracy: 79.6,
      winRate: 68.3,
      avgProfit: 142.1,
      efficiency: 82.9
    },
    improvements: 19,
    lastUpdated: '2024-10-20T14:15:00Z',
    color: '#dc2626',
    version: 'v1.5.7'
  },
  {
    id: 'behavioral-agent',
    name: 'Behavioral Agent',
    type: 'Behavioral',
    status: 'active',
    description: 'Prospect theory based agent with loss aversion modeling',
    performance: {
      accuracy: 88.4,
      winRate: 76.7,
      avgProfit: 159.3,
      efficiency: 90.8
    },
    improvements: 25,
    lastUpdated: '2024-10-20T15:00:00Z',
    color: '#7c3aed',
    version: 'v2.2.3'
  },
  {
    id: 'rintaro-okabe',
    name: 'Rintaro Okabe Strategy',
    type: 'Strategy',
    status: 'idle',
    description: 'Time-series analysis with predictive modeling',
    performance: {
      accuracy: 91.8,
      winRate: 77.5,
      avgProfit: 173.6,
      efficiency: 93.2
    },
    improvements: 22,
    lastUpdated: '2024-10-20T07:45:00Z',
    color: '#ec4899',
    version: 'v1.7.9'
  },
  {
    id: 'advanced-strategy',
    name: 'Advanced Strategy Engine',
    type: 'Strategy',
    status: 'optimizing',
    description: 'Meta-strategy that combines multiple approaches dynamically',
    performance: {
      accuracy: 96.2,
      winRate: 84.1,
      avgProfit: 198.7,
      efficiency: 97.8
    },
    improvements: 38,
    lastUpdated: '2024-10-20T16:20:00Z',
    color: '#059669',
    version: 'v3.0.1'
  }
]

export const improvementData: Improvement[] = [
  {
    id: 'imp-1',
    aiId: 'lelouch-strategy',
    timestamp: '2024-10-20T11:45:00Z',
    type: 'algorithm',
    description: 'Implemented advanced game theory Nash equilibrium calculations',
    impact: 'high',
    metrics: { before: 75.2, after: 78.9, improvement: 4.9 }
  },
  {
    id: 'imp-2',
    aiId: 'drl-agent',
    timestamp: '2024-10-20T10:30:00Z',
    type: 'performance',
    description: 'Optimized neural network architecture with attention mechanism',
    impact: 'high',
    metrics: { before: 69.8, after: 72.3, improvement: 3.6 }
  },
  {
    id: 'imp-3',
    aiId: 'senku-strategy',
    timestamp: '2024-10-20T12:10:00Z',
    type: 'optimization',
    description: 'Enhanced statistical analysis with Bayesian inference',
    impact: 'medium',
    metrics: { before: 79.7, after: 81.4, improvement: 2.1 }
  },
  {
    id: 'imp-4',
    aiId: 'multi-agent-core',
    timestamp: '2024-10-20T13:30:00Z',
    type: 'strategy',
    description: 'Improved inter-agent communication protocol',
    impact: 'high',
    metrics: { before: 71.2, after: 74.8, improvement: 5.1 }
  },
  {
    id: 'imp-5',
    aiId: 'advanced-strategy',
    timestamp: '2024-10-20T16:20:00Z',
    type: 'algorithm',
    description: 'Deployed meta-learning framework for strategy adaptation',
    impact: 'high',
    metrics: { before: 81.5, after: 84.1, improvement: 3.2 }
  },
  {
    id: 'imp-6',
    aiId: 'behavioral-agent',
    timestamp: '2024-10-20T15:00:00Z',
    type: 'performance',
    description: 'Refined prospect theory parameters based on recent data',
    impact: 'medium',
    metrics: { before: 74.9, after: 76.7, improvement: 2.4 }
  }
]

export const performanceData: PerformanceMetric[] = [
  // Generate time series data for the last 30 days
  ...Array.from({ length: 30 }, (_, i) => {
    const date = new Date()
    date.setDate(date.getDate() - (29 - i))
    
    return aiData.map(ai => ({
      timestamp: date.toISOString(),
      aiId: ai.id,
      accuracy: ai.performance.accuracy + (Math.random() - 0.5) * 10,
      winRate: ai.performance.winRate + (Math.random() - 0.5) * 8,
      profit: ai.performance.avgProfit + (Math.random() - 0.5) * 50,
      efficiency: ai.performance.efficiency + (Math.random() - 0.5) * 6
    }))
  }).flat()
]