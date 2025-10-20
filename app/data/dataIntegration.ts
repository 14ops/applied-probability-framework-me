// Data integration utilities to connect with Python backend
import { AIAgent, Improvement, PerformanceMetric } from './mockData'

export class DataIntegration {
  private static instance: DataIntegration
  private backendUrl: string = 'http://localhost:8000' // Python backend URL

  private constructor() {}

  static getInstance(): DataIntegration {
    if (!DataIntegration.instance) {
      DataIntegration.instance = new DataIntegration()
    }
    return DataIntegration.instance
  }

  // Fetch real AI data from Python backend
  async fetchAIAgents(): Promise<AIAgent[]> {
    try {
      const response = await fetch(`${this.backendUrl}/api/agents`)
      if (response.ok) {
        const data = await response.json()
        return this.transformPythonAIData(data)
      }
    } catch (error) {
      console.warn('Failed to fetch from backend, using mock data:', error)
    }
    
    // Fallback to mock data
    const { aiData } = await import('./mockData')
    return aiData
  }

  // Fetch improvement history
  async fetchImprovements(): Promise<Improvement[]> {
    try {
      const response = await fetch(`${this.backendUrl}/api/improvements`)
      if (response.ok) {
        const data = await response.json()
        return this.transformPythonImprovementData(data)
      }
    } catch (error) {
      console.warn('Failed to fetch improvements from backend, using mock data:', error)
    }
    
    // Fallback to mock data
    const { improvementData } = await import('./mockData')
    return improvementData
  }

  // Fetch performance metrics
  async fetchPerformanceData(): Promise<PerformanceMetric[]> {
    try {
      const response = await fetch(`${this.backendUrl}/api/performance`)
      if (response.ok) {
        const data = await response.json()
        return this.transformPythonPerformanceData(data)
      }
    } catch (error) {
      console.warn('Failed to fetch performance data from backend, using mock data:', error)
    }
    
    // Fallback to mock data
    const { performanceData } = await import('./mockData')
    return performanceData
  }

  // Transform Python AI data to frontend format
  private transformPythonAIData(pythonData: any[]): AIAgent[] {
    return pythonData.map(agent => ({
      id: agent.id || agent.name?.toLowerCase().replace(/\s+/g, '-'),
      name: agent.name || 'Unknown Agent',
      type: this.mapAgentType(agent.type || agent.class_name),
      status: agent.status || 'idle',
      description: agent.description || 'No description available',
      performance: {
        accuracy: agent.accuracy || 0,
        winRate: agent.win_rate || 0,
        avgProfit: agent.avg_profit || 0,
        efficiency: agent.efficiency || 0
      },
      improvements: agent.improvement_count || 0,
      lastUpdated: agent.last_updated || new Date().toISOString(),
      color: this.getColorForType(agent.type),
      version: agent.version || 'v1.0.0'
    }))
  }

  // Transform Python improvement data
  private transformPythonImprovementData(pythonData: any[]): Improvement[] {
    return pythonData.map(imp => ({
      id: imp.id || `imp-${Date.now()}-${Math.random()}`,
      aiId: imp.agent_id || imp.ai_id,
      timestamp: imp.timestamp || new Date().toISOString(),
      type: imp.type || 'performance',
      description: imp.description || 'Improvement applied',
      impact: imp.impact || 'medium',
      metrics: {
        before: imp.before_value || 0,
        after: imp.after_value || 0,
        improvement: imp.improvement_percentage || 0
      }
    }))
  }

  // Transform Python performance data
  private transformPythonPerformanceData(pythonData: any[]): PerformanceMetric[] {
    return pythonData.map(perf => ({
      timestamp: perf.timestamp || new Date().toISOString(),
      aiId: perf.agent_id || perf.ai_id,
      accuracy: perf.accuracy || 0,
      winRate: perf.win_rate || 0,
      profit: perf.profit || 0,
      efficiency: perf.efficiency || 0
    }))
  }

  // Map Python agent types to frontend types
  private mapAgentType(pythonType: string): AIAgent['type'] {
    const typeMap: Record<string, AIAgent['type']> = {
      'DRLAgent': 'DRL',
      'MultiAgentCore': 'Multi-Agent',
      'AdversarialAgent': 'Adversarial',
      'BehavioralAgent': 'Behavioral',
      'Strategy': 'Strategy',
      'BasicStrategy': 'Strategy',
      'AdvancedStrategy': 'Strategy',
      'TakeshiStrategy': 'Strategy',
      'LelouchStrategy': 'Strategy',
      'KazuyaStrategy': 'Strategy',
      'SenkuStrategy': 'Strategy',
      'RintaroOkabeStrategy': 'Strategy'
    }
    
    return typeMap[pythonType] || 'Strategy'
  }

  // Get color for agent type
  private getColorForType(type: string): string {
    const colorMap: Record<string, string> = {
      'DRL': '#3b82f6',
      'Strategy': '#8b5cf6',
      'Multi-Agent': '#f59e0b',
      'Adversarial': '#ef4444',
      'Behavioral': '#7c3aed'
    }
    
    return colorMap[this.mapAgentType(type)] || '#6b7280'
  }

  // Send command to Python backend
  async sendCommand(command: string, params: any = {}): Promise<any> {
    try {
      const response = await fetch(`${this.backendUrl}/api/command`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ command, params })
      })
      
      if (response.ok) {
        return await response.json()
      }
    } catch (error) {
      console.error('Failed to send command to backend:', error)
    }
    
    return null
  }

  // Start training for specific AI
  async startTraining(aiId: string): Promise<boolean> {
    const result = await this.sendCommand('start_training', { ai_id: aiId })
    return result?.success || false
  }

  // Stop training for specific AI
  async stopTraining(aiId: string): Promise<boolean> {
    const result = await this.sendCommand('stop_training', { ai_id: aiId })
    return result?.success || false
  }

  // Get real-time status
  async getSystemStatus(): Promise<any> {
    try {
      const response = await fetch(`${this.backendUrl}/api/status`)
      if (response.ok) {
        return await response.json()
      }
    } catch (error) {
      console.warn('Failed to fetch system status:', error)
    }
    
    return {
      total_agents: 10,
      active_agents: 6,
      training_agents: 2,
      total_improvements: 240,
      system_health: 'good'
    }
  }
}

export default DataIntegration.getInstance()