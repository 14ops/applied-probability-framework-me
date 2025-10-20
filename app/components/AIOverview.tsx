'use client'

import { motion } from 'framer-motion'
import { Brain, Activity, TrendingUp, Clock, Zap, Target, DollarSign, Gauge } from 'lucide-react'
import { AIAgent } from '../data/mockData'

interface AIOverviewProps {
  aiData: AIAgent[]
  selectedAI: string | null
  setSelectedAI: (id: string | null) => void
}

export default function AIOverview({ aiData, selectedAI, setSelectedAI }: AIOverviewProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <Activity className="w-4 h-4 text-green-400" />
      case 'training': return <Brain className="w-4 h-4 text-blue-400" />
      case 'optimizing': return <Zap className="w-4 h-4 text-yellow-400" />
      case 'idle': return <Clock className="w-4 h-4 text-gray-400" />
      default: return <Activity className="w-4 h-4" />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'DRL': return 'bg-blue-500/20 text-blue-300 border-blue-500/30'
      case 'Strategy': return 'bg-purple-500/20 text-purple-300 border-purple-500/30'
      case 'Multi-Agent': return 'bg-orange-500/20 text-orange-300 border-orange-500/30'
      case 'Adversarial': return 'bg-red-500/20 text-red-300 border-red-500/30'
      case 'Behavioral': return 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30'
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30'
    }
  }

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card-gradient rounded-xl p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Total AIs</p>
              <p className="text-2xl font-bold">{aiData.length}</p>
            </div>
            <Brain className="w-8 h-8 text-blue-400" />
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card-gradient rounded-xl p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Active</p>
              <p className="text-2xl font-bold text-green-400">
                {aiData.filter(ai => ai.status === 'active').length}
              </p>
            </div>
            <Activity className="w-8 h-8 text-green-400" />
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card-gradient rounded-xl p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Avg Win Rate</p>
              <p className="text-2xl font-bold text-purple-400">
                {(aiData.reduce((acc, ai) => acc + ai.performance.winRate, 0) / aiData.length).toFixed(1)}%
              </p>
            </div>
            <Target className="w-8 h-8 text-purple-400" />
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card-gradient rounded-xl p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Total Improvements</p>
              <p className="text-2xl font-bold text-cyan-400">
                {aiData.reduce((acc, ai) => acc + ai.improvements, 0)}
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-cyan-400" />
          </div>
        </motion.div>
      </div>

      {/* AI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {aiData.map((ai, index) => (
          <motion.div
            key={ai.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`ai-card card-gradient rounded-xl p-6 cursor-pointer border-2 transition-all duration-300 ${
              selectedAI === ai.id 
                ? 'border-blue-500/50 shadow-lg shadow-blue-500/25' 
                : 'border-transparent hover:border-white/20'
            }`}
            onClick={() => setSelectedAI(selectedAI === ai.id ? null : ai.id)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div 
                  className="w-12 h-12 rounded-lg flex items-center justify-center"
                  style={{ backgroundColor: ai.color + '20', border: `1px solid ${ai.color}30` }}
                >
                  <Brain className="w-6 h-6" style={{ color: ai.color }} />
                </div>
                <div>
                  <h3 className="font-semibold text-lg">{ai.name}</h3>
                  <p className="text-sm text-muted-foreground">{ai.version}</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {getStatusIcon(ai.status)}
                <span className={`px-2 py-1 rounded-full text-xs border ${getTypeColor(ai.type)}`}>
                  {ai.type}
                </span>
              </div>
            </div>

            {/* Description */}
            <p className="text-sm text-muted-foreground mb-4">{ai.description}</p>

            {/* Performance Metrics */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="flex items-center space-x-2">
                <Target className="w-4 h-4 text-blue-400" />
                <div>
                  <p className="text-xs text-muted-foreground">Accuracy</p>
                  <p className="font-semibold">{ai.performance.accuracy.toFixed(1)}%</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <TrendingUp className="w-4 h-4 text-green-400" />
                <div>
                  <p className="text-xs text-muted-foreground">Win Rate</p>
                  <p className="font-semibold">{ai.performance.winRate.toFixed(1)}%</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <DollarSign className="w-4 h-4 text-yellow-400" />
                <div>
                  <p className="text-xs text-muted-foreground">Avg Profit</p>
                  <p className="font-semibold">${ai.performance.avgProfit.toFixed(0)}</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Gauge className="w-4 h-4 text-purple-400" />
                <div>
                  <p className="text-xs text-muted-foreground">Efficiency</p>
                  <p className="font-semibold">{ai.performance.efficiency.toFixed(1)}%</p>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between pt-4 border-t border-white/10">
              <div className="flex items-center space-x-2">
                <TrendingUp className="w-4 h-4 text-cyan-400" />
                <span className="text-sm text-cyan-400">{ai.improvements} improvements</span>
              </div>
              <div className="text-xs text-muted-foreground">
                {new Date(ai.lastUpdated).toLocaleDateString()}
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}