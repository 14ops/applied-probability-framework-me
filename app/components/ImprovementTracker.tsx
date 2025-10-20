'use client'

import { motion } from 'framer-motion'
import { TrendingUp, Clock, Zap, Settings, Code, Target, ArrowUp, ArrowDown } from 'lucide-react'
import { Improvement } from '../data/mockData'

interface ImprovementTrackerProps {
  improvementData: Improvement[]
  selectedAI: string | null
}

export default function ImprovementTracker({ improvementData, selectedAI }: ImprovementTrackerProps) {
  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'text-red-400 bg-red-500/20 border-red-500/30'
      case 'medium': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30'
      case 'low': return 'text-green-400 bg-green-500/20 border-green-500/30'
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'performance': return <Target className="w-4 h-4" />
      case 'algorithm': return <Code className="w-4 h-4" />
      case 'strategy': return <Zap className="w-4 h-4" />
      case 'optimization': return <Settings className="w-4 h-4" />
      default: return <TrendingUp className="w-4 h-4" />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'performance': return 'text-blue-400 bg-blue-500/20'
      case 'algorithm': return 'text-purple-400 bg-purple-500/20'
      case 'strategy': return 'text-orange-400 bg-orange-500/20'
      case 'optimization': return 'text-green-400 bg-green-500/20'
      default: return 'text-gray-400 bg-gray-500/20'
    }
  }

  const filteredImprovements = selectedAI 
    ? improvementData.filter(imp => imp.aiId === selectedAI)
    : improvementData

  const sortedImprovements = [...filteredImprovements].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  )

  const impactStats = {
    high: filteredImprovements.filter(imp => imp.impact === 'high').length,
    medium: filteredImprovements.filter(imp => imp.impact === 'medium').length,
    low: filteredImprovements.filter(imp => imp.impact === 'low').length,
  }

  const typeStats = {
    performance: filteredImprovements.filter(imp => imp.type === 'performance').length,
    algorithm: filteredImprovements.filter(imp => imp.type === 'algorithm').length,
    strategy: filteredImprovements.filter(imp => imp.type === 'strategy').length,
    optimization: filteredImprovements.filter(imp => imp.type === 'optimization').length,
  }

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card-gradient rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Impact Distribution</h3>
            <TrendingUp className="w-6 h-6 text-blue-400" />
          </div>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-red-400">High Impact</span>
              <span className="font-semibold">{impactStats.high}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-yellow-400">Medium Impact</span>
              <span className="font-semibold">{impactStats.medium}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-green-400">Low Impact</span>
              <span className="font-semibold">{impactStats.low}</span>
            </div>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card-gradient rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Improvement Types</h3>
            <Settings className="w-6 h-6 text-purple-400" />
          </div>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-blue-400">Performance</span>
              <span className="font-semibold">{typeStats.performance}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-purple-400">Algorithm</span>
              <span className="font-semibold">{typeStats.algorithm}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-orange-400">Strategy</span>
              <span className="font-semibold">{typeStats.strategy}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-green-400">Optimization</span>
              <span className="font-semibold">{typeStats.optimization}</span>
            </div>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card-gradient rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Average Improvement</h3>
            <ArrowUp className="w-6 h-6 text-green-400" />
          </div>
          <div className="text-center">
            <p className="text-3xl font-bold text-green-400">
              +{(filteredImprovements.reduce((acc, imp) => acc + imp.metrics.improvement, 0) / filteredImprovements.length || 0).toFixed(1)}%
            </p>
            <p className="text-sm text-muted-foreground mt-1">Per improvement</p>
          </div>
        </motion.div>
      </div>

      {/* Improvements Timeline */}
      <div className="card-gradient rounded-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">Recent Improvements</h2>
          {selectedAI && (
            <span className="text-sm text-muted-foreground">
              Filtered for selected AI
            </span>
          )}
        </div>

        <div className="space-y-4">
          {sortedImprovements.map((improvement, index) => (
            <motion.div
              key={improvement.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="border border-white/10 rounded-lg p-4 hover:border-white/20 transition-colors"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${getTypeColor(improvement.type)}`}>
                    {getTypeIcon(improvement.type)}
                  </div>
                  <div>
                    <h3 className="font-semibold">{improvement.description}</h3>
                    <p className="text-sm text-muted-foreground">
                      AI: {improvement.aiId.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded-full text-xs border ${getImpactColor(improvement.impact)}`}>
                    {improvement.impact} impact
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {new Date(improvement.timestamp).toLocaleDateString()}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 mt-4 p-3 bg-black/20 rounded-lg">
                <div className="text-center">
                  <p className="text-xs text-muted-foreground">Before</p>
                  <p className="font-semibold text-red-400">{improvement.metrics.before.toFixed(1)}%</p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-muted-foreground">After</p>
                  <p className="font-semibold text-green-400">{improvement.metrics.after.toFixed(1)}%</p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-muted-foreground">Improvement</p>
                  <div className="flex items-center justify-center space-x-1">
                    <ArrowUp className="w-3 h-3 text-green-400" />
                    <p className="font-semibold text-green-400">+{improvement.metrics.improvement.toFixed(1)}%</p>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {sortedImprovements.length === 0 && (
          <div className="text-center py-12">
            <Clock className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">
              {selectedAI ? 'No improvements found for selected AI' : 'No improvements to display'}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}