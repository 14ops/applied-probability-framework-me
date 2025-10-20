'use client'

import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import { TrendingUp, Target, DollarSign, Gauge, Calendar } from 'lucide-react'
import { PerformanceMetric, aiData } from '../data/mockData'

interface PerformanceMetricsProps {
  performanceData: PerformanceMetric[]
  selectedAI: string | null
}

export default function PerformanceMetrics({ performanceData, selectedAI }: PerformanceMetricsProps) {
  // Filter data based on selected AI
  const filteredData = selectedAI 
    ? performanceData.filter(data => data.aiId === selectedAI)
    : performanceData

  // Prepare chart data
  const chartData = filteredData
    .reduce((acc, curr) => {
      const date = new Date(curr.timestamp).toLocaleDateString()
      const existing = acc.find(item => item.date === date)
      
      if (existing) {
        existing.accuracy = (existing.accuracy + curr.accuracy) / 2
        existing.winRate = (existing.winRate + curr.winRate) / 2
        existing.profit = (existing.profit + curr.profit) / 2
        existing.efficiency = (existing.efficiency + curr.efficiency) / 2
      } else {
        acc.push({
          date,
          accuracy: curr.accuracy,
          winRate: curr.winRate,
          profit: curr.profit,
          efficiency: curr.efficiency
        })
      }
      return acc
    }, [] as any[])
    .slice(-14) // Last 14 days

  // AI comparison data
  const comparisonData = aiData.map(ai => ({
    name: ai.name.split(' ')[0],
    accuracy: ai.performance.accuracy,
    winRate: ai.performance.winRate,
    profit: ai.performance.avgProfit,
    efficiency: ai.performance.efficiency,
    color: ai.color
  }))

  // Pie chart data for AI types
  const typeData = aiData.reduce((acc, ai) => {
    const existing = acc.find(item => item.name === ai.type)
    if (existing) {
      existing.value += 1
    } else {
      acc.push({ name: ai.type, value: 1, color: ai.color })
    }
    return acc
  }, [] as any[])

  const COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981']

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card-gradient rounded-xl p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Avg Accuracy</p>
              <p className="text-2xl font-bold text-blue-400">
                {selectedAI 
                  ? aiData.find(ai => ai.id === selectedAI)?.performance.accuracy.toFixed(1) || '0'
                  : (aiData.reduce((acc, ai) => acc + ai.performance.accuracy, 0) / aiData.length).toFixed(1)
                }%
              </p>
            </div>
            <Target className="w-8 h-8 text-blue-400" />
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
              <p className="text-sm text-muted-foreground">Win Rate</p>
              <p className="text-2xl font-bold text-green-400">
                {selectedAI 
                  ? aiData.find(ai => ai.id === selectedAI)?.performance.winRate.toFixed(1) || '0'
                  : (aiData.reduce((acc, ai) => acc + ai.performance.winRate, 0) / aiData.length).toFixed(1)
                }%
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-400" />
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
              <p className="text-sm text-muted-foreground">Avg Profit</p>
              <p className="text-2xl font-bold text-yellow-400">
                ${selectedAI 
                  ? aiData.find(ai => ai.id === selectedAI)?.performance.avgProfit.toFixed(0) || '0'
                  : (aiData.reduce((acc, ai) => acc + ai.performance.avgProfit, 0) / aiData.length).toFixed(0)
                }
              </p>
            </div>
            <DollarSign className="w-8 h-8 text-yellow-400" />
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
              <p className="text-sm text-muted-foreground">Efficiency</p>
              <p className="text-2xl font-bold text-purple-400">
                {selectedAI 
                  ? aiData.find(ai => ai.id === selectedAI)?.performance.efficiency.toFixed(1) || '0'
                  : (aiData.reduce((acc, ai) => acc + ai.performance.efficiency, 0) / aiData.length).toFixed(1)
                }%
              </p>
            </div>
            <Gauge className="w-8 h-8 text-purple-400" />
          </div>
        </motion.div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Trends */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card-gradient rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold">Performance Trends</h3>
            <Calendar className="w-6 h-6 text-blue-400" />
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                />
                <Line type="monotone" dataKey="accuracy" stroke="#3B82F6" strokeWidth={2} />
                <Line type="monotone" dataKey="winRate" stroke="#10B981" strokeWidth={2} />
                <Line type="monotone" dataKey="efficiency" stroke="#8B5CF6" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* AI Comparison */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card-gradient rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold">AI Comparison</h3>
            <TrendingUp className="w-6 h-6 text-green-400" />
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="accuracy" fill="#3B82F6" />
                <Bar dataKey="winRate" fill="#10B981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* AI Type Distribution */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card-gradient rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold">AI Type Distribution</h3>
            <Target className="w-6 h-6 text-purple-400" />
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={typeData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {typeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Profit Trends */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card-gradient rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold">Profit Trends</h3>
            <DollarSign className="w-6 h-6 text-yellow-400" />
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="profit" 
                  stroke="#F59E0B" 
                  strokeWidth={3}
                  dot={{ fill: '#F59E0B', strokeWidth: 2, r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>
    </div>
  )
}