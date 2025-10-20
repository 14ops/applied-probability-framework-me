'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import AIOverview from './components/AIOverview'
import ImprovementTracker from './components/ImprovementTracker'
import PerformanceMetrics from './components/PerformanceMetrics'
import Navigation from './components/Navigation'
import { aiData, improvementData, performanceData } from './data/mockData'

export default function Home() {
  const [activeView, setActiveView] = useState('overview')
  const [selectedAI, setSelectedAI] = useState<string | null>(null)

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.header 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 via-purple-500 to-cyan-400 bg-clip-text text-transparent">
            AI Improvement Visualizer
          </h1>
          <p className="text-xl text-muted-foreground">
            Monitor and visualize all AI agents and their continuous improvements
          </p>
        </motion.header>

        {/* Navigation */}
        <Navigation activeView={activeView} setActiveView={setActiveView} />

        {/* Main Content */}
        <motion.main
          key={activeView}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
          className="mt-8"
        >
          {activeView === 'overview' && (
            <AIOverview 
              aiData={aiData} 
              selectedAI={selectedAI} 
              setSelectedAI={setSelectedAI} 
            />
          )}
          {activeView === 'improvements' && (
            <ImprovementTracker 
              improvementData={improvementData}
              selectedAI={selectedAI}
            />
          )}
          {activeView === 'performance' && (
            <PerformanceMetrics 
              performanceData={performanceData}
              selectedAI={selectedAI}
            />
          )}
        </motion.main>
      </div>
    </div>
  )
}