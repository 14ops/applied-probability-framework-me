'use client'

import { motion } from 'framer-motion'
import { BarChart3, TrendingUp, Users } from 'lucide-react'

interface NavigationProps {
  activeView: string
  setActiveView: (view: string) => void
}

export default function Navigation({ activeView, setActiveView }: NavigationProps) {
  const navItems = [
    { id: 'overview', label: 'AI Overview', icon: Users },
    { id: 'improvements', label: 'Improvements', icon: TrendingUp },
    { id: 'performance', label: 'Performance', icon: BarChart3 }
  ]

  return (
    <nav className="flex justify-center mb-8">
      <div className="card-gradient rounded-full p-2 flex space-x-2">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = activeView === item.id
          
          return (
            <motion.button
              key={item.id}
              onClick={() => setActiveView(item.id)}
              className={`relative px-6 py-3 rounded-full flex items-center space-x-2 transition-all duration-300 ${
                isActive 
                  ? 'text-white' 
                  : 'text-muted-foreground hover:text-white'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {isActive && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full"
                  initial={false}
                  transition={{ type: "spring", stiffness: 500, damping: 30 }}
                />
              )}
              <Icon className="w-5 h-5 relative z-10" />
              <span className="relative z-10 font-medium">{item.label}</span>
            </motion.button>
          )
        })}
      </div>
    </nav>
  )
}