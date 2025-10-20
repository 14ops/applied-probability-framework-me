# AI Improvement Visualizer

A comprehensive visualization dashboard for monitoring all AI agents and their continuous improvements in the framework.

## Features

### 🤖 AI Overview
- **Complete AI Inventory**: View all 10+ AI agents including DRL agents, strategies, multi-agent systems, and specialized agents
- **Real-time Status**: Monitor active, training, optimizing, and idle states
- **Performance Metrics**: Track accuracy, win rates, average profit, and efficiency for each AI
- **Interactive Cards**: Click on any AI to filter other views and see detailed information
- **Type Classification**: Organized by AI type (DRL, Strategy, Multi-Agent, Adversarial, Behavioral)

### 📈 Improvement Tracking
- **Timeline View**: Chronological list of all improvements made to AI systems
- **Impact Analysis**: Categorized by high, medium, and low impact improvements
- **Type Distribution**: Track performance, algorithm, strategy, and optimization improvements
- **Before/After Metrics**: See exact performance gains from each improvement
- **Filterable by AI**: Focus on improvements for specific AI agents

### 📊 Performance Analytics
- **Trend Analysis**: 14-day performance trends with interactive charts
- **Comparative Analysis**: Side-by-side comparison of all AI agents
- **Distribution Charts**: Visual breakdown of AI types and their performance
- **Profit Tracking**: Detailed profit trend analysis over time
- **Key Metrics Dashboard**: Real-time overview of critical performance indicators

## AI Agents Included

1. **Deep RL Agent** - Q-learning based agent with neural network approximation
2. **Takeshi (Aggressive)** - High-risk, high-reward trading strategy
3. **Lelouch (Mastermind)** - Calculated strategic approach with game theory
4. **Kazuya (Conservative)** - Risk-averse capital preservation strategy
5. **Senku (Analytical)** - Data-driven scientific approach
6. **Multi-Agent System** - Coordinated system of 10 collaborative agents
7. **Adversarial Agent** - Counter-adversarial attack detection system
8. **Behavioral Agent** - Prospect theory with loss aversion modeling
9. **Rintaro Okabe Strategy** - Time-series analysis with predictive modeling
10. **Advanced Strategy Engine** - Meta-strategy combining multiple approaches

## Technology Stack

- **Frontend**: Next.js 14 with React 18
- **Styling**: Tailwind CSS with custom design system
- **Animations**: Framer Motion for smooth interactions
- **Charts**: Recharts for data visualization
- **Icons**: Lucide React for consistent iconography
- **TypeScript**: Full type safety throughout

## Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```

3. **Open your browser**:
   Navigate to `http://localhost:3000` (or the port shown in terminal)

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Usage

### Navigation
- **AI Overview**: Main dashboard showing all AI agents with their current status and performance
- **Improvements**: Track all improvements made to AI systems over time
- **Performance**: Detailed analytics and charts for performance monitoring

### Filtering
- Click on any AI card in the Overview to filter the Improvements and Performance views
- Use the navigation tabs to switch between different visualization modes

### Real-time Updates
The dashboard simulates real-time data updates showing:
- Current AI status changes
- New improvements being applied
- Performance metric fluctuations
- Trend analysis over time

## Data Structure

The visualization uses structured data including:
- **AI Agents**: Status, performance metrics, improvement counts, versions
- **Improvements**: Timeline, impact levels, before/after metrics
- **Performance Data**: Historical trends, comparative analysis

## Customization

### Adding New AIs
Edit `app/data/mockData.ts` to add new AI agents:

```typescript
{
  id: 'new-ai',
  name: 'New AI Agent',
  type: 'Strategy',
  status: 'active',
  description: 'Description of the new AI',
  performance: { accuracy: 85, winRate: 70, avgProfit: 150, efficiency: 88 },
  improvements: 15,
  lastUpdated: new Date().toISOString(),
  color: '#3b82f6',
  version: 'v1.0.0'
}
```

### Styling
The app uses a custom design system with:
- Dark theme optimized for data visualization
- Gradient backgrounds and glass-morphism effects
- Consistent color coding for different AI types
- Responsive design for all screen sizes

## Architecture

```
app/
├── components/           # React components
│   ├── AIOverview.tsx   # Main AI dashboard
│   ├── ImprovementTracker.tsx  # Improvements timeline
│   ├── PerformanceMetrics.tsx  # Analytics charts
│   └── Navigation.tsx   # Tab navigation
├── data/
│   └── mockData.ts      # Data structures and mock data
├── globals.css          # Global styles and design system
├── layout.tsx           # Root layout
└── page.tsx             # Main application entry
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the AI Strategy Framework and follows the same licensing terms.

---

**Built with ❤️ for AI researchers and developers**