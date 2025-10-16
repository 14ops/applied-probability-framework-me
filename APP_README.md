# Mines Strategy Framework - Web Application

A modern web application that brings the Applied Probability and Automation Framework to life with real-time visualization and data persistence.

## What Was Built

A complete React-based web application featuring:

- **Interactive Control Panel** - Configure game parameters, select strategies, and control simulations
- **Real-time Game Board** - Visual representation of the Mines game with animated cell reveals
- **Live Statistics Dashboard** - Track performance metrics including bankroll, win rate, and profit
- **Strategy Information** - Detailed breakdown of each strategy's characteristics and risk profile
- **Supabase Integration** - All simulation data is persisted to the database for analysis

## Features

### Four Strategic Approaches

1. **Takeshi (Aggressive)** - High-risk, high-reward strategy targeting maximum short-term gains
2. **Lelouch (Calculated)** - Probability-based approach with adaptive decision-making
3. **Kazuya (Conservative)** - Capital preservation focus with ultra-low risk tolerance
4. **Senku (Analytical)** - Data-driven optimization using machine learning principles

### Real-time Visualization

- Live board state updates showing safe cells and mines
- Animated statistics tracking performance metrics
- Color-coded visual feedback for wins and losses
- Responsive design that works on all screen sizes

### Data Persistence

- All simulations are automatically saved to Supabase
- Track historical performance across different strategies
- Analyze long-term trends and patterns
- Export data for further analysis

## Technology Stack

- **React 19** - Modern UI with hooks and functional components
- **Vite** - Lightning-fast build tool and dev server
- **Supabase** - PostgreSQL database with real-time capabilities
- **Vanilla CSS** - Custom styling with gradients and animations

## Getting Started

The application is ready to run. The dev server starts automatically.

## Database Schema

### simulations table
- `id` - Unique simulation identifier
- `strategy` - Strategy used (takeshi, lelouch, kazuya, senku)
- `board_size` - Game board dimensions
- `mine_count` - Number of mines
- `initial_bankroll` - Starting amount
- `final_bankroll` - Ending amount
- `total_rounds` - Rounds completed
- `status` - Simulation status (pending, running, completed)
- `created_at` - Start timestamp
- `updated_at` - Last update timestamp

## How It Works

1. Select a strategy from the dropdown menu
2. Configure board size, mine count, and bet amount
3. Click "Start Simulation" to begin
4. Watch the board animate as the strategy plays
5. Monitor statistics in real-time
6. All data is automatically saved to Supabase

## Strategy Details

Each strategy has unique characteristics optimized for different scenarios:

- **Risk Levels**: From conservative to very high
- **Position Sizing**: Different bankroll management approaches
- **Decision Logic**: Ranging from rule-based to ML-powered
- **Best Use Cases**: Specific market conditions where each excels

## Future Enhancements

- Historical performance charts
- Strategy comparison tools
- Custom strategy builder
- Multi-simulation tournament mode
- Advanced analytics dashboard
- Real-time collaboration features

## Project Structure

```
/src
  /components
    ControlPanel.jsx    # Game configuration controls
    GameBoard.jsx       # Visual board representation
    Statistics.jsx      # Performance metrics display
    StrategyInfo.jsx    # Strategy details panel
  App.jsx              # Main application component
  main.jsx            # Application entry point
  index.css           # Global styles
```

## Notes

This application demonstrates the theoretical framework described in the extensive documentation. It provides an interactive way to explore different betting strategies and understand their risk-reward characteristics through visual simulation.
