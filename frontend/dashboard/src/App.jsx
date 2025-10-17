import { useState, useEffect } from 'react';
import { createClient } from '@supabase/supabase-js';
import ControlPanel from './components/ControlPanel';
import GameBoard from './components/GameBoard';
import Statistics from './components/Statistics';
import StrategyInfo from './components/StrategyInfo';

const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_SUPABASE_ANON_KEY
);

function App() {
  const [gameState, setGameState] = useState({
    boardSize: 5,
    mineCount: 3,
    betAmount: 1.0,
    strategy: 'lelouch',
    isRunning: false,
    currentRound: 0,
    bankroll: 1000,
    wins: 0,
    losses: 0,
    board: [],
  });

  const [stats, setStats] = useState({
    totalProfit: 0,
    winRate: 0,
    avgProfit: 0,
    maxDrawdown: 0,
  });

  const strategies = [
    { id: 'takeshi', name: 'Takeshi', description: 'Aggressive risk-taker', color: '#ef4444' },
    { id: 'lelouch', name: 'Lelouch', description: 'Calculated mastermind', color: '#8b5cf6' },
    { id: 'kazuya', name: 'Kazuya', description: 'Conservative survivor', color: '#10b981' },
    { id: 'senku', name: 'Senku', description: 'Analytical scientist', color: '#3b82f6' },
  ];

  useEffect(() => {
    initializeBoard();
  }, [gameState.boardSize]);

  const initializeBoard = () => {
    const newBoard = [];
    for (let i = 0; i < gameState.boardSize; i++) {
      const row = [];
      for (let j = 0; j < gameState.boardSize; j++) {
        row.push({ revealed: false, isMine: false });
      }
      newBoard.push(row);
    }
    setGameState(prev => ({ ...prev, board: newBoard }));
  };

  const startSimulation = async () => {
    setGameState(prev => ({ ...prev, isRunning: true }));

    try {
      const { data, error } = await supabase
        .from('simulations')
        .insert([
          {
            strategy: gameState.strategy,
            board_size: gameState.boardSize,
            mine_count: gameState.mineCount,
            initial_bankroll: gameState.bankroll,
            status: 'running',
          },
        ])
        .select()
        .maybeSingle();

      if (error) throw error;

      runSimulationLoop(data?.id);
    } catch (error) {
      console.error('Error starting simulation:', error);
    }
  };

  const stopSimulation = () => {
    setGameState(prev => ({ ...prev, isRunning: false }));
  };

  const runSimulationLoop = async (simulationId) => {
    let currentBankroll = gameState.bankroll;
    let rounds = 0;
    let wins = 0;

    while (gameState.isRunning && rounds < 100) {
      await new Promise(resolve => setTimeout(resolve, 500));

      const won = Math.random() > 0.5;
      const profit = won ? gameState.betAmount * 1.5 : -gameState.betAmount;

      currentBankroll += profit;
      rounds++;
      if (won) wins++;

      setGameState(prev => ({
        ...prev,
        currentRound: rounds,
        bankroll: currentBankroll,
        wins: won ? prev.wins + 1 : prev.wins,
        losses: !won ? prev.losses + 1 : prev.losses,
      }));

      setStats({
        totalProfit: currentBankroll - 1000,
        winRate: (wins / rounds) * 100,
        avgProfit: (currentBankroll - 1000) / rounds,
        maxDrawdown: Math.min(0, currentBankroll - 1000),
      });

      simulateBoard(won);
    }

    if (simulationId) {
      await supabase
        .from('simulations')
        .update({
          status: 'completed',
          final_bankroll: currentBankroll,
          total_rounds: rounds,
        })
        .eq('id', simulationId);
    }

    setGameState(prev => ({ ...prev, isRunning: false }));
  };

  const simulateBoard = (won) => {
    const newBoard = gameState.board.map(row =>
      row.map(cell => ({
        ...cell,
        revealed: Math.random() > 0.7,
        isMine: !won && Math.random() < 0.1,
      }))
    );
    setGameState(prev => ({ ...prev, board: newBoard }));
  };

  return (
    <div style={{
      minHeight: '100vh',
      padding: '2rem',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)'
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        <header style={{
          marginBottom: '2rem',
          textAlign: 'center',
        }}>
          <h1 style={{
            fontSize: '2.5rem',
            fontWeight: '700',
            marginBottom: '0.5rem',
            background: 'linear-gradient(135deg, #60a5fa, #a78bfa)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Mines Strategy Framework
          </h1>
          <p style={{
            color: '#94a3b8',
            fontSize: '1.1rem',
          }}>
            Advanced Probability and Automation System
          </p>
        </header>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '1.5rem',
          marginBottom: '1.5rem',
        }}>
          <ControlPanel
            gameState={gameState}
            setGameState={setGameState}
            strategies={strategies}
            onStart={startSimulation}
            onStop={stopSimulation}
          />
          <Statistics stats={stats} gameState={gameState} />
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
          gap: '1.5rem',
        }}>
          <GameBoard board={gameState.board} />
          <StrategyInfo
            strategy={strategies.find(s => s.id === gameState.strategy)}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
