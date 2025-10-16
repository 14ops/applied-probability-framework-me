function ControlPanel({ gameState, setGameState, strategies, onStart, onStop }) {
  return (
    <div style={{
      background: 'rgba(30, 41, 59, 0.8)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '1.5rem',
      border: '1px solid rgba(148, 163, 184, 0.2)',
    }}>
      <h2 style={{
        fontSize: '1.5rem',
        fontWeight: '600',
        marginBottom: '1.5rem',
        color: '#f1f5f9',
      }}>
        Control Panel
      </h2>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div>
          <label style={{
            display: 'block',
            fontSize: '0.875rem',
            fontWeight: '500',
            marginBottom: '0.5rem',
            color: '#cbd5e1',
          }}>
            Strategy
          </label>
          <select
            value={gameState.strategy}
            onChange={(e) => setGameState(prev => ({ ...prev, strategy: e.target.value }))}
            disabled={gameState.isRunning}
            style={{
              width: '100%',
              padding: '0.75rem',
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid rgba(148, 163, 184, 0.3)',
              borderRadius: '0.5rem',
              color: '#e2e8f0',
              fontSize: '1rem',
            }}
          >
            {strategies.map(strategy => (
              <option key={strategy.id} value={strategy.id}>
                {strategy.name} - {strategy.description}
              </option>
            ))}
          </select>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div>
            <label style={{
              display: 'block',
              fontSize: '0.875rem',
              fontWeight: '500',
              marginBottom: '0.5rem',
              color: '#cbd5e1',
            }}>
              Board Size
            </label>
            <input
              type="number"
              min="3"
              max="8"
              value={gameState.boardSize}
              onChange={(e) => setGameState(prev => ({ ...prev, boardSize: parseInt(e.target.value) }))}
              disabled={gameState.isRunning}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'rgba(15, 23, 42, 0.8)',
                border: '1px solid rgba(148, 163, 184, 0.3)',
                borderRadius: '0.5rem',
                color: '#e2e8f0',
                fontSize: '1rem',
              }}
            />
          </div>

          <div>
            <label style={{
              display: 'block',
              fontSize: '0.875rem',
              fontWeight: '500',
              marginBottom: '0.5rem',
              color: '#cbd5e1',
            }}>
              Mine Count
            </label>
            <input
              type="number"
              min="1"
              max={Math.floor(gameState.boardSize * gameState.boardSize / 2)}
              value={gameState.mineCount}
              onChange={(e) => setGameState(prev => ({ ...prev, mineCount: parseInt(e.target.value) }))}
              disabled={gameState.isRunning}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'rgba(15, 23, 42, 0.8)',
                border: '1px solid rgba(148, 163, 184, 0.3)',
                borderRadius: '0.5rem',
                color: '#e2e8f0',
                fontSize: '1rem',
              }}
            />
          </div>
        </div>

        <div>
          <label style={{
            display: 'block',
            fontSize: '0.875rem',
            fontWeight: '500',
            marginBottom: '0.5rem',
            color: '#cbd5e1',
          }}>
            Bet Amount
          </label>
          <input
            type="number"
            min="0.1"
            step="0.1"
            value={gameState.betAmount}
            onChange={(e) => setGameState(prev => ({ ...prev, betAmount: parseFloat(e.target.value) }))}
            disabled={gameState.isRunning}
            style={{
              width: '100%',
              padding: '0.75rem',
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid rgba(148, 163, 184, 0.3)',
              borderRadius: '0.5rem',
              color: '#e2e8f0',
              fontSize: '1rem',
            }}
          />
        </div>

        <button
          onClick={gameState.isRunning ? onStop : onStart}
          style={{
            width: '100%',
            padding: '1rem',
            marginTop: '1rem',
            background: gameState.isRunning
              ? 'linear-gradient(135deg, #ef4444, #dc2626)'
              : 'linear-gradient(135deg, #10b981, #059669)',
            border: 'none',
            borderRadius: '0.5rem',
            color: 'white',
            fontSize: '1rem',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'transform 0.2s',
          }}
          onMouseEnter={(e) => e.target.style.transform = 'scale(1.02)'}
          onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
        >
          {gameState.isRunning ? 'Stop Simulation' : 'Start Simulation'}
        </button>
      </div>
    </div>
  );
}

export default ControlPanel;
