function GameBoard({ board }) {
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
        Game Board
      </h2>

      <div style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${board.length}, 1fr)`,
        gap: '0.5rem',
        maxWidth: '400px',
        margin: '0 auto',
      }}>
        {board.map((row, i) =>
          row.map((cell, j) => (
            <div
              key={`${i}-${j}`}
              style={{
                aspectRatio: '1',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: cell.revealed
                  ? cell.isMine
                    ? 'linear-gradient(135deg, #ef4444, #dc2626)'
                    : 'linear-gradient(135deg, #10b981, #059669)'
                  : 'rgba(15, 23, 42, 0.8)',
                border: '1px solid rgba(148, 163, 184, 0.3)',
                borderRadius: '0.5rem',
                fontSize: '1.25rem',
                fontWeight: '700',
                color: 'white',
                transition: 'all 0.3s',
                cursor: 'default',
              }}
            >
              {cell.revealed && (cell.isMine ? '💥' : '💎')}
            </div>
          ))
        )}
      </div>

      <div style={{
        marginTop: '1.5rem',
        padding: '1rem',
        background: 'rgba(15, 23, 42, 0.6)',
        borderRadius: '0.5rem',
        display: 'flex',
        justifyContent: 'space-around',
        fontSize: '0.875rem',
        color: '#cbd5e1',
      }}>
        <div>
          <span style={{ marginRight: '0.5rem' }}>💎</span>
          Safe Cell
        </div>
        <div>
          <span style={{ marginRight: '0.5rem' }}>💥</span>
          Mine
        </div>
      </div>
    </div>
  );
}

export default GameBoard;
