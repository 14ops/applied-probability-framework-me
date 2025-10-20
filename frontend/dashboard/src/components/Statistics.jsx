import { useEffect, useState } from 'react';

function Statistics({ stats, gameState }) {
  const [progress, setProgress] = useState({ summary: {}, sessions: [] });

  useEffect(() => {
    let mounted = true;
    const fetchProgress = async () => {
      try {
        const res = await fetch('/api/progress');
        const json = await res.json();
        if (mounted) setProgress(json);
      } catch (_) {}
    };
    fetchProgress();
    const id = setInterval(fetchProgress, 5000);
    return () => { mounted = false; clearInterval(id); };
  }, []);
  const StatCard = ({ label, value, color }) => (
    <div style={{
      background: 'rgba(15, 23, 42, 0.6)',
      padding: '1rem',
      borderRadius: '0.5rem',
      border: `1px solid ${color}40`,
    }}>
      <div style={{
        fontSize: '0.875rem',
        color: '#94a3b8',
        marginBottom: '0.25rem',
      }}>
        {label}
      </div>
      <div style={{
        fontSize: '1.5rem',
        fontWeight: '700',
        color: color,
      }}>
        {value}
      </div>
    </div>
  );

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
        Statistics
      </h2>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        <StatCard
          label="Current Round"
          value={gameState.currentRound}
          color="#60a5fa"
        />
        <StatCard
          label="Bankroll"
          value={`$${gameState.bankroll.toFixed(2)}`}
          color="#10b981"
        />
        <StatCard
          label="Total Profit"
          value={`$${stats.totalProfit.toFixed(2)}`}
          color={stats.totalProfit >= 0 ? '#10b981' : '#ef4444'}
        />
        <StatCard
          label="Win Rate"
          value={`${stats.winRate.toFixed(1)}%`}
          color="#a78bfa"
        />
        <StatCard
          label="Wins"
          value={gameState.wins}
          color="#34d399"
        />
        <StatCard
          label="Losses"
          value={gameState.losses}
          color="#f87171"
        />
        <StatCard
          label="Sessions Logged"
          value={`${progress?.summary?.total_sessions || 0}`}
          color="#38bdf8"
        />
        <StatCard
          label="Log Win Rate"
          value={`${progress?.summary?.win_rate ? (progress.summary.win_rate * 100).toFixed(1) : '0.0'}%`}
          color="#f59e0b"
        />
      </div>
    </div>
  );
}

export default Statistics;
