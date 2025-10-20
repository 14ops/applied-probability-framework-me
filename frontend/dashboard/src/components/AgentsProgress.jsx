import React, { useEffect, useMemo, useState } from 'react';

function AgentsProgress() {
  const [agents, setAgents] = useState([]);
  const [improvements, setImprovements] = useState({ strategies: [], by_strategy: {}, summary: {} });

  useEffect(() => {
    let mounted = true;
    const fetchAgents = async () => {
      try {
        const res = await fetch('/api/agents');
        const json = await res.json();
        if (mounted) setAgents(json?.agents || []);
      } catch (_) {}
    };
    const fetchImprovements = async () => {
      try {
        const res = await fetch('/api/improvements');
        const json = await res.json();
        if (mounted) setImprovements(json);
      } catch (_) {}
    };
    fetchAgents();
    fetchImprovements();
    const id = setInterval(fetchImprovements, 5000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  const strategies = useMemo(() => improvements?.strategies || [], [improvements]);

  return (
    <div style={{
      background: 'rgba(30, 41, 59, 0.8)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '1.5rem',
      border: '1px solid rgba(148, 163, 184, 0.2)',
      color: '#e2e8f0',
    }}>
      <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1rem' }}>Agents & Improvements</h2>

      <div style={{ marginBottom: '1rem' }}>
        <h3 style={{ fontSize: '1rem', color: '#94a3b8', marginBottom: '0.5rem' }}>Agents</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {agents.map((a) => (
            <span key={a} style={{
              border: '1px solid rgba(148,163,184,0.3)',
              borderRadius: '9999px',
              padding: '0.25rem 0.5rem',
              background: 'rgba(15, 23, 42, 0.5)'
            }}>{a}</span>
          ))}
          {agents.length === 0 && (
            <span style={{ color: '#94a3b8' }}>No agents found.</span>
          )}
        </div>
      </div>

      <div>
        <h3 style={{ fontSize: '1rem', color: '#94a3b8', marginBottom: '0.5rem' }}>Strategy Summaries</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem' }}>
          {strategies.map((name) => {
            const s = improvements.summary?.[name] || { win_rate: 0, avg_payout: 0, games: 0 };
            return (
              <div key={name} style={{
                background: 'rgba(15, 23, 42, 0.6)',
                padding: '0.75rem',
                borderRadius: '0.5rem',
                border: '1px solid rgba(148,163,184,0.2)'
              }}>
                <div style={{ fontWeight: 600, marginBottom: '0.25rem', color: '#cbd5e1' }}>{name}</div>
                <div style={{ fontSize: '0.9rem', color: '#94a3b8' }}>Games: {s.games}</div>
                <div style={{ fontSize: '0.9rem', color: '#94a3b8' }}>Win Rate: {(s.win_rate * 100).toFixed(1)}%</div>
                <div style={{ fontSize: '0.9rem', color: '#94a3b8' }}>Avg Payout: {s.avg_payout?.toFixed?.(2) ?? '0.00'}</div>
              </div>
            );
          })}
          {strategies.length === 0 && (
            <div style={{ color: '#94a3b8' }}>No improvements recorded yet.</div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AgentsProgress;
