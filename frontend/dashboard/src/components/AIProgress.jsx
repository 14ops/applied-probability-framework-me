import { useEffect, useMemo, useState } from 'react';

function MetricPill({ label, value, color }) {
  return (
    <div style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: '0.5rem',
      background: 'rgba(15, 23, 42, 0.6)',
      border: `1px solid ${color}40`,
      borderRadius: '9999px',
      padding: '0.25rem 0.6rem',
      color,
      fontSize: '0.8rem',
      fontWeight: 600,
    }}>
      <span style={{ opacity: 0.8 }}>{label}</span>
      <span style={{ color: '#e2e8f0' }}>{value}</span>
    </div>
  );
}

function Timeline({ history, color }) {
  return (
    <div style={{ position: 'relative', paddingLeft: '1rem' }}>
      <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 2, background: `${color}40`, borderRadius: 1 }} />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {history.map((h, idx) => (
          <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: 10, height: 10, borderRadius: 6, background: color }} />
            <div style={{ color: '#e2e8f0' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>{h.label}</div>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{new Date(h.at).toLocaleString()}</div>
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem' }}>
              {'ev' in h && <MetricPill label="EV" value={h.ev.toFixed(2)} color={color} />}
              {'winRate' in h && <MetricPill label="WR" value={`${(h.winRate * 100).toFixed(1)}%`} color={color} />} 
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AgentCard({ agent }) {
  const color = agent.color || '#60a5fa';
  const wr = agent.metrics?.winRate ?? 0;
  const avg = agent.metrics?.avgPayout ?? 0;
  const ev = agent.metrics?.ev ?? 0;

  return (
    <div style={{
      background: 'rgba(30, 41, 59, 0.8)',
      borderRadius: '1rem',
      padding: '1rem',
      border: '1px solid rgba(148, 163, 184, 0.2)'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
        <div style={{ width: 6, height: 24, borderRadius: 3, background: color }} />
        <div>
          <div style={{ color: '#f1f5f9', fontWeight: 700 }}>{agent.name}</div>
          <div style={{ color: '#94a3b8', fontSize: '0.85rem' }}>{agent.type}</div>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <MetricPill label="WR" value={`${(wr * 100).toFixed(1)}%`} color={color} />
          <MetricPill label="Avg" value={avg.toFixed(2)} color={color} />
          <MetricPill label="EV" value={ev.toFixed(2)} color={color} />
        </div>
      </div>

      {agent.history?.length > 0 && (
        <div style={{ marginTop: '0.5rem' }}>
          <Timeline history={agent.history} color={color} />
        </div>
      )}
    </div>
  );
}

export default function AIProgress() {
  const [data, setData] = useState(null);
  const [query, setQuery] = useState('');

  useEffect(() => {
    let isMounted = true;
    fetch('/ai_progress.json')
      .then((r) => r.json())
      .then((j) => { if (isMounted) setData(j); })
      .catch((e) => console.error('Failed to load ai_progress.json', e));
    return () => { isMounted = false; };
  }, []);

  const filteredAgents = useMemo(() => {
    if (!data?.agents) return [];
    if (!query) return data.agents;
    const q = query.toLowerCase();
    return data.agents.filter(a =>
      a.name.toLowerCase().includes(q) ||
      a.type.toLowerCase().includes(q) ||
      (a.history || []).some(h => h.label.toLowerCase().includes(q))
    );
  }, [data, query]);

  if (!data) {
    return (
      <div style={{
        background: 'rgba(30, 41, 59, 0.8)',
        borderRadius: '1rem',
        padding: '1.5rem',
        border: '1px solid rgba(148, 163, 184, 0.2)',
        color: '#94a3b8'
      }}>
        Loading AI progress...
      </div>
    );
  }

  return (
    <div style={{
      background: 'rgba(30, 41, 59, 0.8)',
      borderRadius: '1rem',
      padding: '1.5rem',
      border: '1px solid rgba(148, 163, 184, 0.2)'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem', gap: '0.75rem' }}>
        <h2 style={{ color: '#f1f5f9', fontSize: '1.25rem', fontWeight: 700, margin: 0 }}>AI Progress</h2>
        <div style={{ marginLeft: 'auto' }}>
          <input
            placeholder="Search agents or improvements..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{
              width: '280px',
              padding: '0.6rem 0.8rem',
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid rgba(148, 163, 184, 0.3)',
              borderRadius: '0.5rem',
              color: '#e2e8f0',
            }}
          />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1rem' }}>
        {filteredAgents.map(agent => (
          <AgentCard key={agent.id} agent={agent} />
        ))}
      </div>

      <div style={{ marginTop: '1rem', color: '#94a3b8', fontSize: '0.8rem' }}>
        Updated {new Date(data.updatedAt).toLocaleString()}
      </div>
    </div>
  );
}
