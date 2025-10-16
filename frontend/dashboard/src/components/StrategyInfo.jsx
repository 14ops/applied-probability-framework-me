function StrategyInfo({ strategy }) {
  const strategyDetails = {
    takeshi: {
      characteristics: [
        'High-risk, high-reward approach',
        'Targets 40-60% cell revelation',
        'Aggressive position sizing',
        'Maximizes short-term gains',
      ],
      bestFor: 'High volatility environments with promotional opportunities',
      riskLevel: 'Very High',
    },
    lelouch: {
      characteristics: [
        'Comprehensive probability analysis',
        'Adaptive decision-making',
        'Center-out cell revelation',
        'Modified Kelly Criterion sizing',
      ],
      bestFor: 'Stable conditions requiring careful analysis',
      riskLevel: 'Medium',
    },
    kazuya: {
      characteristics: [
        'Capital preservation focus',
        'Conservative probability thresholds (85%+)',
        'Ultra-low position sizing (1-2%)',
        'Multiple risk protection layers',
      ],
      bestFor: 'Risk-averse players prioritizing stability',
      riskLevel: 'Low',
    },
    senku: {
      characteristics: [
        'Data-driven optimization',
        'Machine learning principles',
        'Multi-objective analysis',
        'Continuous learning and adaptation',
      ],
      bestFor: 'Complex environments with historical data',
      riskLevel: 'Medium-High',
    },
  };

  const details = strategyDetails[strategy?.id] || strategyDetails.lelouch;

  return (
    <div style={{
      background: 'rgba(30, 41, 59, 0.8)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '1.5rem',
      border: '1px solid rgba(148, 163, 184, 0.2)',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        marginBottom: '1.5rem',
      }}>
        <div
          style={{
            width: '4px',
            height: '2rem',
            background: strategy?.color || '#8b5cf6',
            marginRight: '1rem',
            borderRadius: '2px',
          }}
        />
        <div>
          <h2 style={{
            fontSize: '1.5rem',
            fontWeight: '600',
            color: '#f1f5f9',
          }}>
            {strategy?.name || 'Strategy'} Strategy
          </h2>
          <p style={{
            fontSize: '0.875rem',
            color: '#94a3b8',
          }}>
            {strategy?.description}
          </p>
        </div>
      </div>

      <div style={{ marginBottom: '1.5rem' }}>
        <h3 style={{
          fontSize: '1rem',
          fontWeight: '600',
          marginBottom: '0.75rem',
          color: '#cbd5e1',
        }}>
          Key Characteristics
        </h3>
        <ul style={{
          listStyle: 'none',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.5rem',
        }}>
          {details.characteristics.map((char, i) => (
            <li
              key={i}
              style={{
                padding: '0.75rem',
                background: 'rgba(15, 23, 42, 0.6)',
                borderRadius: '0.5rem',
                fontSize: '0.875rem',
                color: '#e2e8f0',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <span style={{
                width: '6px',
                height: '6px',
                background: strategy?.color || '#8b5cf6',
                borderRadius: '50%',
                marginRight: '0.75rem',
              }} />
              {char}
            </li>
          ))}
        </ul>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr',
        gap: '1rem',
      }}>
        <div style={{
          padding: '1rem',
          background: 'rgba(15, 23, 42, 0.6)',
          borderRadius: '0.5rem',
        }}>
          <div style={{
            fontSize: '0.75rem',
            color: '#94a3b8',
            marginBottom: '0.25rem',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}>
            Risk Level
          </div>
          <div style={{
            fontSize: '1.125rem',
            fontWeight: '600',
            color: strategy?.color || '#8b5cf6',
          }}>
            {details.riskLevel}
          </div>
        </div>

        <div style={{
          padding: '1rem',
          background: 'rgba(15, 23, 42, 0.6)',
          borderRadius: '0.5rem',
        }}>
          <div style={{
            fontSize: '0.75rem',
            color: '#94a3b8',
            marginBottom: '0.25rem',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}>
            Best For
          </div>
          <div style={{
            fontSize: '0.875rem',
            color: '#e2e8f0',
            lineHeight: '1.5',
          }}>
            {details.bestFor}
          </div>
        </div>
      </div>
    </div>
  );
}

export default StrategyInfo;
