// src/components/PredictionCard.jsx
// Shows predicted RUL + health status badge + confidence bar

import React from 'react';

const STATUS_CONFIG = {
  healthy:  { label: 'HEALTHY',  cls: 'badge-healthy',  glow: '', icon: '●' },
  warning:  { label: 'WARNING',  cls: 'badge-warning',  glow: '', icon: '▲' },
  critical: { label: 'CRITICAL', cls: 'badge-critical', glow: 'glow-critical', icon: '■' },
};

export default function PredictionCard({ prediction, loading }) {
  if (loading) {
    return (
      <div className="card animate-pulse">
        <p className="section-title">Prediction Result</p>
        <div className="h-24 bg-border rounded-lg mb-4" />
        <div className="h-6 bg-border rounded w-1/2" />
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="card flex flex-col items-center justify-center min-h-[180px]">
        <p className="section-title">Prediction Result</p>
        <p className="text-muted text-sm mt-2">
          Select an engine and cycle, then click Predict.
        </p>
      </div>
    );
  }

  const color = prediction.health_color || 'healthy';
  const cfg = STATUS_CONFIG[color] || STATUS_CONFIG.healthy;
  const confPct = Math.round((prediction.confidence || 0) * 100);

  return (
    <div className={`card animate-fade-in ${cfg.glow}`}>
      <p className="section-title">Prediction Result</p>

      {/* RUL number */}
      <div className="flex items-end gap-3 mb-4">
        <span className="text-6xl font-bold text-text font-mono">
          {Math.round(prediction.predicted_rul)}
        </span>
        <div className="pb-2">
          <p className="text-muted text-sm leading-none">cycles</p>
          <p className="text-muted text-xs mt-1">Remaining Useful Life</p>
        </div>
      </div>

      {/* Health badge */}
      <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-semibold ${cfg.cls} mb-4`}>
        <span>{cfg.icon}</span>
        {cfg.label}
      </div>

      {/* Confidence bar */}
      <div className="mt-3">
        <div className="flex justify-between text-xs text-muted mb-1.5">
          <span>Confidence</span>
          <span className="font-mono">{confPct}%</span>
        </div>
        <div className="w-full bg-bg rounded-full h-1.5">
          <div
            className="h-1.5 rounded-full transition-all duration-500"
            style={{
              width: `${confPct}%`,
              background: color === 'healthy' ? '#10B981'
                : color === 'warning' ? '#F59E0B' : '#EF4444',
            }}
          />
        </div>
      </div>

      {/* Meta */}
      <div className="mt-4 pt-4 border-t border-border flex gap-6 text-xs text-muted">
        <span>Engine <span className="text-text font-mono">{prediction.engine_id}</span></span>
        <span>Cycle <span className="text-text font-mono">{prediction.cycle}</span></span>
        <span>Model <span className="text-text">{prediction.model_used}</span></span>
      </div>
    </div>
  );
}
