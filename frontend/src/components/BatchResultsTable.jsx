// src/components/BatchResultsTable.jsx
// Shows results from CSV batch upload

import React from 'react';

const HEALTH_STYLE = {
  healthy:  'badge-healthy',
  warning:  'badge-warning',
  critical: 'badge-critical',
};

export default function BatchResultsTable({ results }) {
  if (!results?.length) return null;

  return (
    <div className="card">
      <p className="section-title">Batch Upload Results — {results.length} engines</p>
      <div className="overflow-x-auto max-h-80 overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-surface">
            <tr className="border-b border-border">
              {['Engine', 'Cycles', 'Pred. RUL', 'Health', 'Confidence'].map((h) => (
                <th key={h} className="px-3 py-2 text-left text-xs font-semibold text-muted uppercase tracking-wider">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.engine_id} className="border-b border-border/40 hover:bg-bg/50">
                <td className="px-3 py-2 font-mono text-text">{r.engine_id}</td>
                <td className="px-3 py-2 font-mono text-text-secondary">{r.n_cycles ?? '—'}</td>
                <td className="px-3 py-2 font-mono text-text">
                  {r.error ? <span className="text-danger text-xs">{r.error}</span>
                           : Math.round(r.predicted_rul)}
                </td>
                <td className="px-3 py-2">
                  {!r.error && (
                    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${HEALTH_STYLE[r.health_color] || ''}`}>
                      {r.health_label}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 font-mono text-text-secondary">
                  {r.confidence != null ? `${Math.round(r.confidence * 100)}%` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
