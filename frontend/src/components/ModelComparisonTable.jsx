// src/components/ModelComparisonTable.jsx
// Displays RMSE, MAE, Accuracy, F1 for all trained models

import React from 'react';

const HEADERS = ['Model', 'RMSE ↓', 'MAE ↓', 'Accuracy ↑', 'F1 ↑'];

function MetricCell({ value, highlight }) {
  return (
    <td className={`px-4 py-3 font-mono text-sm text-right ${highlight ? 'text-accent font-semibold' : 'text-text'}`}>
      {typeof value === 'number' ? value.toFixed(4) : value}
    </td>
  );
}

export default function ModelComparisonTable({ metrics }) {
  if (!metrics || !Object.keys(metrics).length) {
    return (
      <div className="card">
        <p className="section-title">Model Comparison</p>
        <p className="text-muted text-sm">No metrics found. Run train.py first.</p>
      </div>
    );
  }

  const rows = Object.entries(metrics);

  // Find bests
  const bestRMSE = Math.min(...rows.map(([, v]) => v.rmse));
  const bestMAE  = Math.min(...rows.map(([, v]) => v.mae));
  const bestAcc  = Math.max(...rows.map(([, v]) => v.accuracy));
  const bestF1   = Math.max(...rows.map(([, v]) => v.f1));

  return (
    <div className="card">
      <p className="section-title">Model Comparison</p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              {HEADERS.map((h) => (
                <th
                  key={h}
                  className="px-4 py-2 text-left text-xs font-semibold text-muted uppercase tracking-wider first:text-left last:text-right"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, v]) => (
              <tr key={name} className="border-b border-border/50 hover:bg-bg/50 transition-colors">
                <td className="px-4 py-3 font-medium text-text capitalize">{name}</td>
                <MetricCell value={v.rmse} highlight={v.rmse === bestRMSE} />
                <MetricCell value={v.mae}  highlight={v.mae  === bestMAE}  />
                <MetricCell value={v.accuracy} highlight={v.accuracy === bestAcc} />
                <MetricCell value={v.f1}   highlight={v.f1   === bestF1}   />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-muted mt-3">
        <span className="text-accent font-semibold">Bold blue</span> = best value per metric. 
        RMSE/MAE = lower is better; Accuracy/F1 = higher is better.
      </p>
    </div>
  );
}
