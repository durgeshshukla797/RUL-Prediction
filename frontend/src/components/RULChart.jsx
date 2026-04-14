// src/components/RULChart.jsx
// Actual vs Predicted RUL for all 100 test engines

import React, { useMemo } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine,
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-surface border border-border rounded-lg px-3 py-2 text-xs shadow-lg">
      <p className="text-muted mb-1">Engine {label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex gap-2 items-center">
          <span style={{ color: p.color }}>■</span>
          <span className="text-text-secondary">{p.name}:</span>
          <span className="font-mono text-text">{Math.round(p.value)}</span>
        </div>
      ))}
    </div>
  );
};

export default function RULChart({ predictions, highlightedEngine }) {
  const data = useMemo(() => {
    if (!predictions?.length) return [];
    return predictions.map((p) => ({
      engine: p.engine_id,
      Actual: p.actual_rul,
      Predicted: p.predicted_rul,
    }));
  }, [predictions]);

  if (!data.length) {
    return (
      <div className="card">
        <p className="section-title">Actual vs Predicted RUL — Test Engines</p>
        <div className="flex items-center justify-center h-48 text-muted text-sm">
          No prediction data available. Run train.py first.
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <p className="section-title">Actual vs Predicted RUL — All Test Engines</p>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" />
          <XAxis
            dataKey="engine"
            label={{ value: 'Engine ID', position: 'insideBottom', offset: -2, fill: '#6B7280', fontSize: 11 }}
            tick={{ fill: '#6B7280', fontSize: 10 }}
          />
          <YAxis
            label={{ value: 'RUL (cycles)', angle: -90, position: 'insideLeft', fill: '#6B7280', fontSize: 11 }}
            tick={{ fill: '#6B7280', fontSize: 10 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ color: '#9CA3AF', fontSize: 12, paddingTop: 8 }}
          />
          {highlightedEngine && (
            <ReferenceLine
              x={highlightedEngine}
              stroke="#3B82F6"
              strokeDasharray="4 2"
              label={{ value: `E${highlightedEngine}`, fill: '#3B82F6', fontSize: 10 }}
            />
          )}
          <Line
            type="monotone"
            dataKey="Actual"
            stroke="#10B981"
            dot={false}
            strokeWidth={1.5}
          />
          <Line
            type="monotone"
            dataKey="Predicted"
            stroke="#3B82F6"
            dot={false}
            strokeWidth={1.5}
            strokeDasharray="4 2"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
