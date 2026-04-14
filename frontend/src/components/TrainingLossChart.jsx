// src/components/TrainingLossChart.jsx
// Shows training/validation loss curves with model toggle

import React, { useState } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from 'recharts';

const MODEL_COLORS = {
  hybrid: '#3B82F6',
  lstm:   '#10B981',
  cnn:    '#F59E0B',
};

const MODEL_COLORS_VAL = {
  hybrid: '#93C5FD',
  lstm:   '#6EE7B7',
  cnn:    '#FCD34D',
};

export default function TrainingLossChart({ history }) {
  const available = (history || []).map((h) => h.model);
  const [activeModel, setActiveModel] = useState(available[0] || 'hybrid');

  const modelHistory = history?.find((h) => h.model === activeModel);

  const chartData = modelHistory
    ? modelHistory.epochs.map((ep, i) => ({
        epoch: ep,
        'Train Loss': modelHistory.train_loss[i],
        'Val Loss':   modelHistory.val_loss[i],
        'Train MAE':  modelHistory.train_rul_mae[i],
        'Val MAE':    modelHistory.val_rul_mae[i],
      }))
    : [];

  if (!history?.length) {
    return (
      <div className="card">
        <p className="section-title">Training Loss</p>
        <div className="flex items-center justify-center h-48 text-muted text-sm">
          No training history. Run train.py first.
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <p className="section-title mb-0">Training History</p>
        <div className="flex gap-2">
          {available.map((m) => (
            <button
              key={m}
              onClick={() => setActiveModel(m)}
              className={`px-3 py-1 rounded-md text-xs font-medium border transition-colors
                ${activeModel === m
                  ? 'text-white border-transparent'
                  : 'border-border text-muted hover:text-text'}`}
              style={activeModel === m ? { background: MODEL_COLORS[m] } : {}}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" />
          <XAxis
            dataKey="epoch"
            tick={{ fill: '#6B7280', fontSize: 10 }}
            label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#6B7280', fontSize: 11 }}
          />
          <YAxis tick={{ fill: '#6B7280', fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: '#1F2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#9CA3AF' }}
          />
          <Legend wrapperStyle={{ color: '#9CA3AF', fontSize: 12, paddingTop: 8 }} />
          <Line type="monotone" dataKey="Train Loss" stroke={MODEL_COLORS[activeModel]}
                dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="Val Loss" stroke={MODEL_COLORS_VAL[activeModel]}
                dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
