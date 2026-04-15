// src/components/InputPanel.jsx
// Engine / cycle selector for RUL prediction

import React from 'react';

export default function InputPanel({
  engineIds,
  selectedEngine,
  onEngineChange,
  cycle,
  onCycleChange,
  maxCycle,
  onPredict,
  loading,
}) {
  return (
    <div className="card">
      <p className="section-title">Engine Prediction</p>

      <div className="space-y-4 mt-4">
        {/* Engine dropdown */}
        <div>
          <label className="block text-xs text-muted mb-1.5">Engine ID</label>
          <select
            id="engine-select"
            className="input-base"
            value={selectedEngine || ''}
            onChange={(e) => onEngineChange(Number(e.target.value))}
          >
            <option value="">— Select Engine —</option>
            {engineIds.map((id) => (
              <option key={id} value={id}>
                {id >= 4000 ? 'FD004' : id >= 3000 ? 'FD003' : id >= 2000 ? 'FD002' : 'FD001'} - Engine {id % 1000}
              </option>
            ))}
          </select>
        </div>

        {/* Cycle input */}
        <div>
          <label className="block text-xs text-muted mb-1.5">
            Cycle
            {maxCycle > 0 && (
              <span className="ml-2 text-accent font-mono">(max: {maxCycle})</span>
            )}
          </label>

          {/* Slider — always visible */}
          <input
            type="range"
            min={1}
            max={maxCycle > 0 ? maxCycle : 500}
            value={cycle}
            onChange={(e) => onCycleChange(Number(e.target.value))}
            className="w-full accent-accent mb-2"
          />

          {/* Number input below slider */}
          <input
            id="cycle-input"
            type="number"
            min={1}
            max={maxCycle || 999}
            className="input-base font-mono"
            value={cycle}
            onChange={(e) => onCycleChange(Number(e.target.value))}
            placeholder="e.g. 50"
          />
        </div>

        <button
          id="predict-btn"
          className="btn-primary w-full"
          onClick={onPredict}
          disabled={!selectedEngine || loading}
        >
          {loading ? 'Predicting…' : 'Predict RUL'}
        </button>
      </div>
    </div>
  );
}
