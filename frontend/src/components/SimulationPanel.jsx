// src/components/SimulationPanel.jsx
// Real-time cycle-by-cycle simulation for a selected engine

import React, { useState, useEffect, useRef, useCallback } from 'react';
import PredictionCard from './PredictionCard';
import { predictEngineAtCycle } from '../api/client';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts';

const SPEED_OPTIONS = [
  { label: '0.5×', ms: 2000 },
  { label: '1×',   ms: 1000 },
  { label: '2×',   ms: 500  },
  { label: '5×',   ms: 200  },
];

export default function SimulationPanel({ engineIds }) {
  const [engineId, setEngineId] = useState(engineIds[0] || 1);
  const [maxCycle, setMaxCycle] = useState(100);
  const [currentCycle, setCurrentCycle] = useState(30);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1000);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const intervalRef = useRef(null);

  const runStep = useCallback(async (cycle) => {
    try {
      setLoading(true);
      const result = await predictEngineAtCycle(engineId, cycle);
      setPrediction(result);
      setHistory((prev) => [
        ...prev,
        { cycle, rul: result.predicted_rul, health: result.health_class },
      ]);
    } catch {
      // stop simulation on error
      setRunning(false);
    } finally {
      setLoading(false);
    }
  }, [engineId]);

  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(() => {
        setCurrentCycle((prev) => {
          const next = prev + 1;
          if (next > maxCycle) {
            setRunning(false);
            clearInterval(intervalRef.current);
            return prev;
          }
          runStep(next);
          return next;
        });
      }, speed);
    }
    return () => clearInterval(intervalRef.current);
  }, [running, speed, maxCycle, runStep]);

  const handleStart = () => {
    setHistory([]);
    runStep(currentCycle);
    setRunning(true);
  };

  const handleStop = () => {
    setRunning(false);
    clearInterval(intervalRef.current);
  };

  const handleReset = () => {
    handleStop();
    setCurrentCycle(30);
    setHistory([]);
    setPrediction(null);
  };

  return (
    <div className="card">
      <p className="section-title">Real-time Simulation</p>

      <div className="flex flex-wrap gap-4 mb-4 items-end">
        {/* Engine */}
        <div>
          <label className="block text-xs text-muted mb-1">Engine</label>
          <select
            className="input-base w-32"
            value={engineId}
            onChange={(e) => { handleReset(); setEngineId(Number(e.target.value)); }}
          >
            {engineIds.slice(0, 20).map((id) => (
              <option key={id} value={id}>
                {id >= 4000 ? 'FD004' : id >= 3000 ? 'FD003' : id >= 2000 ? 'FD002' : 'FD001'} - Engine {id % 1000}
              </option>
            ))}
          </select>
        </div>

        {/* Max cycle */}
        <div>
          <label className="block text-xs text-muted mb-1">Max cycle</label>
          <input
            type="number" min={31} max={999}
            className="input-base w-24 font-mono"
            value={maxCycle}
            onChange={(e) => setMaxCycle(Number(e.target.value))}
          />
        </div>

        {/* Speed */}
        <div>
          <label className="block text-xs text-muted mb-1">Speed</label>
          <div className="flex gap-1">
            {SPEED_OPTIONS.map(({ label, ms }) => (
              <button
                key={ms}
                onClick={() => setSpeed(ms)}
                className={`px-2 py-1 text-xs rounded border transition-colors
                  ${speed === ms
                    ? 'bg-accent border-accent text-white'
                    : 'border-border text-muted hover:border-accent'}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div className="flex gap-2">
          {!running ? (
            <button className="btn-primary" onClick={handleStart} disabled={loading}>
              {loading ? '…' : '▶ Start'}
            </button>
          ) : (
            <button className="btn-secondary" onClick={handleStop}>⏸ Pause</button>
          )}
          <button className="btn-secondary" onClick={handleReset}>↺ Reset</button>
        </div>

        {/* Progress */}
        <div className="text-xs text-muted font-mono">
          Cycle: <span className="text-accent">{currentCycle}</span> / {maxCycle}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Live prediction card */}
        <PredictionCard
          prediction={prediction}
          loading={loading && !prediction}
        />

        {/* RUL history sparkline */}
        <div>
          {history.length > 1 ? (
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" />
                <XAxis dataKey="cycle" tick={{ fill: '#6B7280', fontSize: 9 }} />
                <YAxis tick={{ fill: '#6B7280', fontSize: 9 }} domain={[0, 'auto']} />
                <Tooltip
                  contentStyle={{ background: '#1F2937', border: '1px solid #374151', borderRadius: 6, fontSize: 11 }}
                  labelStyle={{ color: '#9CA3AF' }}
                />
                <Line type="monotone" dataKey="rul" stroke="#3B82F6" dot={false} strokeWidth={2} name="Pred. RUL" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-44 text-muted text-sm">
              Start simulation to see live RUL trend.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
