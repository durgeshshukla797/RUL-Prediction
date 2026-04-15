// src/components/UploadPage.jsx
// Upload & Predict — batch inference + cycle-by-cycle simulation from uploaded CSV

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { uploadCsv } from '../api/client';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell, LineChart, Line,
} from 'recharts';

// ── Constants ─────────────────────────────────────────────────────────────────

const HEALTH_META = {
  0: { label: 'Critical', color: '#EF4444', bg: 'rgba(239,68,68,0.12)', ring: 'rgba(239,68,68,0.4)' },
  1: { label: 'Warning',  color: '#F59E0B', bg: 'rgba(245,158,11,0.12)', ring: 'rgba(245,158,11,0.4)' },
  2: { label: 'Healthy',  color: '#10B981', bg: 'rgba(16,185,129,0.12)', ring: 'rgba(16,185,129,0.4)' },
};

const SPEED_OPTIONS = [
  { label: '0.5×', ms: 2000 },
  { label: '1×',   ms: 1000 },
  { label: '2×',   ms: 500  },
  { label: '5×',   ms: 200  },
];

const card = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 16,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function healthBadge(cls) {
  const m = HEALTH_META[cls] ?? HEALTH_META[2];
  return (
    <span style={{
      background: m.bg, color: m.color,
      border: `1px solid ${m.ring}`,
      borderRadius: 999, padding: '2px 10px',
      fontSize: 11, fontWeight: 600, whiteSpace: 'nowrap',
    }}>
      {m.label}
    </span>
  );
}

function RULBar({ value, max = 125 }) {
  const pct = Math.min(100, (value / max) * 100);
  const color = value <= 15 ? '#EF4444' : value <= 30 ? '#F59E0B' : '#10B981';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%' }}>
      <div style={{ flex: 1, height: 6, background: 'rgba(255,255,255,0.07)', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 4, transition: 'width 0.5s ease' }} />
      </div>
      <span style={{ fontFamily: 'monospace', fontSize: 12, color, minWidth: 40 }}>
        {Math.round(value)}
      </span>
    </div>
  );
}

/** Parse raw CSV text → { engineId: [{time, raw}] } */
function parseRawCSV(text) {
  const engines = {};
  for (const line of text.trim().split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const parts = trimmed.split(/\s+/);
    const unit = parseInt(parts[0], 10);
    const time = parseInt(parts[1], 10);
    if (isNaN(unit) || isNaN(time) || unit <= 0 || time <= 0) continue; // skip header / bad rows
    if (!engines[unit]) engines[unit] = [];
    engines[unit].push({ time, raw: trimmed });
  }
  return engines;
}

// ── Drop Zone ─────────────────────────────────────────────────────────────────

function DropZone({ onFile, loading }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  }, [onFile]);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !loading && inputRef.current.click()}
      style={{
        border: `2px dashed ${dragging ? '#6366F1' : 'rgba(255,255,255,0.15)'}`,
        borderRadius: 16, padding: '48px 24px', textAlign: 'center',
        cursor: loading ? 'wait' : 'pointer',
        background: dragging ? 'rgba(99,102,241,0.07)' : 'rgba(255,255,255,0.02)',
        transition: 'all 0.25s ease',
      }}
    >
      <input ref={inputRef} type="file" accept=".csv,.txt" style={{ display: 'none' }}
        onChange={(e) => { if (e.target.files[0]) onFile(e.target.files[0]); }} />
      {loading ? (
        <>
          <div style={{ fontSize: 36, marginBottom: 12 }}>⚙️</div>
          <p style={{ color: '#6366F1', fontWeight: 600, marginBottom: 4 }}>Running inference…</p>
          <p style={{ color: '#6B7280', fontSize: 13 }}>Model predicting RUL for each engine</p>
        </>
      ) : (
        <>
          <div style={{ fontSize: 40, marginBottom: 12 }}>📂</div>
          <p style={{ color: '#E5E7EB', fontWeight: 600, marginBottom: 4 }}>Drop your CSV here or click to browse</p>
          <p style={{ color: '#6B7280', fontSize: 13, marginBottom: 16 }}>
            Whitespace-separated CMAPSS format · <code style={{ color: '#6366F1', fontSize: 11 }}>unit  time  c1  c2  c3  s1…s21</code>
          </p>
          <a href="/sample_engine_data.csv" download="sample_engine_data.csv"
            onClick={(e) => e.stopPropagation()}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              background: 'rgba(99,102,241,0.15)', color: '#818CF8',
              border: '1px solid rgba(99,102,241,0.3)',
              borderRadius: 8, padding: '6px 14px', fontSize: 12,
              fontWeight: 500, textDecoration: 'none',
            }}
          >
            ⬇ Download sample CSV (5 engines: Healthy / Warning / Critical)
          </a>
        </>
      )}
    </div>
  );
}

// ── Stats Strip ───────────────────────────────────────────────────────────────

function StatStrip({ results }) {
  const valids = results.filter(r => r.predicted_rul !== undefined);
  if (!valids.length) return null;
  const ruls = valids.map(r => r.predicted_rul);
  const avgRUL = (ruls.reduce((a, b) => a + b, 0) / ruls.length).toFixed(1);
  const minRUL = Math.min(...ruls).toFixed(0);
  const stats = [
    { label: 'Engines',  value: valids.length,                              color: '#6366F1', unit: '' },
    { label: 'Avg RUL',  value: avgRUL,                                     color: '#6366F1', unit: ' cyc' },
    { label: 'Min RUL',  value: minRUL,                                     color: '#EF4444', unit: ' cyc' },
    { label: 'Critical', value: valids.filter(r => r.health_class === 0).length, color: '#EF4444', unit: '' },
    { label: 'Warning',  value: valids.filter(r => r.health_class === 1).length, color: '#F59E0B', unit: '' },
    { label: 'Healthy',  value: valids.filter(r => r.health_class === 2).length, color: '#10B981', unit: '' },
  ];
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 24 }}>
      {stats.map(s => (
        <div key={s.label} style={{ ...card, padding: '14px 16px' }}>
          <p style={{ fontSize: 11, color: '#6B7280', marginBottom: 4 }}>{s.label}</p>
          <p style={{ fontSize: 22, fontWeight: 700, fontFamily: 'monospace', color: s.color, margin: 0 }}>
            {s.value}<span style={{ fontSize: 11, color: '#6B7280' }}>{s.unit}</span>
          </p>
        </div>
      ))}
    </div>
  );
}

// ── RUL Distribution Chart ────────────────────────────────────────────────────

function RULDistributionChart({ results }) {
  const valids = results.filter(r => r.predicted_rul !== undefined);
  if (valids.length < 2) return null;
  const buckets = {};
  valids.forEach(r => {
    const key = Math.floor(r.predicted_rul / 10) * 10;
    buckets[key] = (buckets[key] || 0) + 1;
  });
  const data = Object.keys(buckets).sort((a, b) => Number(a) - Number(b)).map(k => ({
    range: `${k}–${Number(k) + 10}`,
    count: buckets[k],
    fill: Number(k) <= 15 ? '#EF4444' : Number(k) <= 30 ? '#F59E0B' : '#10B981',
  }));
  return (
    <div style={{ ...card, padding: 20, marginBottom: 24 }}>
      <p style={{ fontSize: 13, fontWeight: 600, color: '#E5E7EB', marginBottom: 16 }}>RUL Distribution</p>
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={data} margin={{ top: 0, right: 0, left: -24, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="range" tick={{ fill: '#6B7280', fontSize: 10 }} />
          <YAxis tick={{ fill: '#6B7280', fontSize: 10 }} allowDecimals={false} />
          <Tooltip
            contentStyle={{ background: '#1F2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
            formatter={(v) => [`${v} engines`, 'Count']}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
            {data.map((d, i) => <Cell key={i} fill={d.fill} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Results Table ─────────────────────────────────────────────────────────────

function ResultsTable({ results }) {
  const [sort, setSort] = useState({ col: 'predicted_rul', dir: 'asc' });
  const [filter, setFilter] = useState('all');
  const toggle = (col) => setSort(s => ({ col, dir: s.col === col && s.dir === 'asc' ? 'desc' : 'asc' }));
  const filtered = results.filter(r => r.predicted_rul !== undefined)
    .filter(r => filter === 'all' || String(r.health_class) === filter);
  const sorted = [...filtered].sort((a, b) => {
    const v = sort.col === 'predicted_rul' ? a.predicted_rul - b.predicted_rul : a.engine_id - b.engine_id;
    return sort.dir === 'asc' ? v : -v;
  });
  const arrow = (col) => sort.col === col ? (sort.dir === 'asc' ? ' ↑' : ' ↓') : '';
  return (
    <div style={{ ...card, overflow: 'hidden', marginBottom: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '14px 20px', borderBottom: '1px solid rgba(255,255,255,0.07)', flexWrap: 'wrap' }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: '#E5E7EB', marginRight: 8 }}>
          Results — {sorted.length} engines
        </span>
        {[['all', 'All'], ['2', 'Healthy'], ['1', 'Warning'], ['0', 'Critical']].map(([v, l]) => (
          <button key={v} onClick={() => setFilter(v)} style={{
            padding: '3px 12px', borderRadius: 999, fontSize: 11, fontWeight: 500, cursor: 'pointer',
            border: `1px solid ${v === 'all' ? (filter === 'all' ? '#6366F1' : 'rgba(255,255,255,0.15)') : v === '2' ? '#10B981' : v === '1' ? '#F59E0B' : '#EF4444'}`,
            background: filter === v ? (v === 'all' ? 'rgba(99,102,241,0.2)' : v === '2' ? 'rgba(16,185,129,0.2)' : v === '1' ? 'rgba(245,158,11,0.2)' : 'rgba(239,68,68,0.2)') : 'transparent',
            color: v === 'all' ? '#E5E7EB' : v === '2' ? '#10B981' : v === '1' ? '#F59E0B' : '#EF4444',
            transition: 'all 0.15s',
          }}>{l}</button>
        ))}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
              {[{ key: 'engine_id', label: 'Engine' }, { key: 'predicted_rul', label: 'Predicted RUL' },
                { key: 'health_class', label: 'Health' }, { key: 'confidence', label: 'Confidence' }, { key: 'n_cycles', label: 'Cycles' }]
                .map(({ key, label }) => (
                  <th key={key} onClick={() => (key === 'engine_id' || key === 'predicted_rul') && toggle(key)}
                    style={{ padding: '10px 16px', textAlign: 'left', fontWeight: 600, color: '#9CA3AF', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', cursor: (key === 'engine_id' || key === 'predicted_rul') ? 'pointer' : 'default', userSelect: 'none' }}>
                    {label}{arrow(key)}
                  </th>
                ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => (
              <tr key={r.engine_id}
                style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)', transition: 'background 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(99,102,241,0.08)'}
                onMouseLeave={e => e.currentTarget.style.background = i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)'}
              >
                <td style={{ padding: '10px 16px', fontFamily: 'monospace', color: '#A5B4FC', fontWeight: 600 }}>#{r.engine_id}</td>
                <td style={{ padding: '10px 16px' }}><RULBar value={r.predicted_rul} /></td>
                <td style={{ padding: '10px 16px' }}>{healthBadge(r.health_class)}</td>
                <td style={{ padding: '10px 16px', fontFamily: 'monospace', color: '#6B7280' }}>
                  {r.confidence !== undefined ? `${(r.confidence * 100).toFixed(1)}%` : '—'}
                </td>
                <td style={{ padding: '10px 16px', color: '#6B7280' }}>{r.n_cycles ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {sorted.length === 0 && (
          <div style={{ textAlign: 'center', padding: 32, color: '#6B7280', fontSize: 13 }}>No engines match the current filter.</div>
        )}
      </div>
    </div>
  );
}

// ── Upload Simulation Panel ───────────────────────────────────────────────────

function UploadSimPanel({ parsedEngines }) {
  const engineIds = Object.keys(parsedEngines).map(Number).sort((a, b) => a - b);
  const [engineId, setEngineId] = useState(engineIds[0] || 1);
  const [maxCycle, setMaxCycle] = useState(1);
  const [currentCycle, setCurrentCycle] = useState(1);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1000);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [simLoading, setSimLoading] = useState(false);
  const intervalRef = useRef(null);

  // Reset when engine changes
  useEffect(() => {
    clearInterval(intervalRef.current);
    setRunning(false);
    const rows = parsedEngines[engineId] || [];
    const mx = rows.length > 0 ? Math.max(...rows.map(r => r.time)) : 1;
    setMaxCycle(mx);
    setCurrentCycle(1);
    setHistory([]);
    setPrediction(null);
  }, [engineId, parsedEngines]);

  const runStep = useCallback(async (cycle) => {
    const rows = (parsedEngines[engineId] || []).filter(r => r.time <= cycle);
    if (!rows.length) return;
    const csvText = rows.map(r => r.raw).join('\n');
    const file = new File([csvText], 'sim_slice.csv', { type: 'text/csv' });
    try {
      setSimLoading(true);
      const result = await uploadCsv(file);
      const pred = result.predictions?.[0];
      if (pred) {
        setPrediction(pred);
        setHistory(h => [...h, {
          cycle,
          rul: parseFloat(pred.predicted_rul.toFixed(1)),
          health: pred.health_class,
        }]);
      }
    } catch {
      setRunning(false);
    } finally {
      setSimLoading(false);
    }
  }, [engineId, parsedEngines]);

  useEffect(() => {
    if (!running) return;
    intervalRef.current = setInterval(() => {
      setCurrentCycle(prev => {
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
    return () => clearInterval(intervalRef.current);
  }, [running, speed, maxCycle, runStep]);

  const handleStart = () => { setHistory([]); runStep(currentCycle); setRunning(true); };
  const handleStop  = () => { setRunning(false); clearInterval(intervalRef.current); };
  const handleReset = () => { handleStop(); setCurrentCycle(1); setHistory([]); setPrediction(null); };

  const meta = prediction ? (HEALTH_META[prediction.health_class] ?? HEALTH_META[2]) : null;
  const btnBase = { borderRadius: 8, padding: '7px 18px', fontSize: 13, fontWeight: 600, cursor: 'pointer', border: 'none', transition: 'opacity 0.2s' };

  return (
    <div style={{ ...card, padding: 24, marginBottom: 24 }}>
      <p style={{ fontSize: 15, fontWeight: 700, color: '#E5E7EB', marginBottom: 18 }}>
        🎬 Live Simulation — Uploaded Engine Data
      </p>

      {/* Controls row */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-end', marginBottom: 24 }}>

        {/* Engine selector */}
        <div>
          <p style={{ fontSize: 11, color: '#6B7280', marginBottom: 6 }}>Engine</p>
          <select value={engineId} onChange={e => { handleReset(); setEngineId(Number(e.target.value)); }}
            style={{ background: '#1F2937', color: '#E5E7EB', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 8, padding: '7px 12px', fontSize: 13 }}>
            {engineIds.map(id => <option key={id} value={id}>Engine #{id} ({(parsedEngines[id] || []).length} cycles)</option>)}
          </select>
        </div>

        {/* Speed */}
        <div>
          <p style={{ fontSize: 11, color: '#6B7280', marginBottom: 6 }}>Speed</p>
          <div style={{ display: 'flex', gap: 4 }}>
            {SPEED_OPTIONS.map(({ label, ms }) => (
              <button key={ms} onClick={() => setSpeed(ms)} style={{
                padding: '6px 12px', borderRadius: 6, fontSize: 11, fontWeight: 600, cursor: 'pointer',
                border: `1px solid ${speed === ms ? '#6366F1' : 'rgba(255,255,255,0.15)'}`,
                background: speed === ms ? 'rgba(99,102,241,0.25)' : 'transparent',
                color: speed === ms ? '#A5B4FC' : '#9CA3AF',
              }}>{label}</button>
            ))}
          </div>
        </div>

        {/* Buttons */}
        <div style={{ display: 'flex', gap: 8 }}>
          {!running ? (
            <button onClick={handleStart} disabled={simLoading}
              style={{ ...btnBase, background: '#4F46E5', color: '#fff', opacity: simLoading ? 0.6 : 1 }}>
              {simLoading ? '…' : '▶ Start'}
            </button>
          ) : (
            <button onClick={handleStop} style={{ ...btnBase, background: '#374151', color: '#E5E7EB' }}>⏸ Pause</button>
          )}
          <button onClick={handleReset} style={{ ...btnBase, background: '#374151', color: '#E5E7EB' }}>↺ Reset</button>
        </div>

        {/* Progress */}
        <div style={{ fontFamily: 'monospace', fontSize: 13, color: '#6B7280' }}>
          Cycle <span style={{ color: '#A5B4FC', fontWeight: 700 }}>{currentCycle}</span> / {maxCycle}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>

        {/* Live prediction card */}
        <div style={{
          background: meta ? meta.bg : 'rgba(255,255,255,0.03)',
          border: `1px solid ${meta ? meta.ring : 'rgba(255,255,255,0.08)'}`,
          borderRadius: 14, padding: 20, minHeight: 160,
          display: 'flex', flexDirection: 'column', justifyContent: 'center',
        }}>
          {prediction ? (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                <span style={{ color: '#9CA3AF', fontSize: 12 }}>Predicted RUL</span>
                {healthBadge(prediction.health_class)}
              </div>
              <p style={{ fontFamily: 'monospace', fontSize: 42, fontWeight: 800, color: meta.color, margin: '0 0 8px' }}>
                {Math.round(prediction.predicted_rul)}
                <span style={{ fontSize: 16, color: '#9CA3AF', fontWeight: 400 }}> cycles</span>
              </p>
              <div style={{ display: 'flex', gap: 16 }}>
                <span style={{ fontSize: 12, color: '#6B7280' }}>
                  Conf: <span style={{ color: '#E5E7EB' }}>{((prediction.confidence ?? 0) * 100).toFixed(1)}%</span>
                </span>
                <span style={{ fontSize: 12, color: '#6B7280' }}>
                  Cycle: <span style={{ color: '#E5E7EB' }}>{currentCycle}</span>
                </span>
              </div>
            </>
          ) : (
            <div style={{ textAlign: 'center', color: '#6B7280' }}>
              <p style={{ fontSize: 28 }}>📊</p>
              <p style={{ fontSize: 13 }}>Press ▶ Start to begin simulation</p>
            </div>
          )}
        </div>

        {/* RUL trend sparkline */}
        <div>
          {history.length > 1 ? (
            <>
              <p style={{ fontSize: 12, color: '#6B7280', marginBottom: 8 }}>RUL trend across cycles</p>
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={history} margin={{ top: 5, right: 10, left: -24, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="cycle" tick={{ fill: '#6B7280', fontSize: 9 }} />
                  <YAxis domain={[0, 'auto']} tick={{ fill: '#6B7280', fontSize: 9 }} />
                  <Tooltip
                    contentStyle={{ background: '#1F2937', border: '1px solid #374151', borderRadius: 8, fontSize: 11 }}
                    formatter={(v) => [`${Math.round(v)} cycles`, 'RUL']}
                  />
                  <Line type="monotone" dataKey="rul" stroke="#6366F1" dot={false} strokeWidth={2.5} name="Pred. RUL" />
                </LineChart>
              </ResponsiveContainer>
            </>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#6B7280', fontSize: 13 }}>
              RUL trend will appear here as simulation runs.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function UploadPage() {
  const [results, setResults]             = useState(null);
  const [parsedEngines, setParsedEngines] = useState(null);
  const [filename, setFilename]           = useState('');
  const [loading, setLoading]             = useState(false);
  const [error, setError]                 = useState(null);

  const handleFile = useCallback(async (file) => {
    setLoading(true);
    setError(null);
    setResults(null);
    setParsedEngines(null);
    setFilename(file.name);

    // Read raw text for simulation
    const rawText = await file.text();
    const engines = parseRawCSV(rawText);

    try {
      const data = await uploadCsv(file);
      setResults(data.predictions || []);
      setParsedEngines(engines);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || 'Upload failed.');
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = () => { setResults(null); setParsedEngines(null); setError(null); setFilename(''); };

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>

      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: '#E5E7EB', margin: 0, marginBottom: 6 }}>
          Upload & Predict
        </h2>
        <p style={{ color: '#6B7280', fontSize: 14, margin: 0 }}>
          Upload a CMAPSS-format CSV for batch RUL inference + live engine simulation.
        </p>
      </div>

      {/* Drop zone */}
      {!results && (
        <div style={{ marginBottom: 24 }}>
          <DropZone onFile={handleFile} loading={loading} />
          {error && (
            <div style={{ marginTop: 16, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 12, padding: '12px 16px', color: '#FCA5A5', fontSize: 13 }}>
              ⚠️ {error}
            </div>
          )}
          <div style={{ marginTop: 20, background: 'rgba(99,102,241,0.06)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 12, padding: '14px 18px' }}>
            <p style={{ color: '#818CF8', fontSize: 12, fontWeight: 600, marginBottom: 8 }}>📋 Expected CSV format</p>
            <p style={{ color: '#6B7280', fontSize: 12, margin: 0, lineHeight: 1.8 }}>
              Whitespace-separated (space or tab), header row optional.<br />
              All 26 columns: <code style={{ color: '#A5B4FC', fontSize: 11 }}>unit  time  c1  c2  c3  s1 … s21</code><br />
              Model features (18): <code style={{ color: '#6EE7B7', fontSize: 11 }}>c1 c2 c3 s2 s3 s4 s6 s7 s8 s9 s11 s12 s13 s14 s15 s17 s20 s21</code>
            </p>
          </div>
        </div>
      )}

      {/* Results */}
      {results && !loading && (
        <>
          {/* Result header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
            <div>
              <p style={{ color: '#E5E7EB', fontWeight: 600, margin: 0 }}>
                ✅ Inference complete — <span style={{ color: '#A5B4FC' }}>{filename}</span>
              </p>
              <p style={{ color: '#6B7280', fontSize: 12, margin: 0 }}>{results.length} engines processed</p>
            </div>
            <button onClick={reset} style={{
              background: 'rgba(99,102,241,0.15)', color: '#818CF8',
              border: '1px solid rgba(99,102,241,0.3)', borderRadius: 8,
              padding: '7px 16px', fontSize: 13, cursor: 'pointer', fontWeight: 500,
            }}>
              ↑ Upload another
            </button>
          </div>

          <StatStrip results={results} />
          <RULDistributionChart results={results} />

          {/* Simulation panel — only when parsedEngines are ready */}
          {parsedEngines && Object.keys(parsedEngines).length > 0 && (
            <UploadSimPanel parsedEngines={parsedEngines} />
          )}

          <ResultsTable results={results} />
        </>
      )}
    </div>
  );
}
