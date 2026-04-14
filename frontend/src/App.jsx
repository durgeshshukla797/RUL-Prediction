// src/App.jsx
// Main application shell — orchestrates all panels and data fetching

import React, { useState, useEffect, useCallback } from 'react';
import InputPanel from './components/InputPanel';
import PredictionCard from './components/PredictionCard';
import RULChart from './components/RULChart';
import SensorChart from './components/SensorChart';
import TrainingLossChart from './components/TrainingLossChart';
import ModelComparisonTable from './components/ModelComparisonTable';
import SimulationPanel from './components/SimulationPanel';
import BatchResultsTable from './components/BatchResultsTable';

import {
  fetchEngines,
  fetchEngineData,
  fetchMetrics,
  fetchHistory,
  fetchPredictions,
  predict,
  uploadCsv,
} from './api/client';

// ── Fallback: show engines 1-100 even when API is offline ─────────────────
const FALLBACK_ENGINE_IDS = Array.from({ length: 100 }, (_, i) => i + 1);

function StatusDot({ ok }) {
  return (
    <span className={`inline-block w-2 h-2 rounded-full ${ok ? 'bg-success' : 'bg-danger animate-pulse'}`} />
  );
}

// ── Setup banner shown when API is offline ─────────────────────────────────
function SetupBanner({ onRetry, onDismiss }) {
  return (
    <div className="bg-warning/5 border border-warning/30 rounded-xl p-4 flex flex-col sm:flex-row gap-3 items-start sm:items-center">
      <div className="flex-1 min-w-0">
        <p className="text-warning text-sm font-semibold mb-1">API not connected</p>
        <p className="text-text-secondary text-xs leading-relaxed">
          The backend is offline. Run these commands in your project directory, then click Retry:
        </p>
        <div className="mt-2 space-y-1">
          <code className="block bg-bg text-accent text-xs rounded px-2 py-1 font-mono">
            python train.py --model hybrid --epochs 30
          </code>
          <code className="block bg-bg text-accent text-xs rounded px-2 py-1 font-mono">
            uvicorn api.app:app --reload --port 8000
          </code>
        </div>
      </div>
      <div className="flex gap-2 shrink-0">
        <button onClick={onRetry} className="btn-primary text-xs py-1.5 px-3">
          Retry
        </button>
        <button onClick={onDismiss} className="btn-secondary text-xs py-1.5 px-3">
          Dismiss
        </button>
      </div>
    </div>
  );
}

export default function App() {
  // ── Global data ──────────────────────────────────────────────────────────
  const [dataset,     setDataset]     = useState('FD001');
  const [engineIds,   setEngineIds]   = useState(FALLBACK_ENGINE_IDS); // pre-seeded with 1-100
  const [metrics,     setMetrics]     = useState({});
  const [history,     setHistory]     = useState([]);
  const [allPreds,    setAllPreds]    = useState([]);
  const [apiOk,       setApiOk]       = useState(false);
  const [showBanner,  setShowBanner]  = useState(true);

  // ── Prediction state ─────────────────────────────────────────────────────
  const [selectedEngine, setSelectedEngine] = useState(null);
  const [cycle, setCycle]                   = useState(30);
  const [model, setModel]                   = useState('hybrid');
  const [engineData, setEngineData]         = useState(null);
  const [prediction, setPrediction]         = useState(null);
  const [predLoading, setPredLoading]       = useState(false);
  const [predError, setPredError]           = useState(null);

  // ── Upload state ─────────────────────────────────────────────────────────
  const [batchResults, setBatchResults]   = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);

  // ── Active tab ───────────────────────────────────────────────────────────
  const [tab, setTab] = useState('dashboard');

  // ── Load data from API ────────────────────────────────────────────────────
  const loadApiData = useCallback(() => {
    Promise.allSettled([
      fetchEngines(dataset),
      fetchMetrics(dataset),
      fetchHistory(dataset),
      fetchPredictions(dataset),
    ]).then(([engines, mets, hist, preds]) => {
      if (engines.status === 'fulfilled') {
        const ids = engines.value.engine_ids || [];
        if (ids.length > 0) setEngineIds(ids); // override fallback with real data
        setApiOk(true);
        setShowBanner(false);
      } else {
        setApiOk(false);
        // keep FALLBACK_ENGINE_IDS — already set as initial state
      }
      if (mets.status === 'fulfilled')  setMetrics(mets.value);
      if (hist.status === 'fulfilled')  setHistory(hist.value);
      if (preds.status === 'fulfilled') setAllPreds(preds.value);
    });
  }, []);

  useEffect(() => { loadApiData(); }, [loadApiData, dataset]);

  // ── Fetch engine sensor data when selection changes ───────────────────────
  useEffect(() => {
    if (!selectedEngine) return;
    if (!apiOk) { setEngineData(null); return; }
    fetchEngineData(selectedEngine, dataset)
      .then((data) => {
        setEngineData(data);
        const maxC = Math.max(...(data.cycles || [30]));
        setCycle(Math.min(maxC, 50));
      })
      .catch(() => setEngineData(null));
  }, [selectedEngine, apiOk]);

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handlePredict = async () => {
    if (!selectedEngine) return;
    if (!apiOk) {
      setPredError('API is offline. Start uvicorn api.app:app --port 8000 first.');
      return;
    }
    setPredLoading(true);
    setPredError(null);
    try {
      const result = await predict(selectedEngine, cycle, dataset);
      setPrediction(result);
    } catch (e) {
      setPredError(e?.response?.data?.detail || 'Prediction failed. Is the API running?');
      setPrediction(null);
    } finally {
      setPredLoading(false);
    }
  };

  const handleUpload = async (file) => {
    setUploadLoading(true);
    setBatchResults(null);
    try {
      const data = await uploadCsv(file, dataset);
      setBatchResults(data.predictions);
    } catch (e) {
      console.error('Upload failed:', e);
    } finally {
      setUploadLoading(false);
    }
  };

  const maxCycle = engineData ? Math.max(...(engineData.cycles || [0])) : 0;

  // ── UI ────────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-bg text-text">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="border-b border-border bg-surface/80 backdrop-blur sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold tracking-tight">
              Predictive Maintenance
              <span className="text-accent ml-2">·</span>
              <span className="text-text-secondary font-normal text-sm ml-2">CMAPSS</span>
            </h1>
            <p className="text-xs text-muted hidden sm:block">
              Hybrid CNN-LSTM · Remaining Useful Life Prediction
            </p>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={loadApiData}
              title="Retry API connection"
              className="text-xs text-muted hover:text-text transition-colors flex items-center gap-1.5"
            >
              <StatusDot ok={apiOk} />
              <span>{apiOk ? 'API Connected' : 'API Offline'}</span>
              {!apiOk && <span className="text-muted/60">· click to retry</span>}
            </button>
            <div className="flex items-center gap-2 bg-surface border border-border px-3 py-1 rounded-full">
              <span className="text-xs text-muted">Dataset:</span>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="bg-transparent text-sm font-medium text-accent outline-none cursor-pointer"
              >
                {['FD001', 'FD002', 'FD003', 'FD004'].map((ds) => (
                  <option key={ds} value={ds} className="bg-surface text-text">
                    {ds}
                  </option>
                ))}
              </select>
            </div>
            <span className="text-xs bg-accent/10 text-accent border border-accent/30 px-2 py-0.5 rounded-full hidden sm:inline-block">
              {engineIds.length} engines
            </span>
          </div>
        </div>

        {/* Tab bar */}
        <div className="max-w-7xl mx-auto px-4 flex gap-0 border-t border-border/50">
          {[
            { id: 'dashboard',  label: 'Dashboard' },
            { id: 'simulation', label: 'Simulation' },
            { id: 'analysis',   label: 'Analysis' },
          ].map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors
                ${tab === id
                  ? 'border-accent text-accent'
                  : 'border-transparent text-muted hover:text-text-secondary'}`}
            >
              {label}
            </button>
          ))}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">

        {/* ── Setup banner (only when API offline & not dismissed) ─────── */}
        {!apiOk && showBanner && (
          <SetupBanner
            onRetry={loadApiData}
            onDismiss={() => setShowBanner(false)}
          />
        )}

        {/* ── DASHBOARD TAB ─────────────────────────────────────────────── */}
        {tab === 'dashboard' && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2">
                <InputPanel
                  engineIds={engineIds}
                  selectedEngine={selectedEngine}
                  onEngineChange={setSelectedEngine}
                  cycle={cycle}
                  onCycleChange={setCycle}
                  maxCycle={maxCycle}
                  onPredict={handlePredict}
                  onUpload={handleUpload}
                  loading={predLoading || uploadLoading}
                  model={model}
                  onModelChange={setModel}
                />
              </div>
              <div>
                <PredictionCard prediction={prediction} loading={predLoading} />
                {predError && (
                  <p className="text-danger text-xs mt-2 px-1 leading-relaxed">{predError}</p>
                )}
              </div>
            </div>

            {batchResults && <BatchResultsTable results={batchResults} />}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <RULChart predictions={allPreds} highlightedEngine={selectedEngine} />
              <SensorChart engineData={engineData} />
            </div>
          </>
        )}

        {/* ── SIMULATION TAB ────────────────────────────────────────────── */}
        {tab === 'simulation' && (
          <SimulationPanel engineIds={engineIds} dataset={dataset} />
        )}

        {/* ── ANALYSIS TAB ─────────────────────────────────────────────── */}
        {tab === 'analysis' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="card">
                <p className="section-title">Model Comparison</p>
                <img
                  src={`http://localhost:8000/plots/${dataset}/model_comparison.png`}
                  alt="Model Comparison"
                  className="w-full h-auto rounded mt-2 border border-border"
                  onError={(e) => e.target.style.display = 'none'}
                />
              </div>
              <div className="card">
                <p className="section-title">Training Loss Curves</p>
                <img
                  src={`http://localhost:8000/plots/${dataset}/loss_curve.png`}
                  alt="Loss Curve"
                  className="w-full h-auto rounded mt-2 border border-border"
                  onError={(e) => e.target.style.display = 'none'}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="card">
                <p className="section-title">Actual vs Predicted RUL</p>
                <img
                  src={`http://localhost:8000/plots/${dataset}/prediction_vs_actual.png`}
                  alt="Prediction vs Actual"
                  className="w-full h-auto rounded mt-2 border border-border"
                  onError={(e) => e.target.style.display = 'none'}
                />
              </div>
              <div className="card">
                <p className="section-title">Error Distribution</p>
                <img
                  src={`http://localhost:8000/plots/${dataset}/error_distribution.png`}
                  alt="Error Distribution"
                  className="w-full h-auto rounded mt-2 border border-border"
                  onError={(e) => e.target.style.display = 'none'}
                />
              </div>
            </div>

            {allPreds.length > 0 && (
              <div className="card">
                <p className="section-title">RUL Scatter — Actual vs Predicted (All Engines)</p>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-2">
                  {allPreds.slice(0, 8).map((p) => (
                    <div key={p.engine_id} className="bg-bg rounded-lg p-3 border border-border">
                      <p className="text-xs text-muted mb-1">Engine {p.engine_id}</p>
                      <p className="font-mono text-sm">
                        A: <span className="text-success">{p.actual_rul}</span>
                      </p>
                      <p className="font-mono text-sm">
                        P: <span className="text-accent">{Math.round(p.predicted_rul)}</span>
                      </p>
                      <span className={`text-xs px-1.5 py-0.5 rounded-full mt-1 inline-block
                        ${p.health_color === 'healthy' ? 'badge-healthy'
                          : p.health_color === 'warning' ? 'badge-warning'
                          : 'badge-critical'}`}>
                        {p.health_label}
                      </span>
                    </div>
                  ))}
                </div>
                {allPreds.length > 8 && (
                  <p className="text-xs text-muted mt-3">Showing 8 of {allPreds.length} engines.</p>
                )}
              </div>
            )}
          </div>
        )}
      </main>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <footer className="border-t border-border mt-12 py-4">
        <p className="text-center text-xs text-muted">
          Predictive Maintenance System · NASA CMAPSS FD001 · Hybrid CNN-LSTM
        </p>
      </footer>
    </div>
  );
}
