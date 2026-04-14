// src/components/InputPanel.jsx
// Left: Engine / cycle selector | Right: CSV upload

import React, { useState, useRef } from 'react';

export default function InputPanel({
  engineIds,
  selectedEngine,
  onEngineChange,
  cycle,
  onCycleChange,
  maxCycle,
  onPredict,
  onUpload,
  loading,
  model,
  onModelChange,
}) {
  const [dragOver, setDragOver] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const fileRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) setUploadFile(f);
  };

  const handleFileChange = (e) => {
    setUploadFile(e.target.files[0] || null);
  };

  const handleUploadSubmit = () => {
    if (uploadFile) onUpload(uploadFile);
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* ── Left: Manual prediction ─────────────────────────────── */}
      <div className="card">
        <p className="section-title">Engine Prediction</p>

        <div className="space-y-4">
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
                <option key={id} value={id}>Engine {id}</option>
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
            <div className="flex gap-2">
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
              {maxCycle > 0 && (
                <input
                  type="range"
                  min={1}
                  max={maxCycle}
                  value={cycle}
                  onChange={(e) => onCycleChange(Number(e.target.value))}
                  className="w-full accent-accent"
                />
              )}
            </div>
          </div>

          {/* Model selector */}
          <div>
            <label className="block text-xs text-muted mb-1.5">Model</label>
            <div className="flex gap-2">
              {['hybrid', 'lstm', 'cnn'].map((m) => (
                <button
                  key={m}
                  onClick={() => onModelChange(m)}
                  className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition-colors
                    ${model === m
                      ? 'bg-accent border-accent text-white'
                      : 'border-border text-muted hover:border-accent hover:text-accent'}`}
                >
                  {m.toUpperCase()}
                </button>
              ))}
            </div>
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

      {/* ── Right: CSV upload ────────────────────────────────────── */}
      <div className="card">
        <p className="section-title">Batch Upload</p>
        <p className="text-xs text-muted mb-4">
          Upload a CMAPSS-format .txt or .csv file to get predictions for all engines.
        </p>

        {/* Drop zone */}
        <div
          id="drop-zone"
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors
            ${dragOver
              ? 'border-accent bg-accent/10'
              : 'border-border hover:border-accent/50'}`}
        >
          <div className="flex flex-col items-center gap-2">
            <svg viewBox="0 0 24 24" className="w-8 h-8 text-muted" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
            {uploadFile ? (
              <span className="text-sm text-accent font-medium">{uploadFile.name}</span>
            ) : (
              <>
                <span className="text-sm text-muted">Drop file here or click to browse</span>
                <span className="text-xs text-muted/60">Supports .txt • .csv (CMAPSS format)</span>
              </>
            )}
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".txt,.csv"
            className="hidden"
            onChange={handleFileChange}
          />
        </div>

        <button
          id="upload-btn"
          className="btn-primary w-full mt-4"
          onClick={handleUploadSubmit}
          disabled={!uploadFile || loading}
        >
          {loading ? 'Processing…' : 'Upload & Predict'}
        </button>
      </div>
    </div>
  );
}
