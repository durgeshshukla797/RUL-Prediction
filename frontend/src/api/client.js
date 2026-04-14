// src/api/client.js
// Central Axios client — all API calls go through here

import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,  // 60s for large files / slow CPU inference
});

// ── Engine endpoints ──────────────────────────────────────────────────────────

export const fetchEngines = (dataset = 'FD001') =>
  api.get('/engines', { params: { dataset } }).then((r) => r.data);

export const fetchEngineData = (engineId, dataset = 'FD001') =>
  api.get(`/engine/${engineId}`, { params: { dataset } }).then((r) => r.data);

export const predictEngineAtCycle = (engineId, cycle, dataset = 'FD001') =>
  api.get(`/engine/${engineId}/predict/${cycle}`, { params: { dataset } })
    .then((r) => r.data);

// ── Prediction endpoints ──────────────────────────────────────────────────────

export const predict = (engineId, cycle, dataset = 'FD001') =>
  api.post('/predict', { engine_id: engineId, cycle, dataset }).then((r) => r.data);

export const uploadCsv = (file, dataset = 'FD001') => {
  const form = new FormData();
  form.append('file', file);
  return api.post(`/upload`, form, { params: { dataset } }).then((r) => r.data);
};

// ── Metrics & History ─────────────────────────────────────────────────────────

export const fetchMetrics = (dataset = 'FD001') =>
  api.get('/metrics', { params: { dataset } }).then((r) => r.data);

export const fetchHistory = (dataset = 'FD001') =>
  api.get('/history', { params: { dataset } }).then((r) => r.data);

export const fetchPredictions = (dataset = 'FD001') =>
  api.get('/predictions', { params: { dataset } }).then((r) => r.data);

export const fetchBestModelInfo = (dataset = 'FD001') =>
  api.get('/best_model_info', { params: { dataset } }).then((r) => r.data);

export default api;
