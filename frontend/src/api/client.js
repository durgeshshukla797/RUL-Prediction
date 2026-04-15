// src/api/client.js
// Central Axios client — all API calls go through here
// Updated to support purely generalized endpoints without dataset dependencies

import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,  // 60s for large files / slow CPU inference
});

// ── Engine endpoints ──────────────────────────────────────────────────────────

export const fetchEngines = () =>
  api.get('/engines').then((r) => r.data);

export const fetchEngineData = (engineId) =>
  api.get(`/engine/${engineId}`).then((r) => r.data);

export const predictEngineAtCycle = (engineId, cycle) =>
  api.get(`/engine/${engineId}/predict/${cycle}`)
    .then((r) => r.data);

// ── Prediction endpoints ──────────────────────────────────────────────────────

export const predict = (engineId, cycle) =>
  api.post('/predict', { engine_id: engineId, cycle }).then((r) => r.data);

export const uploadCsv = (file) => {
  const form = new FormData();
  form.append('file', file);
  return api.post(`/upload`, form).then((r) => r.data);
};

// ── Metrics & History ─────────────────────────────────────────────────────────

export const fetchMetrics = () =>
  api.get('/metrics').then((r) => r.data);

export const fetchHistory = () =>
  api.get('/history').then((r) => r.data);

export const fetchPredictions = () =>
  api.get('/predictions').then((r) => r.data);

export const fetchBestModelInfo = () =>
  api.get('/best_model_info').then((r) => r.data);

export const fetchPerDatasetMetrics = () =>
  api.get('/per_dataset_metrics').then((r) => r.data);

export default api;
