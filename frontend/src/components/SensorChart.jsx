// src/components/SensorChart.jsx
// Sensor trend for a selected engine with a sensor selector

import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts';

const SENSOR_LABELS = {
  c1: 'Op. Cond. 1', c2: 'Op. Cond. 2', c3: 'Op. Cond. 3',
  s2: 'Fan Inlet Temp (°R)', s3: 'LPC Outlet Temp (°R)', s4: 'HPC Outlet Temp (°R)',
  s6: 'Total Temp at Fan Inlet', s7: 'Bypass Ratio',
  s8: 'Bleed Enthalpy', s9: 'Physical Fan Speed',
  s11: 'HPC Outlet Static Pressure', s12: 'Fan Static Pressure',
  s13: 'Corrected Fan Speed', s14: 'Corrected Core Speed',
  s15: 'BPR', s17: 'HPT Coolant Bleed', s20: 'LPT Efficiency',
  s21: 'HPC Outlet Humidity',
};

const CustomTooltip = ({ active, payload, label, sensor }) => {
  if (!active || !payload?.[0]) return null;
  return (
    <div className="bg-surface border border-border rounded-lg px-3 py-2 text-xs shadow-lg">
      <p className="text-muted mb-0.5">Cycle {label}</p>
      <p className="font-mono text-accent">{Number(payload[0].value).toFixed(4)}</p>
    </div>
  );
};

export default function SensorChart({ engineData }) {
  const features = engineData?.features || [];
  const [selectedSensor, setSelectedSensor] = useState(features[0] || '');

  const chartData = useMemo(() => {
    if (!engineData || !selectedSensor) return [];
    const cycles = engineData.cycles || [];
    const values = engineData.sensors?.[selectedSensor] || [];
    return cycles.map((c, i) => ({ cycle: c, value: values[i] ?? null }));
  }, [engineData, selectedSensor]);

  if (!engineData) {
    return (
      <div className="card">
        <p className="section-title">Sensor Trend</p>
        <div className="flex items-center justify-center h-48 text-muted text-sm">
          Select an engine to see sensor trends.
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <p className="section-title mb-0">Sensor Trend — Engine {engineData.engine_id}</p>
        <select
          id="sensor-select"
          className="input-base w-auto text-xs"
          value={selectedSensor}
          onChange={(e) => setSelectedSensor(e.target.value)}
        >
          {features.map((f) => (
            <option key={f} value={f}>
              {f} {SENSOR_LABELS[f] ? `— ${SENSOR_LABELS[f]}` : ''}
            </option>
          ))}
        </select>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" />
          <XAxis
            dataKey="cycle"
            tick={{ fill: '#6B7280', fontSize: 10 }}
            label={{ value: 'Cycle', position: 'insideBottom', offset: -2, fill: '#6B7280', fontSize: 11 }}
          />
          <YAxis tick={{ fill: '#6B7280', fontSize: 10 }} domain={['auto', 'auto']} />
          <Tooltip content={<CustomTooltip sensor={selectedSensor} />} />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#3B82F6"
            dot={false}
            strokeWidth={2}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>

      <p className="text-xs text-muted mt-2">
        {SENSOR_LABELS[selectedSensor] || selectedSensor} · {chartData.length} cycles
      </p>
    </div>
  );
}
