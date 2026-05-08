import { useState } from "react";

const FIELD_CONFIG = [
  {
    key: "Age",
    label: "Age",
    type: "number",
    min: 0,
    max: 120,
    step: 0.1,
    hint: "Patient age (years)",
    default: 25,
  },
  {
    key: "Gender",
    label: "Gender",
    type: "select",
    options: [
      ["0", "Female"],
      ["1", "Male"],
    ],
    default: "1",
  },
  {
    key: "Color",
    label: "Urine Color",
    type: "select",
    options: [
      ["1", "Colorless"],
      ["2", "Pale Yellow"],
      ["3", "Yellow"],
      ["4", "Dark Yellow"],
      ["5", "Amber"],
      ["6", "Orange"],
      ["7", "Brown"],
      ["8", "Red"],
    ],
    default: "3",
  },
  {
    key: "Transparency",
    label: "Transparency",
    type: "select",
    options: [
      ["1", "Clear"],
      ["2", "Slightly Cloudy"],
      ["3", "Cloudy"],
      ["4", "Turbid"],
      ["5", "Very Turbid"],
    ],
    default: "1",
  },
  {
    key: "Glucose",
    label: "Glucose",
    type: "number",
    min: 0,
    max: 5,
    step: 1,
    hint: "0=Neg, 1=Trace, 2=1+, 3=2+, 4=3+, 5=4+",
    default: 0,
  },
  {
    key: "Protein",
    label: "Protein",
    type: "number",
    min: 0,
    max: 5,
    step: 1,
    hint: "0=Neg, 1=Trace, 2=1+, 3=2+, 4=3+, 5=4+",
    default: 0,
  },
  {
    key: "pH",
    label: "pH",
    type: "number",
    min: 4.0,
    max: 9.0,
    step: 0.1,
    hint: "Normal: 4.5–8.0",
    default: 6.0,
  },
  {
    key: "Specific Gravity",
    label: "Specific Gravity",
    type: "number",
    min: 1.0,
    max: 1.04,
    step: 0.001,
    hint: "Normal: 1.005–1.030",
    default: 1.015,
  },
  {
    key: "WBC",
    label: "WBC (cells/hpf)",
    type: "number",
    min: 0,
    max: 20,
    step: 1,
    hint: "White blood cells",
    default: 0,
  },
  {
    key: "RBC",
    label: "RBC (cells/hpf)",
    type: "number",
    min: 0,
    max: 20,
    step: 1,
    hint: "Red blood cells",
    default: 0,
  },
  {
    key: "Epithelial Cells",
    label: "Epithelial Cells",
    type: "number",
    min: 0,
    max: 10,
    step: 1,
    hint: "0=None, 1–3=Few/Mod",
    default: 0,
  },
  {
    key: "Mucous Threads",
    label: "Mucous Threads",
    type: "number",
    min: 0,
    max: 5,
    step: 1,
    hint: "0=Absent",
    default: 0,
  },
  {
    key: "Amorphous Urates",
    label: "Amorphous Urates",
    type: "number",
    min: 0,
    max: 5,
    step: 1,
    hint: "0=Absent",
    default: 0,
  },
  {
    key: "Bacteria",
    label: "Bacteria",
    type: "number",
    min: 1,
    max: 6,
    step: 1,
    hint: "1=None, 2=Rare, 3=Few, 4=Mod, 5=Many, 6=Loaded",
    default: 2,
  },
];

const PRESETS = {
  "Healthy Adult": {
    Age: 28,
    Gender: "1",
    Color: "3",
    Transparency: "1",
    Glucose: 0,
    Protein: 0,
    pH: 6.0,
    "Specific Gravity": 1.02,
    WBC: 1,
    RBC: 0,
    "Epithelial Cells": 0,
    "Mucous Threads": 0,
    "Amorphous Urates": 0,
    Bacteria: 2,
  },
  "UTI Suspect": {
    Age: 32,
    Gender: "0",
    Color: "4",
    Transparency: "3",
    Glucose: 0,
    Protein: 1,
    pH: 8.0,
    "Specific Gravity": 1.025,
    WBC: 8,
    RBC: 3,
    "Epithelial Cells": 2,
    "Mucous Threads": 1,
    "Amorphous Urates": 0,
    Bacteria: 5,
  },
};

export default function PredictForm({ onSubmit, loading, error }) {
  const [form, setForm] = useState(
    Object.fromEntries(FIELD_CONFIG.map((f) => [f.key, f.default])),
  );

  const handleChange = (key, val) => setForm((p) => ({ ...p, [key]: val }));

  const applyPreset = (name) => setForm({ ...PRESETS[name] });

  const handleSubmit = (e) => {
    e.preventDefault();
    const numeric = Object.fromEntries(
      Object.entries(form).map(([k, v]) => [k, parseFloat(v)]),
    );
    onSubmit(numeric);
  };

  return (
    <div>
      <div className="card" style={{ marginBottom: "1.5rem" }}>
        <p className="section-title">Quick Fill</p>
        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          {Object.keys(PRESETS).map((name) => (
            <button
              key={name}
              className="btn-secondary"
              onClick={() => applyPreset(name)}
              type="button"
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="card">
          <p className="section-title">Urinalysis Input</p>
          <h2 className="section-heading">Patient Data Entry</h2>

          <div className="form-grid">
            {FIELD_CONFIG.map((f) => (
              <div className="form-group" key={f.key}>
                <label>{f.label}</label>
                {f.type === "select" ? (
                  <select
                    value={form[f.key]}
                    onChange={(e) => handleChange(f.key, e.target.value)}
                  >
                    {f.options.map(([v, l]) => (
                      <option key={v} value={v}>
                        {l}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    value={form[f.key]}
                    onChange={(e) => handleChange(f.key, e.target.value)}
                    required
                  />
                )}
                {f.hint && <p className="hint">{f.hint}</p>}
              </div>
            ))}
          </div>

          <div
            style={{
              marginTop: "2rem",
              display: "flex",
              gap: "1rem",
              alignItems: "center",
            }}
          >
            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? <span className="spinner" /> : "🔍"}
              {loading ? "Analyzing..." : "Run Prediction"}
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() =>
                setForm(
                  Object.fromEntries(
                    FIELD_CONFIG.map((f) => [f.key, f.default]),
                  ),
                )
              }
            >
              Reset
            </button>
          </div>

          {error && <div className="error-box">❌ {error}</div>}
        </div>
      </form>
    </div>
  );
}
