import { useState, useEffect } from "react";
import PredictForm from "./components/PredictForm";
import ResultCard from "./components/ResultCard";
import ModelDashboard from "./components/ModelDashboard";
import EDAGallery from "./components/EDAGallery";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:5000";

export default function App() {
  const [tab, setTab] = useState("predict");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [metadata, setMetadata] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/metadata`)
      .then(r => r.json())
      .then(setMetadata)
      .catch(() => {});
  }, []);

  const handlePredict = async (formData) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      setResult(data);
      setTab("result");
    } catch (e) {
      setError("Could not reach the server. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-inner">
          <div className="logo-block">
            <span className="logo-icon">🔬</span>
            <div>
              <h1 className="logo-title">UrineScope AI</h1>
              <p className="logo-sub">Urinalysis Disease Prediction System</p>
            </div>
          </div>
          <nav className="nav-pills">
            {[
              { id: "predict", label: "Predict", icon: "🧪" },
              { id: "result",  label: "Result",  icon: "📋", disabled: !result },
              { id: "dashboard", label: "Models", icon: "📊" },
              { id: "eda",     label: "EDA",     icon: "📈" },
            ].map(t => (
              <button
                key={t.id}
                className={`nav-pill ${tab === t.id ? "active" : ""} ${t.disabled ? "disabled" : ""}`}
                onClick={() => !t.disabled && setTab(t.id)}
              >
                <span>{t.icon}</span> {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* ── Banner ── */}
      <div className="disclaimer-banner">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational/screening purposes only.
        Always consult a licensed healthcare professional for diagnosis.
      </div>

      {/* ── Main Content ── */}
      <main className="main-content">
        {tab === "predict" && (
          <PredictForm onSubmit={handlePredict} loading={loading} error={error} />
        )}
        {tab === "result" && result && (
          <ResultCard result={result} onBack={() => setTab("predict")} />
        )}
        {tab === "dashboard" && (
          <ModelDashboard metadata={metadata} api={API} />
        )}
        {tab === "eda" && (
          <EDAGallery api={API} />
        )}
      </main>

      <footer className="app-footer">
        <p>UrineScope AI • Powered by Random Forest (F1: 0.9598 | Recall: 0.9476 | AUC: 0.9797)</p>
        <p>Dataset: rojo_urinalysis.csv • 1,000 samples • SMOTE + 5-Fold CV</p>
      </footer>
    </div>
  );
}
