export default function ResultCard({ result, onBack }) {
  if (!result) return null;

  const isPositive = result.prediction === 1;
  const riskClass  = {
    "High Risk":     "risk-high",
    "Moderate Risk": "risk-moderate",
    "Low Risk":      "risk-low",
    "Minimal Risk":  "risk-minimal",
  }[result.risk_level] || "risk-minimal";

  const prob = result.positive_probability;

  return (
    <div style={{ maxWidth: "720px", margin: "0 auto" }}>
      {/* ── Verdict ── */}
      <div className="card" style={{ marginBottom: "1.25rem" }}>
        <div className="result-hero">
          <p className="section-title">Prediction Result</p>
          <div className={`result-verdict ${isPositive ? "positive" : "negative"}`}>
            {isPositive ? "⚠ POSITIVE" : "✓ NEGATIVE"}
          </div>
          <div className={`result-risk-badge ${riskClass}`}>
            {result.risk_level}
          </div>

          {/* Probability bar */}
          <div className="prob-bar-wrap" style={{ maxWidth: "480px", margin: "1.5rem auto 0.5rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "0.4rem" }}>
              <span>Negative Probability</span>
              <span>Positive Probability</span>
            </div>
            <div className="prob-bar-track">
              <div
                className={`prob-bar-fill ${isPositive ? "positive" : "negative"}`}
                style={{ width: `${prob}%` }}
              />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", marginTop: "0.3rem", fontFamily: "var(--font-mono)" }}>
              <span style={{ color: "var(--negative)" }}>{(100 - prob).toFixed(1)}%</span>
              <span style={{ color: "var(--positive)" }}>{prob.toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Metrics row */}
        <div className="metrics-row">
          <div className="metric-tile">
            <div className="val">{prob.toFixed(1)}%</div>
            <div className="lbl">Positive Prob.</div>
          </div>
          <div className="metric-tile">
            <div className="val">{result.confidence.toFixed(1)}%</div>
            <div className="lbl">Model Confidence</div>
          </div>
          <div className="metric-tile">
            <div className="val" style={{ fontSize: "1.1rem" }}>{result.label}</div>
            <div className="lbl">Classification</div>
          </div>
        </div>
      </div>

      {/* ── Clinical flags ── */}
      {result.clinical_flags && result.clinical_flags.length > 0 && (
        <div className="card" style={{ marginBottom: "1.25rem" }}>
          <p className="section-title">Clinical Flags</p>
          <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.5rem" }}>
            Detected Abnormalities
          </h3>
          <ul className="flags-list">
            {result.clinical_flags.map((f, i) => (
              <li key={i} className="flag-item">
                <span>⚠</span> {f}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── Model info ── */}
      <div className="card" style={{ marginBottom: "1.25rem" }}>
        <p className="section-title">Model Performance</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: "0.75rem" }}>
          {[["Recall","94.76%"],["F1 Score","95.98%"],["Accuracy","96.03%"],["ROC-AUC","97.97%"]].map(([k,v]) => (
            <div key={k} style={{ textAlign: "center", background: "var(--bg-elevated)", borderRadius: "8px", padding: "0.75rem", border: "1px solid var(--border)" }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.2rem", fontWeight: 700, color: "var(--accent)" }}>{v}</div>
              <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.5px", marginTop: "0.2rem" }}>{k}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Disclaimer ── */}
      <div style={{
        background: "rgba(245,158,11,0.06)",
        border: "1px solid rgba(245,158,11,0.2)",
        borderRadius: "8px",
        padding: "0.9rem 1.1rem",
        fontSize: "0.8rem",
        color: "#fcd34d",
        marginBottom: "1.5rem",
        lineHeight: "1.6"
      }}>
        {result.disclaimer}
      </div>

      <button className="btn-primary" onClick={onBack}>
        ← New Prediction
      </button>
    </div>
  );
}
