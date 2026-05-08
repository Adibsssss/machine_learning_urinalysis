export default function ModelDashboard({ metadata }) {
  if (!metadata) return (
    <div className="card" style={{ textAlign: "center", padding: "3rem" }}>
      <div className="spinner" style={{ margin: "0 auto" }} />
      <p style={{ color: "var(--text-secondary)", marginTop: "1rem" }}>Loading model data…</p>
    </div>
  );

  const { best_model_name, best_params, final_metrics, model_comparison } = metadata;
  const models = Object.entries(model_comparison);

  const metricsOrder = ["Accuracy", "Recall", "F1", "Precision", "ROC-AUC"];

  return (
    <div>
      <p className="section-title">Model Intelligence</p>
      <h2 className="section-heading">Performance Dashboard</h2>

      {/* ── Summary Stats ── */}
      <div className="dashboard-grid" style={{ marginBottom: "1.5rem" }}>
        {[
          { label: "Best Model",   value: best_model_name.split(" ")[0], sub: best_model_name },
          { label: "Recall ⭐",     value: `${(final_metrics.Recall*100).toFixed(1)}%`,   sub: "Medical priority metric" },
          { label: "F1 Score ⭐",   value: `${(final_metrics.F1*100).toFixed(1)}%`,       sub: "Harmonic mean P/R" },
          { label: "ROC-AUC",      value: `${(final_metrics["ROC-AUC"]*100).toFixed(1)}%`, sub: "Discriminative power" },
          { label: "Accuracy",     value: `${(final_metrics.Accuracy*100).toFixed(1)}%`,  sub: "Overall correctness" },
          { label: "Validation",   value: "5-Fold",  sub: "Stratified K-Fold CV" },
          { label: "Imbalance Fix",value: "SMOTE",   sub: "Synthetic oversampling" },
          { label: "Train Samples",value: "1,260",   sub: "After SMOTE (balanced)" },
        ].map(({ label, value, sub }) => (
          <div key={label} className="stat-card">
            <span className="s-label">{label}</span>
            <span className="s-value" style={{ fontSize: value.length > 6 ? "1.1rem" : "1.8rem" }}>{value}</span>
            <span className="s-sub">{sub}</span>
          </div>
        ))}
      </div>

      {/* ── Model Comparison Table ── */}
      <div className="card" style={{ marginBottom: "1.5rem" }}>
        <p className="section-title" style={{ marginBottom: "0.75rem" }}>
          5-Model Comparison — 5-Fold Cross-Validation (SMOTE applied)
        </p>
        <div style={{ overflowX: "auto" }}>
          <table className="model-table">
            <thead>
              <tr>
                <th>Model</th>
                {metricsOrder.map(m => (
                  <th key={m}>
                    {m} {m === "Recall" || m === "F1" ? "⭐" : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.sort((a, b) => b[1].F1 - a[1].F1).map(([name, scores]) => (
                <tr key={name} className={name === best_model_name ? "best-row" : ""}>
                  <td style={{ fontFamily: "var(--font-main)", fontWeight: 500 }}>{name}</td>
                  {metricsOrder.map(m => (
                    <td key={m} style={{ color: (m==="Recall"||m==="F1") && name===best_model_name ? "var(--accent)" : undefined }}>
                      {(scores[m]*100).toFixed(2)}%
                      <span
                        className="metric-bar"
                        style={{ width: `${scores[m]*40}px` }}
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p style={{ marginTop: "0.75rem", fontSize: "0.75rem", color: "var(--text-muted)" }}>
          ⭐ Recall & F1 are prioritized for this medical dataset to minimize false negatives (missed diagnoses).
        </p>
      </div>

      {/* ── Best params ── */}
      <div className="card">
        <p className="section-title">Hyperparameter Tuning</p>
        <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "1rem" }}>
          {best_model_name} — GridSearchCV Best Parameters
        </h3>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
          {Object.entries(best_params).map(([k, v]) => (
            <div key={k} style={{
              background: "var(--bg-elevated)",
              border: "1px solid rgba(0,212,255,0.2)",
              borderRadius: "6px",
              padding: "0.5rem 0.9rem",
              fontFamily: "var(--font-mono)",
              fontSize: "0.82rem"
            }}>
              <span style={{ color: "var(--text-secondary)" }}>{k}: </span>
              <span style={{ color: "var(--accent)" }}>{String(v)}</span>
            </div>
          ))}
        </div>

        {/* Pipeline steps */}
        <div style={{ marginTop: "1.5rem" }}>
          <p className="section-title">ML Pipeline Steps</p>
          <div style={{ display: "flex", gap: "0", alignItems: "center", flexWrap: "wrap", marginTop: "0.5rem" }}>
            {[
              "Load CSV",
              "EDA",
              "Feature Engineering (+7)",
              "StandardScaler",
              "SMOTE",
              "5-Fold CV",
              "5-Model Compare",
              "GridSearchCV",
              "Final Model",
            ].map((step, i, arr) => (
              <div key={step} style={{ display: "flex", alignItems: "center" }}>
                <div style={{
                  background: i === arr.length - 1 ? "rgba(0,212,255,0.15)" : "var(--bg-elevated)",
                  border: `1px solid ${i === arr.length - 1 ? "var(--accent)" : "var(--border)"}`,
                  borderRadius: "6px",
                  padding: "0.4rem 0.7rem",
                  fontSize: "0.75rem",
                  color: i === arr.length - 1 ? "var(--accent)" : "var(--text-secondary)",
                  whiteSpace: "nowrap"
                }}>{step}</div>
                {i < arr.length - 1 && (
                  <span style={{ color: "var(--text-muted)", margin: "0 4px", fontSize: "0.8rem" }}>→</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
