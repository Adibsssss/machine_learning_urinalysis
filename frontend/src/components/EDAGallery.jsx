import { useState, useEffect } from "react";

const PLOT_INFO = {
  "01_class_distribution.png"   : "Class Distribution (Before SMOTE)",
  "02_correlation_heatmap.png"  : "Feature Correlation Heatmap",
  "03_feature_distributions.png": "Feature Distributions by Diagnosis Class",
  "04_boxplots.png"             : "Box Plots — Feature Spread by Class",
  "05_age_distribution.png"     : "Age Distribution by Diagnosis",
  "06_smote_balance.png"        : "Class Balance: Before vs After SMOTE",
  "07_model_comparison.png"     : "5-Model CV Comparison — All Metrics",
  "08_radar_chart.png"          : "Radar Chart — Model Comparison",
  "09_confusion_matrix.png"     : "Confusion Matrix — Best Model (Tuned)",
  "10_roc_curve.png"            : "ROC Curve — Best Model (AUC)",
  "11_feature_importance.png"   : "Feature Importances — Random Forest",
  "12_kfold_scores.png"         : "5-Fold Recall & F1 Scores",
};

export default function EDAGallery({ api }) {
  const [plots, setPlots] = useState([]);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    fetch(`${api}/eda_list`)
      .then(r => r.json())
      .then(d => setPlots(d.plots || []))
      .catch(() => setPlots(Object.keys(PLOT_INFO)));
  }, [api]);

  return (
    <div>
      <p className="section-title">Exploratory Data Analysis</p>
      <h2 className="section-heading">EDA & Visualization Gallery</h2>

      <div className="eda-gallery">
        {(plots.length ? plots : Object.keys(PLOT_INFO)).map(name => (
          <div
            key={name}
            className="eda-card"
            onClick={() => setSelected(name)}
            style={{ cursor: "pointer" }}
          >
            <img
              src={`${api}/eda/${name}`}
              alt={PLOT_INFO[name] || name}
              loading="lazy"
            />
            <div className="eda-card-caption">
              {PLOT_INFO[name] || name}
            </div>
          </div>
        ))}
      </div>

      {/* ── Lightbox ── */}
      {selected && (
        <div
          onClick={() => setSelected(null)}
          style={{
            position: "fixed", inset: 0,
            background: "rgba(0,0,0,0.85)",
            display: "flex", alignItems: "center", justifyContent: "center",
            zIndex: 999, padding: "2rem", cursor: "zoom-out"
          }}
        >
          <div onClick={e => e.stopPropagation()}
            style={{ maxWidth: "90vw", maxHeight: "90vh", background: "white", borderRadius: "10px", overflow: "hidden" }}>
            <img
              src={`${api}/eda/${selected}`}
              alt={selected}
              style={{ maxWidth: "100%", maxHeight: "80vh", display: "block" }}
            />
            <div style={{
              padding: "0.75rem 1rem",
              background: "var(--bg-card)",
              color: "var(--text-secondary)",
              fontSize: "0.85rem",
              fontFamily: "var(--font-mono)",
              display: "flex", justifyContent: "space-between"
            }}>
              <span>{PLOT_INFO[selected] || selected}</span>
              <button className="btn-secondary"
                onClick={() => setSelected(null)}
                style={{ padding: "0.25rem 0.75rem", fontSize: "0.8rem" }}>
                ✕ Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
