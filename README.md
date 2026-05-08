# 🔬 UrineScope AI — Urinalysis Disease Prediction System

> AI-powered urinalysis screening tool using Random Forest with SMOTE, 5-Fold CV, and hyperparameter tuning.  
> **Recall: 94.76% | F1: 95.98% | ROC-AUC: 97.97%**

---

## 📁 Project Structure

```
urinalysis_ml/
├── backend/
│   ├── rojo_urinalysis.csv       # Original dataset (1,000 samples)
│   ├── ml_pipeline.py            # Full ML pipeline (EDA→SMOTE→CV→Tuning)
│   ├── app.py                    # Flask REST API
│   ├── requirements.txt
│   ├── render.yaml               # Render.com deployment config
│   ├── models/
│   │   ├── best_model.pkl        # Trained Random Forest (tuned)
│   │   ├── scaler.pkl            # StandardScaler
│   │   ├── feature_names.json    # 21 feature names
│   │   └── metadata.json        # Model metrics & comparison
│   └── eda_plots/                # 12 EDA visualizations
│
└── frontend/
    ├── src/
    │   ├── App.jsx               # Main app with tabbed navigation
    │   ├── App.css               # Clinical dark theme
    │   └── components/
    │       ├── PredictForm.jsx   # 14-field urinalysis input form
    │       ├── ResultCard.jsx    # Prediction result with risk level
    │       ├── ModelDashboard.jsx # 5-model comparison dashboard
    │       └── EDAGallery.jsx    # 12-plot EDA gallery
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── vercel.json               # Vercel deployment config
```

---

## 🧠 ML Pipeline

### Dataset
- **Source**: rojo_urinalysis.csv
- **Samples**: 1,000 | **Features**: 14 + 7 engineered = **21 total**
- **Target**: Diagnosis (0=Negative 63%, 1=Positive 37%)
- **Missing values**: None

### Feature Engineering
| Engineered Feature | Formula | Clinical Rationale |
|---|---|---|
| `Infection_Score` | WBC + Bacteria + RBC | Combined infection indicator |
| `Sediment_Score` | Epithelial + Mucous + Urates | Sediment burden |
| `pH_Abnormal` | pH < 4.5 or > 8.5 | Critical pH flag |
| `SG_Abnormal` | SG < 1.005 or > 1.030 | Concentration anomaly |
| `Has_Protein` | Protein > 0 | Proteinuria binary flag |
| `Has_Glucose` | Glucose > 0 | Glucosuria binary flag |
| `Age_Group` | Bins: 0–12/13–18/19–40/41–65/65+ | Pediatric vs adult risk |

### Class Imbalance → SMOTE
- Before: {Negative: 630, Positive: 370}
- After:  {Negative: 630, Positive: 630} ← balanced

### Cross-Validation
- **Strategy**: StratifiedKFold (n_splits=5, shuffle=True)
- Ensures class balance is preserved in each fold

### 5-Model Comparison

| Model | Accuracy | Recall ⭐ | F1 ⭐ | Precision | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** 🏆 | **95.95%** | **94.44%** | **95.85%** | **97.41%** | **98.10%** |
| Gradient Boosting | 95.79% | 94.44% | 95.69% | 97.12% | 97.84% |
| Decision Tree | 91.27% | 91.43% | 91.23% | 91.19% | 91.27% |
| SVM | 87.86% | 86.83% | 87.67% | 88.75% | 95.13% |
| Logistic Regression | 81.11% | 80.00% | 80.86% | 81.98% | 88.34% |

> ⭐ **Recall & F1 are primary metrics** — in medical screening, false negatives (missed disease) are more costly than false positives.

### Hyperparameter Tuning (Random Forest)
GridSearchCV with F1 scoring over 5-fold CV:
```python
{
  "n_estimators": [100, 200, 300],
  "max_depth": [None, 10, 20],
  "min_samples_split": [2, 5],
  "min_samples_leaf": [1, 2],
  "class_weight": ["balanced", None]
}
```
**Best**: `{class_weight: balanced, max_depth: None, n_estimators: 200, ...}`

### Final Metrics (Tuned Model, 5-Fold CV Predict)
| Metric | Score |
|---|---|
| **Recall** ⭐ | **94.76%** |
| **F1 Score** ⭐ | **95.98%** |
| Accuracy | 96.03% |
| Precision | 97.23% |
| ROC-AUC | 97.97% |

---

## 🚀 Deployment

### Backend (Render.com)
1. Push `backend/` to a GitHub repository
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set:
   - **Build Command**: `pip install -r requirements.txt && python ml_pipeline.py`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Note your Render URL: `https://your-app.onrender.com`

### Frontend (Vercel)
1. Push `frontend/` to GitHub
2. Import project on [vercel.com](https://vercel.com)
3. Set environment variable: `VITE_API_URL=https://your-app.onrender.com`
4. Deploy — Vercel auto-detects Vite

### Local Development
```bash
# Backend
cd backend
pip install -r requirements.txt
python ml_pipeline.py          # Train & save model
python app.py                  # Start Flask on :5000

# Frontend
cd frontend
npm install
npm run dev                    # Start Vite on :5173
```

---

## 🔌 API Reference

### `POST /predict`
```json
// Request
{
  "Age": 32, "Gender": 0, "Color": 4, "Transparency": 3,
  "Glucose": 0, "Protein": 1, "pH": 8.0, "Specific Gravity": 1.025,
  "WBC": 8, "RBC": 3, "Epithelial Cells": 2,
  "Mucous Threads": 1, "Amorphous Urates": 0, "Bacteria": 5
}

// Response
{
  "prediction": 1,
  "label": "Positive",
  "confidence": 87.5,
  "positive_probability": 87.5,
  "risk_level": "High Risk",
  "clinical_flags": ["Elevated WBC (possible infection)", "High bacteria count"],
  "disclaimer": "⚠️ AI screening tool only..."
}
```

### `GET /metadata`
Returns model info, comparison table, best params, feature list.

### `GET /eda/<filename>`
Serves EDA plot PNG images (e.g., `/eda/09_confusion_matrix.png`)

### `GET /eda_list`
Returns list of available EDA plot filenames.

---

## ⚠️ Medical Disclaimer
This tool is for **educational and screening purposes only**. It does not constitute medical advice or diagnosis. All results must be reviewed by a licensed healthcare professional.
