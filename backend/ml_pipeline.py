"""
============================================================
 Urinalysis UTI/Kidney Disease Prediction — ML Pipeline
 Dataset  : rojo_urinalysis.csv  (1000 samples, 15 features)
 Target   : Diagnosis  (0 = Negative, 1 = Positive)
 Priority : Recall & F1 (medical dataset — minimize false negatives)
 Models   : Logistic Regression, Decision Tree, Random Forest, Naive Bayes
 Tuning   : GridSearchCV scored on Recall
============================================================
"""

import warnings, os, pickle, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (classification_report, confusion_matrix,
                                       roc_auc_score, roc_curve, f1_score,
                                       recall_score, precision_score, accuracy_score)
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.naive_bayes       import GaussianNB
from imblearn.over_sampling    import SMOTE

# ────────────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
PLOTS  = os.path.join(BASE, "eda_plots")
MODELS = os.path.join(BASE, "models")

os.makedirs(PLOTS,  exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 1 — LOAD DATA")
print("="*60)

df = pd.read_csv(os.path.join(BASE, "rojo_urinalysis.csv"))

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print(f"  Shape         : {df.shape}")
print(f"  Columns       : {df.columns.tolist()}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Target dist.  :\n{df['Diagnosis'].value_counts(normalize=True).round(3)}")

# ════════════════════════════════════════════════════════
# STEP 2 — EDA
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 2 — EDA")
print("="*60)

# 2a. Class Distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["Diagnosis"].value_counts()
bars = ax.bar(["Negative (0)", "Positive (1)"], [counts[0], counts[1]],
              color=["#4C72B0", "#DD8452"], edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, [counts[0], counts[1]]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f"{val}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=10)
ax.set_title("Class Distribution (Before SMOTE)", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
ax.set_ylim(0, max(counts) * 1.2)
plt.tight_layout()
plt.savefig(f"{PLOTS}/01_class_distribution.png", dpi=120)
plt.close()

# 2b. Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS}/02_correlation_heatmap.png", dpi=120)
plt.close()

# 2c. Feature Distributions by Class
num_features = [c for c in df.columns if c != "Diagnosis"]
df_neg = df[df["Diagnosis"] == 0]
df_pos = df[df["Diagnosis"] == 1]

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()
for i, feat in enumerate(num_features):
    axes[i].hist(df_neg[feat].dropna(), bins=20, alpha=0.6, color="#4C72B0", label="Negative", edgecolor="none")
    axes[i].hist(df_pos[feat].dropna(), bins=20, alpha=0.6, color="#DD8452", label="Positive", edgecolor="none")
    axes[i].set_title(feat, fontsize=9, fontweight="bold")
    axes[i].legend(fontsize=7)
    axes[i].tick_params(labelsize=7)
for j in range(len(num_features), len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("Feature Distributions by Diagnosis", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOTS}/03_feature_distributions.png", dpi=120, bbox_inches="tight")
plt.close()

# 2d. Box Plots
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()
for i, feat in enumerate(num_features):
    df.boxplot(column=feat, by="Diagnosis", ax=axes[i],
               boxprops=dict(color="#4C72B0"),
               medianprops=dict(color="#DD8452", linewidth=2))
    axes[i].set_title(feat, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("Diagnosis")
    axes[i].tick_params(labelsize=7)
plt.suptitle("Box Plots by Diagnosis Class", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS}/04_boxplots.png", dpi=120, bbox_inches="tight")
plt.close()

# 2e. Age Distribution
fig, ax = plt.subplots(figsize=(8, 4))
for cls, color, label in [(0, "#4C72B0", "Negative"), (1, "#DD8452", "Positive")]:
    ax.hist(df[df["Diagnosis"] == cls]["Age"], bins=30, alpha=0.7,
            color=color, label=label, edgecolor="white")
ax.set_title("Age Distribution by Diagnosis", fontsize=13, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/05_age_distribution.png", dpi=120)
plt.close()

print("  EDA plots saved.")

# ════════════════════════════════════════════════════════
# STEP 3 — PREPROCESSING & FEATURE ENGINEERING
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 3 — PREPROCESSING & FEATURE ENGINEERING")
print("="*60)

X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"].astype(int)

X["Infection_Score"] = X["WBC"] + X["Bacteria"] + X["RBC"]
X["Sediment_Score"]  = X["Epithelial Cells"] + X["Mucous Threads"] + X["Amorphous Urates"]
X["pH_Abnormal"]     = ((X["pH"] < 4.5) | (X["pH"] > 8.5)).astype(int)
X["SG_Abnormal"]     = ((X["Specific Gravity"] < 1.005) | (X["Specific Gravity"] > 1.030)).astype(int)
X["Has_Protein"]     = (X["Protein"] > 0).astype(int)
X["Has_Glucose"]     = (X["Glucose"] > 0).astype(int)
X["Age_Group"]       = pd.cut(X["Age"], bins=[0, 12, 18, 40, 65, 150],
                               labels=[0, 1, 2, 3, 4]).astype(int)

print(f"  Features after engineering : {X.shape[1]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
feature_names = X.columns.tolist()

# ════════════════════════════════════════════════════════
# STEP 4 — SMOTE
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 4 — SMOTE")
print("="*60)

smote = SMOTE(random_state=SEED)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print(f"  Before SMOTE: {dict(pd.Series(y).value_counts())}")
print(f"  After  SMOTE: {dict(pd.Series(y_resampled).value_counts())}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, data, title in zip(axes, [y, y_resampled], ["Before SMOTE", "After SMOTE"]):
    counts = pd.Series(data).value_counts()
    ax.bar(["Negative", "Positive"], [counts[0], counts[1]],
           color=["#4C72B0", "#DD8452"], edgecolor="white")
    for j, c in enumerate([counts[0], counts[1]]):
        ax.text(j, c + 3, str(c), ha="center", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.2)
plt.suptitle("Class Balance: Before vs After SMOTE", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS}/06_smote_balance.png", dpi=120)
plt.close()

# ════════════════════════════════════════════════════════
# STEP 5 — 5-FOLD CV & 4-MODEL COMPARISON
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 5 — 5-FOLD CV & MODEL COMPARISON")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree"      : DecisionTreeClassifier(random_state=SEED),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=SEED),
    "Naive Bayes"        : GaussianNB(),
}

results = {}
for name, model in models.items():
    acc  = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring="accuracy").mean()
    rec  = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring="recall").mean()
    f1   = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring="f1").mean()
    prec = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring="precision").mean()
    results[name] = {"Accuracy": acc, "Recall": rec, "F1": f1, "Precision": prec}
    print(f"  {name:<22} | Acc={acc:.4f} | Rec={rec:.4f} | F1={f1:.4f} | Prec={prec:.4f}")

results_df = pd.DataFrame(results).T.round(4)
print("\n  Full comparison table:")
print(results_df.to_string())

# Model comparison bar chart
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
metrics     = ["Accuracy", "Precision", "Recall", "F1"]
colors      = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
model_names = list(models.keys())
abbrevs     = ["LR", "DT", "RF", "NB"]

for ax, metric, color in zip(axes, metrics, colors):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(range(len(model_names)), vals, color=color, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(abbrevs, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    if metric in ["Recall", "F1"]:
        ax.spines["top"].set_color("#DD8452")
        ax.spines["top"].set_linewidth(2.5)
        ax.spines["right"].set_color("#DD8452")
        ax.spines["right"].set_linewidth(2.5)

fig.suptitle("4-Model Comparison (5-Fold CV)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{PLOTS}/07_model_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# Radar Chart — 4 Models
cats   = ["Accuracy", "Precision", "Recall", "F1"]
N      = len(cats)
angles = [n / float(N) * 2 * 3.141592653589793 for n in range(N)]
angles += angles[:1]

fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111, polar=True)

palette = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

for (name, vals), color in zip(results.items(), palette):
    v = [vals[m] for m in cats] + [vals[cats[0]]]
    ax.plot(angles, v, linewidth=2, linestyle="solid", label=name, color=color)
    ax.fill(angles, v, alpha=0.10, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats, size=10, fontweight="bold")
ax.set_ylim(0, 1)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
ax.set_title("Radar Chart — 4 Models (5-Fold CV)", fontsize=13, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig(f"{PLOTS}/08_radar_chart.png", dpi=120, bbox_inches="tight")
plt.close()

# ════════════════════════════════════════════════════════
# STEP 6 — HYPERPARAMETER TUNING (Best by Recall)
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 6 — HYPERPARAMETER TUNING (Best Model by Recall)")
print("="*60)

best_name = results_df["Recall"].idxmax()
print(f"  Best model by Recall: {best_name}")

param_grids = {
    "Random Forest": {
        "n_estimators"     : [100, 200, 300],
        "max_depth"        : [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf" : [1, 2],
        "class_weight"     : ["balanced", None]
    },
    "Logistic Regression": {
        "C"           : [0.01, 0.1, 1, 10, 100],
        "penalty"     : ["l1", "l2"],
        "solver"      : ["liblinear", "saga"],
        "class_weight": ["balanced", None]
    },
    "Decision Tree": {
        "max_depth"        : [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion"        : ["gini", "entropy"],
        "class_weight"     : ["balanced", None]
    },
    "Naive Bayes": {
        "var_smoothing": np.logspace(-9, -6, 4)
    }
}

base_models = {
    "Random Forest"      : RandomForestClassifier(random_state=SEED),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree"      : DecisionTreeClassifier(random_state=SEED),
    "Naive Bayes"        : GaussianNB(),
}

grid_search = GridSearchCV(
    estimator  = base_models[best_name],
    param_grid = param_grids[best_name],
    cv         = skf,
    scoring    = "recall",
    n_jobs     = -1,
    refit      = True,
    verbose    = 0
)
grid_search.fit(X_resampled, y_resampled)

best_model     = grid_search.best_estimator_
best_params    = grid_search.best_params_
best_cv_recall = grid_search.best_score_

print(f"  Best Params   : {best_params}")
print(f"  Best CV Recall: {round(best_cv_recall, 4)}")

# ════════════════════════════════════════════════════════
# STEP 7 — FINAL MODEL EVALUATION
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 7 — FINAL MODEL EVALUATION")
print("="*60)

y_pred = cross_val_predict(best_model, X_resampled, y_resampled, cv=skf)
y_prob = cross_val_predict(best_model, X_resampled, y_resampled,
                            cv=skf, method="predict_proba")[:, 1]

final_metrics = {
    "Accuracy" : round(accuracy_score(y_resampled, y_pred), 4),
    "Precision": round(precision_score(y_resampled, y_pred), 4),
    "Recall"   : round(recall_score(y_resampled, y_pred), 4),
    "F1"       : round(f1_score(y_resampled, y_pred), 4),
    "ROC-AUC"  : round(roc_auc_score(y_resampled, y_prob), 4),
}

for k, v in final_metrics.items():
    flag = " ⭐" if k in ["Recall", "F1"] else ""
    print(f"  {k:<12}: {v}{flag}")

print("\n  Classification Report:")
print(classification_report(y_resampled, y_pred, target_names=["Negative", "Positive"]))

# Confusion Matrix
cm = confusion_matrix(y_resampled, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred Neg", "Pred Pos"],
            yticklabels=["Act Neg",  "Act Pos"])
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig(f"{PLOTS}/09_confusion_matrix.png", dpi=120)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_resampled, y_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="#DD8452", lw=2.5, label=f"AUC = {final_metrics['ROC-AUC']}")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5)
ax.fill_between(fpr, tpr, alpha=0.15, color="#DD8452")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve — {best_name}")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/10_roc_curve.png", dpi=120)
plt.close()

# Feature Importance (if available)
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    colors_imp = ["#DD8452" if i >= len(imp) - 5 else "#4C72B0" for i in range(len(imp))]
    imp.plot(kind="barh", ax=ax, color=colors_imp, edgecolor="white")
    ax.set_title(f"Feature Importances — {best_name}\n(Orange = Top 5)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/11_feature_importance.png", dpi=120)
    plt.close()

# K-Fold Recall & F1 per fold
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, scoring, color in zip(axes, ["recall", "f1"], ["#DD8452", "#55A868"]):
    scores = cross_val_score(best_model, X_resampled, y_resampled, cv=skf, scoring=scoring)
    ax.bar([f"Fold {i+1}" for i in range(5)], scores, color=color, alpha=0.85, edgecolor="white")
    ax.axhline(scores.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean = {scores.mean():.4f}")
    for j, sc in enumerate(scores):
        ax.text(j, sc + 0.003, f"{sc:.3f}", ha="center", fontsize=9)
    ax.set_title(f"5-Fold {scoring.upper()} Scores", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_ylabel("Score")
plt.suptitle(f"K-Fold (5) Recall & F1 — {best_name} (Tuned)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS}/12_kfold_scores.png", dpi=120)
plt.close()

# ════════════════════════════════════════════════════════
# STEP 8 — SAVE MODEL & ARTEFACTS
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 8 — SAVE MODEL & ARTEFACTS")
print("="*60)

with open(os.path.join(MODELS, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
with open(os.path.join(MODELS, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODELS, "feature_names.json"), "w") as f:
    json.dump(feature_names, f)

metadata = {
    "best_model_name" : best_name,
    "best_params"     : {k: str(v) for k, v in best_params.items()},
    "final_metrics"   : final_metrics,
    "model_comparison": {m: {k: float(v) for k, v in vals.items()}
                         for m, vals in results.items()},
    "class_labels"    : {0: "Negative", 1: "Positive"},
    "features"        : feature_names,
    "smote_applied"   : True,
    "k_folds"         : 5,
}
with open(os.path.join(MODELS, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Model saved    : models/best_model.pkl")
print(f"  Scaler saved   : models/scaler.pkl")
print(f"  Metadata saved : models/metadata.json")
print(f"  EDA plots      : {len(os.listdir(PLOTS))} files saved")

print("\n" + "="*60)
print("  PIPELINE COMPLETE")
print("="*60)
print(f"\n  Best Model  : {best_name}")
print(f"  Recall      : {final_metrics['Recall']}")
print(f"  F1 Score    : {final_metrics['F1']}")
print(f"  ROC-AUC     : {final_metrics['ROC-AUC']}")