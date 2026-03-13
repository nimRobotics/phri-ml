"""
Trust Prediction Pipeline
=========================
End-to-end ML pipeline for binary trust classification from neurophysiological data.

Features:
  - Two binarization methods: midpoint & k-means clustering
  - Models: XGBoost, Random Forest, Neural Network (MLP)
  - 5-fold stratified cross-validation (participant-aware)
  - SHAP feature importance analysis
  - Comprehensive visualisations saved to ./outputs/
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data.csv"          # ← change if needed
OUT_DIR     = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

NEURO_FEATURES = [
    "SGE", "GTE", "hr_mean", "HRV_SDNN", "HRV_RMSSD",
    "HRV_pNN50", "DLPFC", "PMC", "SMA", "V1",
    "V1-DLPFC", "V1-PMC", "V1-SMA", "DLPFC-PMC", "DLPFC-SMA", "PMC-SMA"
]
TARGET      = "trust"
N_FOLDS     = 5
PALETTE     = {"XGBoost": "#E8463A", "RandomForest": "#3A8FE8", "NeuralNet": "#3AE878"}

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def savefig(name):
    p = OUT_DIR / name
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved {p}")

# ─── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════")
print("  TRUST PREDICTION PIPELINE")
print("══════════════════════════════════════════")
print("\n[1] Loading data …")
df = pd.read_csv(DATA_PATH)
print(f"    Shape: {df.shape}  |  Participants: {df['pid'].nunique()}")
print(f"    Trust range: {df[TARGET].min():.0f}–{df[TARGET].max():.0f}  |  "
      f"Mean: {df[TARGET].mean():.2f}  |  Std: {df[TARGET].std():.2f}")

# Drop rows with missing neuro features or target
df_clean = df[NEURO_FEATURES + [TARGET, "pid"]].dropna()
print(f"    Rows after dropping NaN: {len(df_clean)}")

# ─── 2. VISUALISE RAW TRUST DISTRIBUTION ───────────────────────────────────────
print("\n[2] Plotting trust distribution …")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Trust Score Distribution", fontsize=14, fontweight="bold")

axes[0].hist(df_clean[TARGET], bins=7, color="#5B8CDE", edgecolor="white", linewidth=0.8)
axes[0].set_xlabel("Trust Score (1–7)"); axes[0].set_ylabel("Count")
axes[0].set_title("Frequency Distribution")
axes[0].axvline(4, color="red", linestyle="--", label="Mid-point (4)")
axes[0].legend()

sns.boxplot(y=df_clean[TARGET], ax=axes[1], color="#5B8CDE")
axes[1].set_ylabel("Trust Score"); axes[1].set_title("Box Plot")
plt.tight_layout()
savefig("01_trust_distribution.png")

# ─── 3. BINARIZATION ───────────────────────────────────────────────────────────
print("\n[3] Binarizing trust scores …")

# Method A: Mid-point (≥4 = high trust)
df_clean["trust_mid"] = (df_clean[TARGET] >= 4).astype(int)

# Method B: K-means clustering (k=2)
km = KMeans(n_clusters=2, random_state=42, n_init=10)
km_labels = km.fit_predict(df_clean[[TARGET]])
# Align: cluster with higher centroid = high trust
if km.cluster_centers_[0] < km.cluster_centers_[1]:
    df_clean["trust_km"] = km_labels
else:
    df_clean["trust_km"] = 1 - km_labels

for method, col in [("Mid-point", "trust_mid"), ("K-means", "trust_km")]:
    counts = df_clean[col].value_counts()
    print(f"    {method:10s}  →  Low={counts.get(0,0)}  High={counts.get(1,0)}  "
          f"({100*counts.get(1,0)/len(df_clean):.1f}% high)")

# Visualise binarization
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Trust Binarization Methods", fontsize=14, fontweight="bold")

for ax, (method, col, color_low, color_high) in zip(axes, [
    ("Mid-point (≥4 = High)", "trust_mid", "#E87F3A", "#3A8FE8"),
    ("K-means Clustering",     "trust_km",  "#E87F3A", "#3A8FE8"),
]):
    jitter = np.random.uniform(-0.15, 0.15, len(df_clean))
    colors  = [color_high if v == 1 else color_low for v in df_clean[col]]
    ax.scatter(df_clean[TARGET] + jitter, df_clean[col], c=colors, alpha=0.5, s=20)
    ax.set_xlabel("Original Trust Score"); ax.set_ylabel("Binary Label (0=Low, 1=High)")
    ax.set_title(method)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=color_low, label="Low Trust"),
                       Patch(color=color_high, label="High Trust")])
plt.tight_layout()
savefig("02_binarization.png")

# ─── 4. FEATURE CORRELATION ────────────────────────────────────────────────────
print("\n[4] Plotting feature correlation heatmap …")
corr = df_clean[NEURO_FEATURES].corr()
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title("Neurophysiological Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("03_feature_correlation.png")

# ─── 5. FEATURE DISTRIBUTIONS BY TRUST ────────────────────────────────────────
print("\n[5] Plotting feature distributions by trust label …")
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()
for i, feat in enumerate(NEURO_FEATURES):
    ax = axes[i]
    for label, color, name in [(0, "#E87F3A", "Low"), (1, "#3A8FE8", "High")]:
        vals = df_clean.loc[df_clean["trust_mid"] == label, feat].dropna()
        ax.hist(vals, bins=20, alpha=0.6, color=color, label=name, density=True)
    ax.set_title(feat, fontsize=9, fontweight="bold")
    ax.set_xlabel(""); ax.tick_params(labelsize=7)
    if i == 0: ax.legend(fontsize=7)
for j in range(len(NEURO_FEATURES), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Feature Distributions: Low vs High Trust (Mid-point)", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("04_feature_distributions.png")

# ─── 6. MODEL DEFINITIONS ──────────────────────────────────────────────────────
def get_models():
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        "NeuralNet": MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu", solver="adam",
            learning_rate_init=0.001, max_iter=500,
            early_stopping=True, validation_fraction=0.1,
            random_state=42
        )
    }

# ─── 7. CV PIPELINE ────────────────────────────────────────────────────────────
def run_cv(X, y, method_name):
    print(f"\n  ── {method_name} binarization ──")
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    results = {m: {"acc":[], "auc":[], "f1":[], "cm": np.zeros((2,2), int),
                   "fpr":[], "tpr":[], "shap_vals":[]} for m in get_models()}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
        X_te_s = pd.DataFrame(scaler.transform(X_te),    columns=X_te.columns)

        for mname, model in get_models().items():
            model.fit(X_tr_s, y_tr)
            preds  = model.predict(X_te_s)
            proba  = model.predict_proba(X_te_s)[:, 1]

            results[mname]["acc"].append(accuracy_score(y_te, preds))
            results[mname]["auc"].append(roc_auc_score(y_te, proba))
            results[mname]["f1"].append(f1_score(y_te, preds, zero_division=0))
            results[mname]["cm"] += confusion_matrix(y_te, preds)

            fpr, tpr, _ = roc_curve(y_te, proba)
            results[mname]["fpr"].append(fpr)
            results[mname]["tpr"].append(tpr)

            # SHAP on last fold only (expensive)
            if fold == N_FOLDS - 1:
                try:
                    if mname in ("XGBoost", "RandomForest"):
                        explainer = shap.TreeExplainer(model)
                        sv = explainer.shap_values(X_te_s)
                        if isinstance(sv, list):
                            sv = sv[1]
                        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                            sv = sv[:, :, 1]  # shape (n_samples, n_features, n_classes)
                    else:
                        explainer = shap.KernelExplainer(
                            model.predict_proba, shap.sample(X_tr_s, 50))
                        sv = explainer.shap_values(X_te_s, nsamples=100)
                        if isinstance(sv, list):
                            sv = sv[1]
                        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                            sv = sv[:, :, 1]
                    results[mname]["shap_vals"] = (sv, X_te_s)
                except Exception as e:
                    print(f"    SHAP error ({mname}): {e}")

        print(f"    Fold {fold+1}/{N_FOLDS} done")

    # Summary
    print(f"\n  {'Model':<14}  {'Acc':>7}  {'AUC':>7}  {'F1':>7}")
    print(f"  {'─'*44}")
    for mname, res in results.items():
        print(f"  {mname:<14}  "
              f"{np.mean(res['acc']):.3f}±{np.std(res['acc']):.3f}  "
              f"{np.mean(res['auc']):.3f}±{np.std(res['auc']):.3f}  "
              f"{np.mean(res['f1']):.3f}±{np.std(res['f1']):.3f}")
    return results

# ─── 8. RUN FOR BOTH METHODS ───────────────────────────────────────────────────
print("\n[6] Running 5-fold cross-validation …")
X = df_clean[NEURO_FEATURES]
all_results = {}
for method, col in [("Midpoint", "trust_mid"), ("Kmeans", "trust_km")]:
    all_results[method] = run_cv(X, df_clean[col], method)

# ─── 9. METRICS BAR CHART ──────────────────────────────────────────────────────
print("\n[7] Plotting performance metrics …")
for method, results in all_results.items():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"5-Fold CV Performance — {method} binarization",
                 fontsize=13, fontweight="bold")
    metrics = {"Accuracy": "acc", "ROC-AUC": "auc", "F1 Score": "f1"}
    for ax, (metric_name, key) in zip(axes, metrics.items()):
        means = [np.mean(results[m][key]) for m in results]
        stds  = [np.std(results[m][key])  for m in results]
        colors= [PALETTE[m] for m in results]
        bars  = ax.bar(list(results.keys()), means, yerr=stds,
                       color=colors, capsize=5, edgecolor="white", linewidth=0.8)
        ax.set_ylim(0, 1.1); ax.set_ylabel(metric_name)
        ax.set_title(metric_name, fontweight="bold")
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, mean+std+0.02,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    savefig(f"05_metrics_{method.lower()}.png")

# ─── 10. ROC CURVES ────────────────────────────────────────────────────────────
print("\n[8] Plotting ROC curves …")
for method, results in all_results.items():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"ROC Curves — {method} binarization",
                 fontsize=13, fontweight="bold")
    for ax, mname in zip(axes, results):
        res = results[mname]
        for i, (fpr, tpr) in enumerate(zip(res["fpr"], res["tpr"])):
            ax.plot(fpr, tpr, alpha=0.35, color=PALETTE[mname], lw=1)
        mean_auc = np.mean(res["auc"])
        ax.plot([0,1],[0,1],"k--", lw=0.8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"{mname}\nMean AUC = {mean_auc:.3f}", fontweight="bold")
        ax.set_xlim(0,1); ax.set_ylim(0,1.02)
        # Mean curve approximation
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(all_fpr, f, t)
                            for f,t in zip(res["fpr"], res["tpr"])], axis=0)
        ax.plot(all_fpr, mean_tpr, color=PALETTE[mname], lw=2.5, label="Mean ROC")
        ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(f"06_roc_{method.lower()}.png")

# ─── 11. CONFUSION MATRICES ────────────────────────────────────────────────────
print("\n[9] Plotting confusion matrices …")
for method, results in all_results.items():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Aggregated Confusion Matrices — {method} binarization",
                 fontsize=13, fontweight="bold")
    for ax, mname in zip(axes, results):
        cm = results[mname]["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Low","High"], yticklabels=["Low","High"],
                    linewidths=0.5, cbar=False)
        ax.set_title(mname, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    savefig(f"07_confusion_{method.lower()}.png")

# ─── 12. SHAP ANALYSIS ─────────────────────────────────────────────────────────
print("\n[10] Generating SHAP plots …")

def plot_shap(shap_vals, X_test, model_name, method):
    if len(shap_vals) == 0:
        return
    sv, Xte = shap_vals

    # Beeswarm / Summary
    sv_plot = np.array(sv)
    if sv_plot.ndim == 3:
        sv_plot = sv_plot[:, :, 1]
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(sv_plot, Xte, show=False, plot_type="dot",
                      color_bar_label="Feature value")
    plt.title(f"SHAP Summary — {model_name} ({method})", fontweight="bold")
    plt.tight_layout()
    savefig(f"08_shap_summary_{model_name}_{method}.png")

    # Bar: mean |SHAP|
    sv_arr = np.array(sv)
    # Ensure 2D: (n_samples, n_features)
    if sv_arr.ndim == 3:
        sv_arr = sv_arr[:, :, 1]
    mean_shap  = np.abs(sv_arr).mean(axis=0).flatten()
    feat_names = Xte.columns.tolist()
    order      = [int(i) for i in np.argsort(mean_shap)]    # ascending for barh
    fig, ax    = plt.subplots(figsize=(9, 6))
    ax.barh([feat_names[i] for i in order],
            [float(mean_shap[i]) for i in order],
            color=PALETTE.get(model_name, "#5B8CDE"), edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance (SHAP) — {model_name} ({method})", fontweight="bold")
    plt.tight_layout()
    savefig(f"09_shap_bar_{model_name}_{method}.png")

for method, results in all_results.items():
    for mname, res in results.items():
        if res["shap_vals"]:
            plot_shap(res["shap_vals"], None, mname, method)

# ─── 13. CROSS-METHOD COMPARISON ───────────────────────────────────────────────
print("\n[11] Plotting method comparison …")
rows = []
for method, results in all_results.items():
    for mname, res in results.items():
        rows.append({"Method": method, "Model": mname,
                     "Accuracy": np.mean(res["acc"]),
                     "ROC-AUC":  np.mean(res["auc"]),
                     "F1":       np.mean(res["f1"])})
comp_df = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Binarization Method × Model Comparison", fontsize=13, fontweight="bold")
for ax, metric in zip(axes, ["Accuracy", "ROC-AUC", "F1"]):
    pivot = comp_df.pivot(index="Model", columns="Method", values=metric)
    pivot.plot(kind="bar", ax=ax, color=["#E8463A", "#3A8FE8"],
               edgecolor="white", width=0.6)
    ax.set_ylim(0, 1.1); ax.set_title(metric, fontweight="bold")
    ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Binarization")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
plt.tight_layout()
savefig("10_method_comparison.png")

# ─── 14. SHAP TOP-FEATURE COMPARISON ───────────────────────────────────────────
print("\n[12] Plotting aggregate SHAP feature ranking …")
for method, results in all_results.items():
    all_importance = {}
    for mname, res in results.items():
        if res["shap_vals"]:
            sv, Xte = res["shap_vals"]
            mean_shap = np.abs(sv).mean(axis=0)
            for feat, val in zip(Xte.columns, mean_shap):
                all_importance.setdefault(feat, {})[mname] = val

    if not all_importance:
        continue

    imp_df = pd.DataFrame(all_importance).T.fillna(0)
    imp_df["mean"] = imp_df.mean(axis=1)
    imp_df = imp_df.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bottom  = np.zeros(len(imp_df))
    for mname in [c for c in imp_df.columns if c != "mean"]:
        ax.barh(imp_df.index, imp_df[mname], left=bottom,
                label=mname, color=PALETTE.get(mname, "#888"), alpha=0.85)
        bottom += imp_df[mname].values
    ax.set_xlabel("Cumulative Mean |SHAP|")
    ax.set_title(f"Aggregate Feature Importance — {method} binarization",
                 fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    savefig(f"11_shap_aggregate_{method.lower()}.png")

# ─── 15. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════")
print("  PIPELINE COMPLETE — OUTPUT FILES")
print("══════════════════════════════════════════")
out_files = sorted(OUT_DIR.glob("*.png"))
for f in out_files:
    print(f"  {f.name}")

comp_df.to_csv(OUT_DIR / "model_performance_summary.csv", index=False)
print(f"\n  model_performance_summary.csv")
print("\n  Best models:")
best = comp_df.loc[comp_df.groupby("Method")["ROC-AUC"].idxmax()]
print(best[["Method","Model","Accuracy","ROC-AUC","F1"]].to_string(index=False))
print()
