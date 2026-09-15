"""Train two potability classifiers -- a TabPFN primary and a sklearn-only fallback -- plus
the WQI regressor, saved as two separate joblib files so a constrained deploy target (e.g.
Render free tier's 512MB RAM) that can't load TabPFN's ~200MB weights + torch can still load
model_core.joblib and run on the fallback. A single combined file would fail to unpickle
*anything* in it if TabPFN's class can't be imported.

TabPFN is the current state-of-the-art for small tabular data (<=10k rows, <=500 features --
this dataset is exactly that size) and empirically beat every tuned alternative tried here
(plain RF, tuned HistGB, SMOTE+stacking ensemble): accuracy 0.688 vs ~0.63-0.67, ROC-AUC
0.723 vs ~0.66-0.67. First run needs a TABPFN_TOKEN env var (from https://ux.priorlabs.ai)
to download weights; they're cached locally afterward, no network/token needed later.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

from wqi import compute_wqi, wqi_category

df = pd.read_csv("data/water_potability.csv")
X, y = df.drop(columns=["Potability"]), df["Potability"]

# WQI regressor: predicts a value *derived from* these same features, so it legitimately
# scores much higher than Potability (near-zero correlation with its own features).
X_filled = X.fillna(X.median())
wqi_scores = X_filled.apply(lambda row: compute_wqi(row.to_dict()), axis=1)
wqi_train_X, wqi_test_X, wqi_train_y, wqi_test_y = train_test_split(X, wqi_scores, test_size=0.2, random_state=42)
wqi_model = HistGradientBoostingRegressor(random_state=42).fit(wqi_train_X, wqi_train_y)
wqi_pred = wqi_model.predict(wqi_test_X)
wqi_r2 = r2_score(wqi_test_y, wqi_pred)
wqi_cat_acc = accuracy_score([wqi_category(v) for v in wqi_test_y], [wqi_category(v) for v in wqi_pred])
print(f"WQI regressor: r2={wqi_r2:.3f} category_accuracy={wqi_cat_acc:.3f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)


def best_accuracy_threshold(fitted_model, X_val, y_val):
    val_proba = fitted_model.predict_proba(X_val)[:, 1]
    candidates = np.linspace(0.05, 0.95, 91)
    accs = [accuracy_score(y_val, val_proba >= t) for t in candidates]
    return float(candidates[int(np.argmax(accs))])


def evaluate(fitted_model, threshold, X_test, y_test):
    proba = fitted_model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return accuracy_score(y_test, pred), f1_score(y_test, pred), roc_auc_score(y_test, proba)


# Fallback: pure sklearn, no extra deps, low memory -- always deployable.
fallback_model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42).fit(X_fit, y_fit)
fallback_threshold = best_accuracy_threshold(fallback_model, X_val, y_val)
fallback_model.fit(X_train, y_train)
fb_acc, fb_f1, fb_auc = evaluate(fallback_model, fallback_threshold, X_test, y_test)
print(f"fallback (HistGB): threshold={fallback_threshold:.3f} accuracy={fb_acc:.3f} f1={fb_f1:.3f} roc_auc={fb_auc:.3f}")

# Primary: TabPFN.
tabpfn_model = TabPFNClassifier(random_state=42).fit(X_fit, y_fit)
tabpfn_threshold = best_accuracy_threshold(tabpfn_model, X_val, y_val)
tabpfn_model.fit(X_train, y_train)
tp_acc, tp_f1, tp_auc = evaluate(tabpfn_model, tabpfn_threshold, X_test, y_test)
print(f"primary (TabPFN): threshold={tabpfn_threshold:.3f} accuracy={tp_acc:.3f} f1={tp_f1:.3f} roc_auc={tp_auc:.3f}")

assert fb_acc > 0.55 and tp_acc > 0.55, "accuracy at/below random-guess baseline -- pipeline is broken"

importances = permutation_importance(tabpfn_model, X_test, y_test, scoring="f1", n_repeats=5, random_state=42)
feature_importance = dict(zip(X.columns, importances.importances_mean.round(4).tolist()))
print("feature importance:", sorted(feature_importance.items(), key=lambda kv: -kv[1]))

metrics = {
    "potability_accuracy": tp_acc, "potability_f1": tp_f1, "potability_roc_auc": tp_auc,
    "fallback_accuracy": fb_acc, "fallback_roc_auc": fb_auc,
    "wqi_r2": wqi_r2, "wqi_category_accuracy": wqi_cat_acc, "dataset_rows": len(df),
}

joblib.dump(
    {
        "fallback_model": fallback_model, "fallback_threshold": fallback_threshold,
        "columns": list(X.columns), "feature_importance": feature_importance,
        "wqi_model": wqi_model, "metrics": metrics,
    },
    "model_core.joblib",
)
joblib.dump({"model": tabpfn_model, "threshold": tabpfn_threshold}, "model_tabpfn.joblib")
print("saved model_core.joblib + model_tabpfn.joblib")
