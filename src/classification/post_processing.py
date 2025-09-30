import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.utility import load_data


def pick_best_threshold(target_pp, clf, X_test, y_test, w_test):
    y_proba = clf.predict_proba(X_test)[:, 1]

    def weighted_pp_rate(thr):
        y_hat = (y_proba >= thr).astype(int)
        return (w_test * y_hat).sum() / w_test.sum()

    grid = np.linspace(0.1, 0.9, 36)
    rows = []
    for t in grid:
        y_hat = (y_proba >= t).astype(int)
        rows.append({
            "t": t,
            "w_prec": precision_score(y_test, y_hat, sample_weight=w_test, zero_division=0),
            "w_rec": recall_score(y_test, y_hat, sample_weight=w_test, zero_division=0),
            "w_f1": f1_score(y_test, y_hat, sample_weight=w_test, zero_division=0),
            "w_pp": weighted_pp_rate(t)
        })

    t_best = min(grid, key=lambda t: abs(weighted_pp_rate(t) - target_pp))
    print(f"Best threshold for {target_pp} weighted PP rate: {t_best}")

    return t_best


def evaluate(clf, X_test, y_test, w_test, threshold):
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "weighted_accuracy": accuracy_score(y_test, y_pred, sample_weight=w_test),
        "weighted_precision": precision_score(y_test, y_pred, sample_weight=w_test, zero_division=0),
        "weighted_recall": recall_score(y_test, y_pred, sample_weight=w_test, zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, sample_weight=w_test, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba, sample_weight=w_test),
    }

    print("\nWeighted Test Metrics:")
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.4f}")


def list_top_feautures(clf, columns):

    booster = clf.get_booster()
    score_dict = booster.get_score(importance_type='gain')

    idx_to_name = dict(enumerate(columns.tolist()))
    items = []
    for k, gain in score_dict.items():
        try:
            idx = int(k[1:])
            items.append((idx_to_name.get(idx, k), gain))
        except:
            items.append((k, gain))

    topK = sorted(items, key=lambda t: t[1], reverse=True)
    i = 0
    for name, gain in topK:
        i += 1
        print(f"{i}: {name}: {gain:.2f}")
