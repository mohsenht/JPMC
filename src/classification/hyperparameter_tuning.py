import numpy as np
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.utility import train_test_weight_split

X_train, X_test, y_train, y_test, w_train, w_test = train_test_weight_split()

pos = (y_train == 1)
neg = ~pos
spw_base = float(w_train[neg].sum() / w_train[pos].sum())

space = {
    "learning_rate": [0.02, 0.05, 0.1],
    "max_depth": [4, 6, 8],
    "min_child_weight": [1, 3, 7],
    "gamma": [0.0, 1.0, 3.0],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [0.5, 1.0, 2.0],
    "scale_pos_weight": [spw_base * 0.75, spw_base, spw_base * 1.5],
}

param_sets = [
    dict(zip(space.keys(), vals))
    for vals in product(*space.values())
]

rng = np.random.default_rng(89)
if len(param_sets) > 30:
    param_sets = list(rng.choice(param_sets, size=30, replace=False))


def cv_weighted_aucpr(params):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=89)
    ap_scores = []

    for tr_idx, va_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        w_tr, w_va = w_train[tr_idx], w_train[va_idx]

        clf = XGBClassifier(
            n_estimators=5000,
            tree_method='hist',
            objective='binary:logistic',
            eval_metric='aucpr',
            max_delta_step=1,
            random_state=89,
            early_stopping_rounds=150,
            **params
        )

        clf.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            sample_weight_eval_set=[w_va],
            verbose=False
        )

        y_va_proba = clf.predict_proba(X_va)[:, 1]
        ap = average_precision_score(y_va, y_va_proba, sample_weight=w_va)
        ap_scores.append(ap)

    return float(np.mean(ap_scores))


results = []
for ps in param_sets:
    ap = cv_weighted_aucpr(ps)
    results.append((ap, ps))
    print(f"Average Precision={ap:.4f}  |  {ps}")

best_ap, best_params = max(results, key=lambda t: t[0])
print("\nBest mean weighted AP:", round(best_ap, 4))
print("Best params:", best_params)
