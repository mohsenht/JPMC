from xgboost import XGBClassifier

from src.classification.post_processing import evaluate
from src.utility import train_test_weight_split


def train_model():
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_weight_split()

    clf = XGBClassifier(
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=3,
        gamma=3.0,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.5,
        scale_pos_weight=7,
        objective='binary:logistic',
        tree_method='hist',
        eval_metric='aucpr',
        max_delta_step=1,
        random_state=89,
        early_stopping_rounds=200,
    )
    clf.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        sample_weight_eval_set=[w_test],
        verbose=False
    )

    evaluate(clf, X_test, y_test, w_test)

    return clf
