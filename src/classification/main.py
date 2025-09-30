from src.classification.hyperparameter_tuning import tune_hyperparameters
from src.classification.post_processing import evaluate, list_top_feautures, pick_best_threshold
from src.classification.training import train_model

if __name__ == "__main__":
    # best_params = tune_hyperparameters() # Commented since already tuned
    classifier, x_test, y_test, w_test = train_model()
    list_top_feautures(classifier, x_test.columns)
    threshold = pick_best_threshold(0.06, classifier, x_test, y_test, w_test)
    evaluate(classifier, x_test, y_test, w_test, threshold)