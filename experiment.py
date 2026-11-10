import numpy as np

from src.LogisticRegression import LogisticRegressionGD
from src.data import generate_data
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


SEED = 42
N_SAMPLES = 1000
TRUE_WEIGHTS = np.array([2.0, -1.5, 0.8])
TRUE_BIAS = -0.3

TRAIN_FRACTION = 0.6
VALIDATION_FRACTION = 0.2

EPOCHS = 5000
LEARNING_RATE = 0.03
ALPHA = 0.01
L1_RATIO = 0.0
TOL = 1e-8


def calculate_logloss(y_true, probabilities):
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

    return -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )


def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def evaluate(model, x, y):
    probabilities = model.predict_proba(x)[:, 1]
    predictions = model.predict(x)

    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()

    return {
        "logloss": calculate_logloss(y, probabilities),
        "accuracy": calculate_accuracy(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }



x, y = generate_data(
    n_samples=N_SAMPLES,
    true_weights=TRUE_WEIGHTS,
    true_bias=TRUE_BIAS,
    seed=SEED,
)

rng = np.random.default_rng(SEED)
indices = rng.permutation(N_SAMPLES)
x = x[indices]
y = y[indices]

train_end = int(N_SAMPLES * TRAIN_FRACTION)
validation_end = train_end + int(N_SAMPLES * VALIDATION_FRACTION)

x_train = x[:train_end]
y_train = y[:train_end]

x_validation = x[train_end:validation_end]
y_validation = y[train_end:validation_end]

x_test = x[validation_end:]
y_test = y[validation_end:]


def main():
    model = LogisticRegressionGD(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        alpha=ALPHA,
        l1_ratio=L1_RATIO,
        tol=TOL,
    )

    model.fit(x_train, y_train)

    train_metrics = evaluate(model, x_train, y_train)
    validation_metrics = evaluate(model, x_validation, y_validation)
    test_metrics = evaluate(model, x_test, y_test)

    result = {
        "max_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "alpha": ALPHA,
        "l1_ratio": L1_RATIO,
        "n_iter": model.n_iter_,
        "converged": model.converged_,

        "train_logloss": train_metrics["logloss"],
        "validation_logloss": validation_metrics["logloss"],
        "test_logloss": test_metrics["logloss"],

        "train_accuracy": train_metrics["accuracy"],
        "validation_accuracy": validation_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],

        "train_precision": train_metrics["precision"],
        "validation_precision": validation_metrics["precision"],
        "test_precision": test_metrics["precision"],

        "train_recall": train_metrics["recall"],
        "validation_recall": validation_metrics["recall"],
        "test_recall": test_metrics["recall"],

        "train_f1": train_metrics["f1"],
        "validation_f1": validation_metrics["f1"],
        "test_f1": test_metrics["f1"],

        "weight_error": np.linalg.norm(
            model.coef_ - TRUE_WEIGHTS
        ),
        "bias_error": abs(model.intercept_ - TRUE_BIAS),
    }

    return result

if __name__ == '__main__':
    result = main()
    print(result)