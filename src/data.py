import numpy as np

from LogisticRegression import sigmoid


def generate_data(n_samples, true_weights, true_bias, seed):
    rng = np.random.default_rng(seed)

    true_weights = np.asarray(true_weights)
    n_features = len(true_weights)

    x = rng.normal(
        loc=0,
        scale=1,
        size=(n_samples, n_features),
    )

    logits = x @ true_weights + true_bias
    probabilities = sigmoid(logits)
    y = rng.binomial(n=1, p=probabilities)

    return x, y
