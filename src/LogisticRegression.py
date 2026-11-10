import numpy as np

def sigmoid(x):

    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

class LogisticRegressionGD:

    def __init__(self, epochs: int, learning_rate: float, alpha: float, l1_ratio: float, tol: float):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol

    def fit(self, x: np.ndarray, y: np.ndarray):

        n = x.shape[0]
        m = x.shape[1]
        w = np.zeros(m)
        b = 0
        loss_history = []
        converged = False
        n_iter = 0
        previous_logloss = np.inf
        y = np.asarray(y).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError("Target values must be 0 and 1")

        for i in range(1, self.epochs + 1):

            z = x @ w + b
            p = sigmoid(z)
            p_for_loss = np.clip(p, 1e-15, 1 - 1e-15)
            logloss = - 1 / n * np.sum(y * np.log(p_for_loss) + (1 - y) * np.log(1 - p_for_loss)) + self.alpha * (self.l1_ratio * np.sum(np.abs(w)) + (1 - self.l1_ratio) / 2 * np.sum(w ** 2))
            loss_history.append(logloss)

            n_iter = i

            if not np.isfinite(logloss):
                break

            if (np.isfinite(previous_logloss)and abs(previous_logloss - logloss) < self.tol):
                converged = True
                break

            previous_logloss = logloss

            err = p - y
            dw = x.T @ err / n + self.alpha * (self.l1_ratio * np.sign(w) + (1 - self.l1_ratio) * w)
            db = np.mean(err)
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db



        self.coef_ = w
        self.intercept_ = b
        self.loss_history_ = loss_history
        self.classes_ = classes
        self.n_features_in_ = m
        self.n_iter_ = n_iter
        self.converged_ = converged

        return self

    def predict_proba(self, data: np.ndarray) -> np.ndarray:

        if data.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {data.shape[1]}")

        logit = data @ self.coef_ + self.intercept_

        prob_class_1 = sigmoid(logit)
        prob_class_0 = 1 - prob_class_1

        return np.column_stack((prob_class_0, prob_class_1))

    def predict(self, data: np.ndarray, threshold: float = 0.5) -> np.ndarray:

        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1")


        prob_class_1 = self.predict_proba(data)[:, 1]

        return (prob_class_1 >= threshold).astype(int)