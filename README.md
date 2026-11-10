# Logistic Regression from Scratch

An educational implementation of binary logistic regression written with NumPy.
The model is trained with full-batch gradient descent and supports L1, L2, and
Elastic Net regularization. The repository also contains reproducible synthetic
data generation and two experiments that study optimization and regularization.

The goal of this project is not to replace scikit-learn. It is to make the
mathematics of logistic regression explicit and to connect every formula with a
small, readable implementation.

## What is implemented

- binary logistic regression for labels `0` and `1`;
- numerically protected sigmoid and binary cross-entropy;
- full-batch gradient descent;
- L1, L2, and Elastic Net regularization;
- convergence detection using a loss tolerance;
- `fit`, `predict_proba`, and `predict` methods;
- configurable classification threshold;
- accuracy, precision, recall, F1, and confusion-matrix counts;
- deterministic synthetic data generation with a random seed;
- optimizer and regularization experiments saved as CSV files.

## Project structure

```text
LogisticRegressionFromScratch/
├── src/
│   ├── __init__.py                       # Package exports
│   ├── LogisticRegression.py             # Sigmoid and LogisticRegressionGD
│   └── data.py                           # Reproducible synthetic data generation
├── results/
│   ├── optimizer_results.csv
│   └── regularization_results.csv
├── experiment.py                         # Trains and evaluates the selected configuration
├── README.md
├── .gitignore
└── LICENSE
```

This layout is intentionally small. The reusable code lives in `src`, the
experiment remains an obvious entry point in the repository root, and compact
experiment outputs are stored in `results`. A `tests` directory can be added
later when automated comparisons with scikit-learn are introduced.

## Mathematical model

Let

- $X \in \mathbb{R}^{n \times m}$ be the feature matrix;
- $n$ be the number of samples;
- $m$ be the number of features;
- $w \in \mathbb{R}^{m}$ be the coefficient vector;
- $b \in \mathbb{R}$ be the intercept;
- $y_i \in \{0,1\}$ be the target for sample $i$.

The linear score, also called the logit, is

$$
z = Xw + b.
$$

The sigmoid converts each logit into the estimated probability of class 1:

$$
p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}.
$$

The probability of class 0 is $1-p_i$.

## What exactly is log loss?

Logistic regression does not directly minimize the number of incorrect class
predictions. It learns probabilities. Therefore it needs a loss function that
evaluates how well the predicted probabilities agree with the observed binary
targets.

For one sample, the target is treated as a Bernoulli random variable. Its
conditional probability can be written as

$$
P(y_i \mid x_i,w,b)=p_i^{y_i}(1-p_i)^{1-y_i}.
$$

This expression covers both possible targets:

- if $y_i=1$, it becomes $P(y_i=1)=p_i$;
- if $y_i=0$, it becomes $P(y_i=0)=1-p_i$.

Assuming that the samples are conditionally independent, the likelihood of all
observed targets is

$$
L(w,b)=\prod_{i=1}^{n}p_i^{y_i}(1-p_i)^{1-y_i}.
$$

Maximum-likelihood estimation chooses the parameters that make the observed
targets as probable as possible. Products are inconvenient to differentiate
and can become numerically tiny, so the logarithm is applied. Because the
logarithm is strictly increasing, maximizing $L$ and maximizing $\log L$ give
the same parameters.

Gradient descent is conventionally written as a minimization algorithm, so the
negative average log-likelihood is used:

$$
\mathcal{L}_{\mathrm{log}}(w,b)=
-\frac{1}{n}\sum_{i=1}^{n}
\left[y_i\log p_i + (1-y_i)\log(1-p_i)\right].
$$

This function has several equivalent names:

- **binary log loss**;
- **binary cross-entropy**;
- **average negative log-likelihood of the Bernoulli model**.

The abbreviation **BCE** means *binary cross-entropy*. It is not a separate
model and not an additional probability function. It is simply another common
name for the loss above. This README uses $\mathcal{L}_{\mathrm{log}}$ to make
its role explicit.

For one sample, the loss is $-\log p_i$ when the true class is 1 and
$-\log(1-p_i)$ when the true class is 0. A confident incorrect prediction
therefore receives a much larger penalty than an uncertain prediction. For
example, if the correct class is 1, predicting $p=0.9$ gives a loss of about
$0.105$, while predicting $p=0.01$ gives a loss of about $4.605$.

Accuracy cannot be used as the training objective because it changes only when
a probability crosses the classification threshold. It is piecewise constant
and does not provide a useful gradient. Log loss is smooth with respect to the
logit and produces the simple derivative $p_i-y_i$.

## Elastic Net objective

The implementation minimizes

$$
J(w,b)=\mathcal{L}_{\mathrm{log}}(w,b)
+\alpha\left[
\rho\sum_{j=1}^{m}|w_j|
+\frac{1-\rho}{2}\sum_{j=1}^{m}w_j^2
\right].
$$

where

- $\alpha \ge 0$ controls the total regularization strength;
- $\rho$ is `l1_ratio`;
- $\rho=0$ gives pure L2 regularization;
- $\rho=1$ gives pure L1 regularization;
- $0<\rho<1$ gives Elastic Net.

The related terms are easy to mix up, so their roles are summarized below.

| Quantity | Meaning | Training direction | Includes regularization |
|---|---|---|:---:|
| $L(w,b)$ | Likelihood of the observed targets | Maximize | No |
| $\log L(w,b)$ | Log-likelihood | Maximize | No |
| $\mathcal{L}_{\mathrm{log}}(w,b)$ | Log loss, BCE, or negative log-likelihood | Minimize | No |
| $J(w,b)$ | Complete training objective | Minimize | Yes |

In the implementation, `calculate_logloss` reports only
$\mathcal{L}_{\mathrm{log}}$, which is appropriate for comparing predictive
probabilities. The model's `loss_history_` stores $J$, because that is the
regularized objective optimized by gradient descent.

The intercept is not regularized. L2 smoothly shrinks all coefficients. L1 can
push coefficients toward zero and can therefore act as a simple feature
selector, although ordinary subgradient descent does not guarantee exact zeros.

## Gradients and parameter update

For logistic regression, the derivative of log loss with respect to the logit
is

$$
\frac{\partial \mathcal{L}_{\mathrm{log}}}{\partial z_i}=p_i-y_i.
$$

Therefore the gradients of the regularized objective are

$$ \nabla_w J =  \frac{1}{n}X^T(p-y) + \alpha\left[\rho\,\mathrm{sign}(w)+(1-\rho)w\right], $$

$$
\frac{\partial J}{\partial b}=\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i).
$$

One gradient-descent step with learning rate $\eta$ is

$$
w \leftarrow w-\eta\nabla_wJ,
\qquad
b \leftarrow b-\eta\frac{\partial J}{\partial b}.
$$

The model predicts class 1 when the estimated probability is at least the
chosen threshold $t$:

$$
\hat y_i=1 \quad \text{if } p_i\ge t,
\qquad
\hat y_i=0 \quad \text{if } p_i<t.
$$

The default threshold is $t=0.5$. It affects the final class labels but does
not participate in model training.

## Public API

### Constructor parameters

| Parameter | Meaning |
|---|---|
| `epochs` | Maximum number of gradient-descent iterations |
| `learning_rate` | Gradient-descent step size $\eta$ |
| `alpha` | Total regularization strength |
| `l1_ratio` | Mixture between L2 (`0`) and L1 (`1`) |
| `tol` | Minimum objective change required to continue training |

### Attributes created by `fit`

| Attribute | Meaning |
|---|---|
| `coef_` | Learned coefficient vector |
| `intercept_` | Learned intercept |
| `loss_history_` | Regularized training objective at each iteration |
| `classes_` | Observed target classes, expected to be `[0, 1]` |
| `n_features_in_` | Number of training features |
| `n_iter_` | Number of executed iterations |
| `converged_` | Whether the tolerance criterion was reached |

### Methods

| Method | Output |
|---|---|
| `fit(X, y)` | Fits the model and returns `self` |
| `predict_proba(X)` | Array of shape `(n_samples, 2)` with class probabilities |
| `predict(X, threshold=0.5)` | One-dimensional array of predicted classes |

## Synthetic dataset

The experiment generates features from a standard normal distribution:

$$
X_{ij}\sim\mathcal{N}(0,1).
$$

Targets are sampled from the same probabilistic model that the estimator is
trying to learn:

$$
y_i\sim\mathrm{Bernoulli}\left(
\sigma(x_i^Tw_{\mathrm{true}}+b_{\mathrm{true}})
\right).
$$

The current configuration is

| Setting | Value |
|---|---:|
| Samples | 1,000 |
| Features | 3 |
| True weights | `[2.0, -1.5, 0.8]` |
| True bias | `-0.3` |
| Train / validation / test | 60% / 20% / 20% |
| Seed | 42 |

The rows are shuffled deterministically before splitting. Because every feature
is sampled from $\mathcal{N}(0,1)$, the inputs are already on comparable
scales. Real datasets should normally be standardized before comparing L1 and
L2 penalties.

## Evaluation metrics

The experiments calculate the following metrics separately on train and
validation data:

| Metric | Purpose |
|---|---|
| Log loss | Evaluates probability quality |
| Accuracy | Fraction of correct class predictions |
| Precision | Fraction of predicted positives that are correct |
| Recall | Fraction of actual positives found by the model |
| F1 | Harmonic mean of precision and recall |
| Weight error | Euclidean distance from the known synthetic weights |
| Bias error | Absolute error of the learned intercept |

The test set is intentionally excluded from hyperparameter selection. It should
be evaluated once after the final configuration has been selected.

## Optimizer experiment

The optimizer experiment varied `learning_rate` and `max_epochs` while keeping
regularization disabled. The complete 50-row table is stored in
[`results/optimizer_results.csv`](results/optimizer_results.csv).

### Best validation result for each epoch budget

| Max epochs | Learning rate | Iterations | Converged | Validation log loss | Validation F1 |
|---:|---:|---:|:---:|---:|---:|
| 100 | 0.200 | 100 | No | 0.421477 | 0.741573 |
| 1,000 | 0.030 | 1,000 | No | **0.418705** | 0.741573 |
| 2,500 | 0.010 | 2,500 | No | 0.419473 | 0.741573 |
| 5,000 | 0.005 | 5,000 | No | 0.419474 | 0.741573 |
| 10,000 | 0.005 | 10,000 | No | 0.419597 | 0.741573 |

The lowest validation log loss in this grid was obtained with a learning rate
of `0.03` after 1,000 iterations. The optimizer had not reached the tolerance
criterion, but continuing toward the minimum training loss slightly worsened
validation log loss. Early stopping therefore acted as implicit regularization.

### Speed after convergence

| Learning rate | Iterations to convergence | Validation log loss |
|---:|---:|---:|
| 0.030 | 4,804 | 0.423578 |
| 0.050 | 3,061 | 0.423659 |
| 0.070 | 2,271 | 0.423703 |
| 0.100 | 1,652 | 0.423742 |
| 0.120 | 1,403 | 0.423759 |
| 0.150 | 1,149 | 0.423779 |
| 0.200 | 887 | 0.423801 |

Within the tested range, larger learning rates reached essentially the same
training solution in fewer iterations. A learning rate of `0.03` was retained
for the regularization study because it was stable and exposed the difference
between early stopping and full convergence.

## Regularization experiment

The regularization experiment used `learning_rate=0.03`, `max_epochs=10000`,
and different combinations of `alpha` and `l1_ratio`. The full table is stored
in [`results/regularization_results.csv`](results/regularization_results.csv).

### Pure L2 results

| Alpha | Iterations | Converged | Train log loss | Validation log loss | Validation F1 |
|---:|---:|:---:|---:|---:|---:|
| 0.0000 | 4,804 | Yes | 0.364278 | 0.423578 | 0.741573 |
| 0.0005 | 4,613 | Yes | 0.364341 | 0.422719 | 0.741573 |
| 0.0010 | 4,440 | Yes | 0.364466 | 0.421978 | 0.741573 |
| 0.0050 | 3,483 | Yes | 0.366824 | 0.418847 | 0.741573 |
| 0.0100 | 2,816 | Yes | 0.371290 | **0.418559** | 0.741573 |
| 0.0500 | 1,321 | Yes | 0.408620 | 0.439674 | 0.739884 |
| 0.1000 | 903 | Yes | 0.443123 | 0.466392 | 0.752941 |

A small L2 penalty improved validation log loss after the unregularized model
was trained to convergence. The best value in this experiment was
`alpha=0.01`. Stronger regularization increased bias and degraded probability
quality. The higher F1 at `alpha=0.1` is caused by changed class decisions at
the fixed threshold of `0.5`; it does not indicate better probability
estimates.

### Effect of the L1 ratio at alpha = 0.01

| L1 ratio | Regularization | Validation log loss | Validation F1 |
|---:|---|---:|---:|
| 0.0 | L2 | **0.418559** | 0.741573 |
| 0.2 | Elastic Net | 0.418651 | 0.741573 |
| 0.4 | Elastic Net | 0.418787 | 0.741573 |
| 0.5 | Elastic Net | 0.418875 | 0.741573 |
| 0.6 | Elastic Net | 0.418979 | 0.741573 |
| 0.8 | Elastic Net | 0.419242 | 0.741573 |
| 1.0 | L1 | 0.419597 | 0.741573 |

Increasing the L1 component did not help because all three generated features
are informative. There are no irrelevant coefficients for L1 to remove. The
pure L1 configuration with `alpha=0.1` also failed to meet the strict tolerance
within 10,000 iterations because a basic subgradient step can oscillate around
zero.

## Main findings

1. Learning rate controls optimization speed, while the number of epochs is
   only an upper limit when tolerance-based stopping is active.
2. Training the unregularized model to convergence slightly overfits the
   validation split.
3. Stopping the unregularized model near 1,000 iterations and training an
   L2-regularized model to convergence produce almost the same validation log
   loss (`0.418705` versus `0.418559`).
4. A small L2 penalty (`alpha=0.01`) gives the best validation log loss in the
   completed regularization grid.
5. L1 is not useful for the current low-dimensional dataset because every
   feature is relevant.
6. Classification metrics can remain unchanged while log loss changes because
   the probabilities move without crossing the classification threshold.

These conclusions are based on one seed and one train/validation split. The
difference between the best configurations is small and should be confirmed
over multiple seeds before making a strong statistical claim.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install numpy scikit-learn matplotlib
```

## Usage

Run the selected final configuration from the repository root:

```bash
python experiment.py
```

The script prints its hyperparameters, convergence information, and train and
validation metrics. The complete optimizer and regularization grids are kept as
the two versioned CSV files in `results`; rerunning the selected configuration
does not overwrite those experiment tables.

The estimator can also be used directly:

```python
import numpy as np

from src.LogisticRegression import LogisticRegressionGD

X = np.array([
    [-1.0, 0.2],
    [-0.5, 0.1],
    [0.7, 0.8],
    [1.2, 1.0],
])
y = np.array([0, 0, 1, 1])

model = LogisticRegressionGD(
    epochs=5000,
    learning_rate=0.03,
    alpha=0.01,
    l1_ratio=0.0,
    tol=1e-8,
)

model.fit(X, y)
probabilities = model.predict_proba(X)
predictions = model.predict(X, threshold=0.5)
```

## Numerical considerations

- Logits are clipped before the sigmoid to prevent overflow in `exp`.
- Probabilities are clipped before logarithms to prevent `log(0)`.
- The reported train and validation log losses do not include the penalty;
  `loss_history_` stores the complete regularized training objective.
- L1 uses a simple subgradient based on `sign(w)`. A proximal optimizer would
  be preferable if exact sparse coefficients were a project requirement.

## Limitations and future work

- Only binary classification is supported.
- Training uses full-batch gradient descent rather than mini-batches.
- The dataset is synthetic and correctly specified by logistic regression.
- The experiment currently uses one random seed and one data split.
- The implementation does not provide validation-based checkpoint restoration.
- The model has not yet been systematically compared with scikit-learn.

Reasonable next steps are:

1. add unit tests and compare predictions with scikit-learn;
2. run the strongest configurations over multiple seeds;
3. separate data loss and regularization penalty histories;
4. add `decision_function` and fitted-state checks;
5. evaluate the final configuration once on the untouched test set;
6. move on to an equivalent implementation in PyTorch.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
