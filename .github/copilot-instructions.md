# Copilot instructions

## Repository overview

This repository is an Uzbek-language collection of standalone scikit-learn demonstrations. There is no package/module architecture, application entry point, dependency manifest, test suite, or configured formatter/linter. Each Python file is an executable script that creates its own synthetic dataset, trains models immediately at import/run time, prints metrics, and usually opens one or more Matplotlib figures.

- `main.py` demonstrates multiple linear regression for house price prediction. It creates `Maydon` (area), `Xonalar_soni` (room count), and `Narx` (price), splits the data into train/test sets, standardizes features, and reports MSE, RMSE, and R².
- `regressions.py` demonstrates polynomial regression for synthetic gas prices. It plots univariate models for degrees 1, 3, and 6, then evaluates multivariate polynomial models for degrees 2 and 5 using MSE and R². Its reported evaluation is in-sample on the generated data.
- `classifications.py` compares KNN, linear SVM, decision tree, and random forest classifiers for synthetic teacher categories. It reports train/test accuracy and displays a confusion matrix for each classifier.
- `logistic_regression.py` compares random forest and logistic regression on a synthetic teacher-category dataset, with accuracy output and a logistic-regression confusion matrix.
- `README.md` is the conceptual documentation, written primarily in Uzbek, covering regression types, polynomial degree, overfitting, and the example results.

The scripts do not share data or helper functions. Changes to model behavior should therefore be made in the specific demonstration script rather than assuming a shared pipeline.

## Running the code

The scripts require Python with NumPy, pandas, Matplotlib, scikit-learn, and (for `classifications.py` and `logistic_regression.py`) seaborn installed. No `requirements.txt` or other dependency manifest is currently checked in.

Run an individual demonstration from the repository root:

```bash
python3 main.py
python3 regressions.py
python3 classifications.py
python3 logistic_regression.py
```

For a headless environment such as CI, use a non-interactive Matplotlib backend:

```bash
MPLBACKEND=Agg python3 regressions.py
```

There is no automated test suite. The closest focused check for a single script is to compile it without executing plots or model code:

```bash
python3 -m py_compile main.py
```

Replace `main.py` with the script being changed. To syntax-check every script:

```bash
python3 -m py_compile main.py regressions.py classifications.py logistic_regression.py
```

No build or lint command is configured. When validating a behavioral change, run the affected script directly and inspect both its printed metrics and generated plots.

## Conventions and implementation details

- Synthetic data is generated at runtime rather than loaded from files. Preserve the existing `np.random.seed(42)` calls when reproducibility matters.
- The scripts use top-level procedural code, short explanatory comments, and printed results instead of functions, classes, or a command-line interface. Follow that structure unless the repository is intentionally being refactored.
- Domain labels, column names, comments, plot titles, and console output are generally Uzbek or Uzbek-transliterated. Keep user-visible terminology consistent with the surrounding script.
- Scikit-learn estimators are fit directly on NumPy arrays or pandas selections. `main.py` is the exception that explicitly standardizes features before fitting; do not assume scaling is applied in the other examples.
- Train/test splitting uses a fixed `random_state=42` where present. Preserve the split and seed when comparing model changes so results remain comparable.
- Plots are part of each example's output and are displayed with `plt.show()`. Avoid removing visualization code when changing model calculations; use `MPLBACKEND=Agg` for non-interactive validation instead.
- Polynomial examples construct `PolynomialFeatures` inside each degree loop and fit a fresh `LinearRegression` model per degree. Classification examples likewise fit separate estimators and collect train/test metrics explicitly.
- Existing evaluations are instructional demonstrations, not a shared production-quality evaluation framework. In particular, `regressions.py` evaluates predictions on the same generated data used for fitting.
