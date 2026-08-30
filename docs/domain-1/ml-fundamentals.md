# ML Fundamentals

*Covers task statement 1.5: familiarity with the fundamentals of machine learning
(feature engineering, model comparison, cross-validation).*

## The three learning paradigms

| Paradigm | Data | Goal | Typical algorithms |
| --- | --- | --- | --- |
| **Supervised** | Labeled `(X, y)` | Predict `y` for new `X` | Linear/logistic regression, decision trees, random forest, **XGBoost**, SVM, neural nets |
| **Unsupervised** | Unlabeled `X` | Find structure | k-means, DBSCAN, PCA, t-SNE/UMAP, topic models |
| **Reinforcement** | Environment + reward | Learn a policy that maximises reward | Q-learning, PPO (used in [RLHF](../domain-3/rlhf-alignment.md)) |

Two more you should recognise by name:

- **Semi-supervised** — a small labeled set plus a large unlabeled set.
- **Self-supervised** — labels are *derived from the data itself*. This is how LLMs are
  pretrained: predict the next token (GPT-style) or a masked token (BERT-style). No
  human annotation is required, which is exactly why pretraining can consume trillions
  of tokens.

!!! tip "Exam reflex"
    "LLM pretraining is an example of which kind of learning?" → **self-supervised**
    (a special case of unsupervised in the sense that no human labels exist).

## Classification vs. regression

- **Classification** predicts a discrete class → measured with accuracy, precision,
  recall, F1, ROC-AUC.
- **Regression** predicts a continuous value → measured with MSE, RMSE, MAE, R²
  (the *proportion of explained variance*, named explicitly in the blueprint).

See [Statistics & Performance Metrics](../domain-4/statistics.md) for the full metric
reference.

## Feature engineering

Turning raw data into inputs a model can learn from. Still highly examinable even in an
LLM context, because task 1.5 and 1.10 name it directly.

**Numerical**

- **Scaling / normalization** — min-max to `[0,1]`; **standardization** to mean 0,
  std 1. Required for distance- and gradient-based models (k-NN, SVM, neural nets);
  irrelevant for tree ensembles.
- **Binning / discretisation** — continuous → buckets.
- **Log transforms** — tame skewed distributions.

**Categorical**

- **One-hot encoding** — one binary column per category. Safe default; explodes in
  width with high cardinality.
- **Label / ordinal encoding** — integer per category. Only valid when order is
  meaningful, otherwise the model invents a false ranking.
- **Target / mean encoding** — replace the category with the mean target. Powerful,
  leaks easily; must be fit inside cross-validation folds.
- **Embeddings** — learned dense vectors; the approach that scales to huge vocabularies
  and the direct ancestor of [text embeddings](embeddings.md).

**Text** (the classical pipeline, pre-transformers)

1. Tokenization → 2. lowercasing → 3. stop-word removal → 4. **stemming or
   lemmatization** → 5. vectorization.

- **Bag of Words (`CountVectorizer`)** — raw counts. Loses word order.
- **TF-IDF (`TfidfVectorizer`)** — term frequency × inverse document frequency.
  Down-weights words common to every document, up-weights distinctive ones.
- **n-grams** — capture short local order ("not good" ≠ "good").

!!! note "TF-IDF in one line"
    $\text{tfidf}(t,d) = \text{tf}(t,d) \times \log\frac{N}{\text{df}(t)}$ — a word that
    appears often *in this document* but rarely *across the corpus* gets the highest weight.

**Missing values** — drop rows/columns, or impute with mean/median/mode, or a
model-based imputation. The choice must be fit on training data only.

**Feature selection** — filter (correlation, chi²), wrapper (recursive feature
elimination) or embedded (L1/Lasso, tree importances) methods.

**Dimensionality reduction** — **PCA** (linear, preserves variance) for compression;
**t-SNE**/**UMAP** for *visualization only* — they do not preserve global distances and
should never be used as model input.

## Bias, variance and the fit spectrum

| | Underfitting | Good fit | Overfitting |
| --- | --- | --- | --- |
| Bias | High | Balanced | Low |
| Variance | Low | Balanced | High |
| Training error | High | Low | Very low |
| Validation error | High | Low | High |
| Fix | Bigger model, better features, train longer | — | More data, regularization, dropout, early stopping, simpler model |

The **bias–variance trade-off** is the classical framing: total error ≈ bias² +
variance + irreducible noise. Increasing model capacity lowers bias and raises variance.

!!! warning "The classic exam signature"
    *"Training accuracy 99%, validation accuracy 62%"* → **overfitting**. The remedies
    are regularization, more/augmented data, dropout, early stopping — **not** a bigger
    model.

## Data splits

- **Train** — the model learns from it.
- **Validation (dev)** — tune hyperparameters, choose architectures, early stopping.
- **Test** — touched **once**, at the very end, to estimate generalisation.

Typical split 70/15/15 or 80/10/10. Reusing the test set for tuning silently turns it
into a validation set and inflates your reported score.

**Data leakage** is the cardinal sin: any information from validation/test reaching
training. Common sources — scaling fit on the full dataset before splitting, target
encoding computed globally, duplicate rows across splits, time-series shuffled so the
future leaks into the past.

## Cross-validation

Explicitly named in task 1.5.

**k-fold cross-validation** — split data into *k* folds; train on *k−1*, validate on the
held-out fold; rotate; average the *k* scores. `k = 5` or `10` is standard.

*Why:* a single split gives one noisy estimate that depends on which rows landed where.
Averaging over folds gives a lower-variance estimate **and** a standard deviation you
can use to tell whether two models really differ.

Variants you should recognise:

| Variant | When |
| --- | --- |
| **Stratified k-fold** | Classification, especially imbalanced classes — preserves class proportions in each fold |
| **Leave-one-out (LOOCV)** | Very small datasets; *k* = *n*; extremely expensive |
| **Group k-fold** | Rows are correlated by group (same patient, same user) — keeps a group entirely inside one fold |
| **Time-series split** | Temporal data — always train on the past, validate on the future; **never shuffle** |
| **Nested CV** | Hyperparameter tuning *and* unbiased performance estimate at the same time |

!!! danger "Cross-validation is rarely used to train LLMs"
    Full k-fold requires training *k* models. For a foundation model costing millions of
    GPU-hours that is absurd — LLMs use a single fixed held-out validation set. Know
    cross-validation as a **classical ML** technique; that distinction itself is an
    exam-worthy nuance.

## Comparing models

Task 1.5 and 2.2 both mention model comparison. A defensible comparison requires:

1. **The same data splits** for every candidate.
2. **A metric that matches the business goal** — accuracy is misleading on imbalanced
   data; use F1, PR-AUC or recall when the positive class is rare.
3. **A baseline.** Majority class, random, or a simple logistic regression. A model that
   cannot beat the baseline is not a model.
4. **Variance awareness** — report mean ± std across CV folds or seeds. A 0.3% gap is
   noise.
5. **Cost accounting** — latency, memory, GPU-hours, $/1k tokens. The best model on the
   leaderboard may be unshippable.

**Occam's razor / parsimony:** when two models perform equivalently, prefer the simpler,
cheaper, more interpretable one.

## Key takeaways

- Feature engineering vocabulary — one-hot vs. ordinal, TF-IDF vs. bag-of-words,
  standardization vs. normalization, PCA vs. t-SNE.
- Overfitting = low training error + high validation error; fix with regularization,
  more data, dropout, early stopping.
- k-fold CV gives a lower-variance performance estimate; stratify for classification,
  never shuffle time series, group when rows are correlated.
- Data leakage invalidates every number you report — fit all transforms on train only.
- LLM pretraining is **self-supervised**, which is why it needs no human labels.
