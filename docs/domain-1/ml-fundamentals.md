# ML Fundamentals

*Covers task statement 1.5: familiarity with the fundamentals of machine learning
(feature engineering, model comparison, cross-validation).*

The exam is about generative AI, so it is tempting to skip the classical material. Do not.
NVIDIA names feature engineering, model comparison and cross-validation explicitly, and the
concepts here — overfitting, leakage, validation strategy, baselines — reappear constantly in
LLM questions wearing different clothes. When a question asks why a RAG evaluation is
unreliable, the answer is usually a classical one.

---

## 1. What "learning from data" actually means

A machine learning model is a function with adjustable knobs. You have some inputs `X` and
some desired outputs `y`, and you want a function `f` such that `f(X) ≈ y`. **Training** is
the search for knob settings that make the approximation good.

That framing is worth holding onto because it makes the failure modes obvious. There are only
a few ways this can go wrong: the function family might be too simple to represent the true
relationship, it might be so flexible that it memorises the training data instead of learning
the pattern, or the data you trained on might not resemble the data you will face later.
Almost every practical ML problem is one of those three.

Traditional programming and machine learning are inverses of each other:

```text
Traditional programming:     rules  +  data   ──►  answers
Machine learning:            data   +  answers ──►  rules
```

You do not write the rule that distinguishes spam from not-spam. You supply thousands of
examples of each, and the training procedure infers the rule. The consequence — and this is
the source of most trouble in the field — is that **the rule the model infers is only as good
as the examples you gave it**, and you cannot inspect that rule directly to check.

---

## 2. The learning paradigms

### Supervised learning

You have **labeled** data: pairs of `(input, correct output)`. The model makes a prediction,
you compare it to the label, and you adjust the parameters to reduce the difference.

```text
X (features)                          y (label)
────────────────────────────────────  ─────────
"Free money click here now"           spam
"Meeting moved to 3pm"                not spam
"You have won a prize"                spam
```

Algorithms: linear and logistic regression, decision trees, random forests, **XGBoost**,
support vector machines, neural networks.

The binding constraint is labels. They are expensive — someone has to produce them — and their
quality caps everything downstream. A model trained on noisy labels cannot be more accurate
than the labels themselves.

### Unsupervised learning

No labels. You have `X` and want to discover structure in it.

- **Clustering** (k-means, DBSCAN, hierarchical) — group similar items together.
- **Dimensionality reduction** (PCA, t-SNE, UMAP) — find a lower-dimensional representation.
- **Anomaly detection** — identify records unlike the rest.
- **Association rules** — find things that co-occur.

Nobody tells the algorithm what the right answer is, which means there is no clean way to
measure whether it found the right one. Evaluating unsupervised results usually requires human
judgement or a downstream task.

### Reinforcement learning

An **agent** takes **actions** in an **environment** and receives a **reward** signal. It
learns a **policy** — a strategy for choosing actions — that maximises cumulative reward over
time.

The distinguishing feature is that there is no correct answer to compare against, only a
scalar signal saying "that went well" or "that went badly", often delayed. This is what makes
RL hard and why it appears in LLM work only at the alignment stage, where **PPO** optimises a
model against a learned reward. See [Alignment & RLHF](../domain-3/rlhf-alignment.md).

### Self-supervised learning — the one that made LLMs possible

This is the paradigm the exam actually cares about, and it deserves proper explanation.

The bottleneck in supervised learning is labels. Self-supervised learning removes the
bottleneck by **manufacturing labels from the data itself**. No human annotation is involved,
so the amount of training data available is limited only by how much text exists.

The trick for language is almost embarrassingly simple. Take any sentence:

```text
"The cat sat on the mat"
```

Now hide part of it and make predicting the hidden part the task:

```text
GPT-style (next-token prediction)          BERT-style (masked language modeling)
─────────────────────────────────          ─────────────────────────────────────
input: "The"              → "cat"          input: "The cat [MASK] on the mat"
input: "The cat"          → "sat"          target: "sat"
input: "The cat sat"      → "on"
input: "The cat sat on"   → "the"
input: "The cat sat on the" → "mat"
```

Every sentence in the corpus becomes dozens of training examples, and the "labels" were
already there — they are just the words themselves. A trillion-token corpus yields a trillion
supervised examples for free.

!!! important "Why this is the whole ballgame"
    To predict the next word reliably across billions of documents, a model has to learn
    syntax, semantics, world facts, reasoning patterns, code structure and style. None of that
    is explicitly taught. It is all a **side effect** of getting good at a task whose labels
    are free.

    That is the answer to "why did LLMs happen when they did": the objective scaled to
    unlimited data, and the transformer made that data trainable on GPUs.

**Semi-supervised learning** is the middle ground — a small labeled set plus a large unlabeled
one, using the unlabeled data to learn structure and the labeled data to attach meaning to it.

!!! tip "Exam reflex"
    "LLM pretraining is an example of which type of learning?" → **self-supervised**. It is
    sometimes described as a special case of unsupervised learning, since no *human* labels
    exist. It is explicitly **not** supervised, and it is **not** reinforcement learning —
    that comes later, during alignment.

---

## 3. Classification vs. regression

A distinction that determines which metrics and which loss functions apply.

| | **Classification** | **Regression** |
| --- | --- | --- |
| Predicts | A discrete class | A continuous number |
| Examples | Spam/not-spam, sentiment, which of 57 topics | House price, temperature, latency in ms |
| Output layer | Sigmoid (binary) or softmax (multi-class) | A single linear unit |
| Typical loss | Cross-entropy | MSE, MAE, Huber |
| Typical metrics | Accuracy, precision, recall, F1, ROC-AUC | MSE, RMSE, MAE, **R²** |

Note that **R²** is named directly in the blueprint as "proportion of explained variance", and
that **language modeling is technically classification** — a multi-class problem over the
whole vocabulary, trained with cross-entropy. That is why perplexity, which is exponentiated
cross-entropy, is the language modeling metric. See
[Evaluation Metrics](../domain-3/evaluation-metrics.md).

---

## 4. Feature engineering

A **feature** is one measurable input to the model. **Feature engineering** is the work of
turning raw data into features the model can actually learn from.

For a long time this was where most of the value in a machine learning project lived. Deep
learning reduced its importance for images and text — a neural network learns its own
representations — but it remains critical for tabular data, and the blueprint names it
explicitly.

### Scaling and standardization — and why they matter more than they look

Suppose you are predicting loan default from two features:

```text
                  age     annual_salary
customer A        25          50,000
customer B        35          52,000
```

A k-nearest-neighbours model computes Euclidean distance between customers:

```text
distance = √( (35 − 25)²  +  (52000 − 50000)² )
         = √( 100         +  4,000,000 )
         = √4,000,100
         ≈ 2000.02
```

Look at the two contributions. Age contributed 100 to the sum; salary contributed 4,000,000.
The age difference is **completely invisible** — it changes the distance by 0.025%. The model
is not ignoring age because age is unimportant; it is ignoring it because salary is measured
in bigger numbers.

Two standard fixes:

- **Min-max normalization** rescales to `[0, 1]`: `x' = (x − min) / (max − min)`. Preserves
  the shape of the distribution; sensitive to outliers, since one extreme value squashes
  everything else.
- **Standardization (z-score)** rescales to mean 0 and standard deviation 1:
  `x' = (x − μ) / σ`. The more common default. Handles outliers better and is what most
  algorithms implicitly expect.

**Which models need it:** anything based on distances (k-NN, k-means, SVM with RBF kernel) and
anything trained by gradient descent (neural networks, logistic regression) — in the latter
case because wildly different feature scales produce an elongated loss surface that gradient
descent traverses very slowly.

**Which models do not:** decision trees and their ensembles (random forest, XGBoost). A tree
splits on thresholds within a single feature at a time, so the relative scale of different
features is irrelevant to it.

### Encoding categorical variables

Models consume numbers, so categories must be converted. How you convert matters.

**One-hot encoding** creates one binary column per category:

```text
colour                red   green   blue
─────                 ───   ─────   ────
"red"        ──►       1      0      0
"green"      ──►       0      1      0
"blue"       ──►       0      0      1
```

Safe default. Every category is equidistant from every other, which is exactly right when
there is no inherent ordering. The cost is width: a "city" column with 5,000 values becomes
5,000 columns, most of them zero almost always.

**Label / ordinal encoding** assigns an integer per category:

```text
"red" → 0,  "green" → 1,  "blue" → 2
```

This is compact, and it is **wrong** unless the ordering is real. A linear model now believes
that green sits exactly between red and blue, and that blue is "twice" green. It will
confidently learn relationships that do not exist. Use ordinal encoding only when the order is
genuine: `small < medium < large`, `bronze < silver < gold`.

**Target (mean) encoding** replaces each category with the mean of the target for that
category — "customers in this city default 4.2% of the time". Powerful for high-cardinality
features, and dangerous: it uses the target variable, so if you compute it on the full dataset
before splitting you have leaked the answer into the features. It must be computed **inside**
each cross-validation fold.

**Embeddings** learn a dense vector per category during training. This is what scales to
enormous vocabularies, and it is the direct ancestor of the [text embeddings](embeddings.md)
that power RAG.

### Text features: the classical pipeline

Before transformers, turning text into numbers meant this.

**Bag of Words** counts how often each vocabulary word appears in each document. Order is
discarded entirely — "dog bites man" and "man bites dog" produce identical vectors.

**TF-IDF** improves on raw counts by asking not just "how often does this word appear here?"
but "how *distinctive* is this word?".

$$ \text{tfidf}(t, d) = \underbrace{\text{tf}(t, d)}_{\text{frequency in this document}} \times \underbrace{\log\frac{N}{\text{df}(t)}}_{\text{rarity across the corpus}} $$

Work through it with three documents:

```text
d1: "the GPU accelerates training"
d2: "the model requires training data"
d3: "the restaurant serves good food"
```

The word **"the"** appears in all 3 documents, so `df = 3` and:

```text
idf("the") = log(3 / 3) = log(1) = 0
```

Its IDF is **zero**, so its TF-IDF weight is zero in every document. A word that appears
everywhere carries no information about which document you are looking at, and the formula
kills it automatically — no stop-word list required.

The word **"GPU"** appears in 1 document, so `df = 1`:

```text
idf("GPU") = log(3 / 1) = log(3) ≈ 1.099
```

High weight. "GPU" is highly informative about which document you are in.

**n-grams** partially restore word order by treating adjacent pairs or triples as single
features. `ngram_range=(1,2)` captures "not good" as its own feature, which matters enormously
for sentiment — unigrams alone would see "good" and get the polarity backwards.

!!! warning "Do not apply this pipeline before a transformer"
    Lowercasing, stop-word removal and stemming are **classical** preprocessing. Transformers
    are trained on raw natural text and rely on function words, casing and full morphology.
    Feeding BERT stemmed, stop-word-stripped text degrades it. Use the model's own tokenizer
    and nothing else. See [Data Preprocessing](../domain-4/data-preprocessing.md).

### Missing values

Every strategy encodes an assumption:

| Strategy | Assumption |
| --- | --- |
| Drop the rows | The missing rows are a small, unbiased subset |
| Drop the column | The column is mostly empty and not worth rescuing |
| Mean imputation | The distribution is roughly symmetric |
| Median imputation | The distribution is skewed or outlier-prone |
| Mode imputation | The variable is categorical |
| Forward/backward fill | The data is a time series and adjacent values are similar |
| Model-based (k-NN, iterative) | Missingness is predictable from other columns |
| **Add a "was missing" indicator** | **The fact of missingness is itself informative** |

That last one is the most commonly forgotten and often the most valuable. If income is missing
disproportionately for people who declined to answer, "declined to answer" is a real signal —
imputing the mean throws it away.

### Feature selection and dimensionality reduction

More features is not better. Irrelevant features add noise, increase overfitting risk, slow
training, and — in high dimensions — dilute distance measures until nearest-neighbour methods
stop working (the "curse of dimensionality").

**Feature selection** keeps a subset of the original columns:

- **Filter** methods score each feature independently (correlation with the target, chi²) and
  keep the best. Fast, ignores interactions.
- **Wrapper** methods (recursive feature elimination) train the model repeatedly with different
  subsets. Accurate, expensive.
- **Embedded** methods get selection for free from the model: **L1/Lasso** regularization drives
  some coefficients to exactly zero; tree ensembles report feature importances.

**Dimensionality reduction** creates new, fewer features from combinations of the old:

- **PCA** finds the orthogonal directions of maximum variance and projects onto the top few.
  Linear, deterministic, and **safe to use as model input**.
- **t-SNE** and **UMAP** are non-linear and excellent at revealing cluster structure — but they
  are **for visualization only**. Their axes are meaningless, distances between clusters are
  not interpretable, and different runs give different layouts. Never feed their output into a
  model. This distinction is a favourite exam trap; see
  [EDA & Visualization](../domain-4/eda-visualization.md).

---

## 5. Underfitting, overfitting, and the bias–variance trade-off

### The picture

Imagine fitting a curve to ten noisy data points that were generated by a gentle quadratic.

- Fit a **straight line**. It cannot bend, so it misses the curvature. It is wrong on the
  training data *and* wrong on new data. This is **underfitting** — the model family is too
  simple to represent reality.
- Fit a **9th-degree polynomial**. With ten points and ten coefficients it can pass exactly
  through every single one. Training error is **zero**. But between the points it swings
  wildly, because it has fitted the *noise* as if it were signal. On new data it is terrible.
  This is **overfitting**.
- Fit a **quadratic**. Small training error, small error on new data. This is the right
  capacity.

The signature is the *gap*:

| | Underfitting | Good fit | Overfitting |
| --- | --- | --- | --- |
| Training error | High | Low | Very low / zero |
| Validation error | High | Low | High |
| **Gap between them** | Small | Small | **Large** |
| Bias | High | Balanced | Low |
| Variance | Low | Balanced | High |

### Bias and variance, defined properly

Imagine training your model many times on different random samples from the same population.

- **Bias** is how far the *average* prediction is from the truth. High bias means the model is
  systematically wrong in the same direction — it is too rigid to capture the real pattern.
- **Variance** is how much predictions *change* from one training sample to another. High
  variance means the model is chasing whatever noise happened to be in this particular sample.

Total expected error decomposes as:

$$ \text{Error} = \text{Bias}^2 + \text{Variance} + \text{irreducible noise} $$

That third term is the noise inherent in the data. No model can beat it, and a model claiming
to has almost certainly leaked.

**Increasing model capacity lowers bias and raises variance.** That is the trade-off. Adding
layers, adding features, training longer — all move you rightward along the same axis.

### Diagnosing from learning curves

Plot training and validation error against training-set size or epochs:

```text
  error                                error
    │╲                                   │╲
    │ ╲___________ validation            │ ╲_______________ validation
    │ ╱‾‾‾‾‾‾‾‾‾‾‾ training              │      ╲
    │╱                                   │       ╲__________ training
    └──────────────► data                └──────────────────► data
    both high, converged                 large persistent gap
    → UNDERFITTING                       → OVERFITTING
    more data will not help              more data will help
```

That right-hand conclusion is practically important: **if the curves have converged and both
are bad, collecting more data is a waste of money.** You need a better model, better features,
or a reformulated problem. If there is still a wide gap, more data is the single most reliable
fix.

### Fixing each

**Underfitting** — increase capacity (deeper, wider, more trees), add or improve features,
reduce regularization, train longer.

**Overfitting** — more training data, data augmentation, **regularization** (L1/L2, weight
decay), **dropout**, **early stopping**, a simpler model, or ensembling. These are covered
mechanically in [Neural Networks](neural-networks.md).

!!! warning "The classic exam signature"
    *"Training accuracy 99%, validation accuracy 62%."* → **overfitting**, every time. The
    correct remedies are regularization, more data, dropout, early stopping. Distractors will
    offer "increase model depth" and "train for more epochs", both of which make it worse.

---

## 6. Data splits and the cardinal sin of leakage

### The three sets

| Set | Purpose | How often you touch it |
| --- | --- | --- |
| **Training** | The model learns its parameters from this | Constantly |
| **Validation (dev)** | Tune hyperparameters, choose architecture, decide when to stop | Many times |
| **Test** | Estimate performance on genuinely unseen data | **Once, at the very end** |

Typical proportions: 70/15/15 or 80/10/10, with more data allowing a smaller proportion held
out (with a million rows, 1% is 10,000 examples — plenty).

Why three and not two? Because tuning against a set *uses it up*. Each time you look at
validation performance and change something in response, you fit a little bit of your
decision-making to that specific set. After fifty experiments, validation score is
optimistically biased. The test set exists to give you one honest number at the end,
uncontaminated by that process. **If you look at the test set and then change something, it
has become a validation set and you no longer have a test set.**

### Data leakage

**Leakage** is any situation where information that would not be available at prediction time
finds its way into training. It is the single most common cause of results that look
spectacular and then collapse in production.

Concrete examples, each of which has sunk real projects:

**Preprocessing fitted before splitting.** You call `StandardScaler().fit(X)` on the whole
dataset, then split. The mean and standard deviation used to scale your training data were
computed partly from the test set. The test set has influenced training.

**Target encoding computed globally.** Same shape of error, but worse — you have injected the
target variable itself into the features.

**Duplicate rows across splits.** The same record, or a near-duplicate, appears in both
training and test. The model has memorised the answer. Endemic in scraped text corpora, which
is why deduplication is mandatory for LLM data.

**Shuffling a time series.** You shuffle rows and split randomly, so the model trains on
Tuesday and Thursday and is tested on Wednesday. It has seen the future. Any temporal data
must be split **chronologically**.

**A feature that encodes the answer.** The classic: predicting whether a patient has a disease
from a dataset that includes `treatment_prescribed`. Or predicting churn from a field that is
only populated after the customer churns.

**Group leakage.** The same patient contributes multiple rows, some in training and some in
test. The model recognises the patient, not the condition.

!!! danger "The heuristic that saves careers"
    **Results that are surprisingly good are leakage until proven otherwise.** 99.8% accuracy
    on a genuinely hard problem is not a breakthrough; it is a bug. Before you present it, go
    and find the leak.

The structural defence is **scikit-learn's `Pipeline`**, which chains preprocessing and the
model into one object so that every transform is fitted *inside* each fold automatically. It
is not a convenience feature; it is a correctness feature.

---

## 7. Cross-validation

Named explicitly in task 1.5, and NVIDIA lists "Cross-Validation in Machine Learning" in its
suggested reading.

### The problem it solves

A single train/validation split gives you exactly one number, and that number depends on which
rows happened to land where. With a small dataset the variation is large. You might measure
84% and conclude your model is good, when a different random split would have given 76%.

### The mechanism

**k-fold cross-validation** splits the data into `k` equal parts. Then, `k` times, it trains on
`k−1` parts and validates on the held-out part, rotating which part is held out:

```text
5-fold cross-validation on 1000 rows (200 per fold)

fold 1:  [VALID][train][train][train][train]  →  score 0.81
fold 2:  [train][VALID][train][train][train]  →  score 0.79
fold 3:  [train][train][VALID][train][train]  →  score 0.85
fold 4:  [train][train][train][VALID][train]  →  score 0.78
fold 5:  [train][train][train][train][VALID]  →  score 0.82
                                                 ──────────
                                        mean = 0.810,  std = 0.026
```

Every row is used for validation exactly once and for training `k−1` times. `k = 5` or `k = 10`
are the conventional choices.

### What you actually gain

Two things, and the second is the one people forget.

**A lower-variance estimate.** Averaging five measurements is more stable than taking one.

**A measure of uncertainty.** The standard deviation across folds tells you how much your
estimate can be trusted. This is what lets you say whether two models genuinely differ.

Suppose a competing model scores `0.83 ± 0.030`. Is it better than your `0.810 ± 0.026`? The
means differ by 0.02 while the fold-to-fold variation is 0.026–0.030. **You cannot conclude
anything.** Reporting only "0.83 beats 0.81" would be reporting noise as a result — and this
is exactly the mistake the Experimentation domain is designed to test.

The cost is that you train `k` models instead of one.

### The variants, and when each is required

| Variant | Use when | Why |
| --- | --- | --- |
| **Stratified k-fold** | Classification, especially imbalanced | Preserves the class proportions in every fold. Without it, a rare class may be absent from a fold entirely, making that fold's score meaningless |
| **Group k-fold** | Rows are correlated by group (same patient, same user, same document) | Keeps an entire group inside one fold, so the model cannot recognise the group rather than the pattern |
| **Time-series split** | Temporal data | Always trains on the past and validates on the future. **Never shuffle** — that leaks the future into the past |
| **Leave-one-out (LOOCV)** | Very small datasets | `k = n`. Nearly unbiased, very high variance, and extremely expensive |
| **Nested CV** | You need to tune hyperparameters *and* report an unbiased score | An inner loop tunes; an outer loop evaluates. Prevents the tuning process from contaminating the estimate |

Time-series split deserves the picture, because "just use k-fold" is a genuinely dangerous
default here:

```text
split 1:  [train][valid][ ... unused ... ]
split 2:  [train][train][valid][ ... unused ... ]
split 3:  [train][train][train][valid][ ... ]
          ────────────────────────────► time
```

### Cross-validation and LLMs

!!! danger "A distinction worth knowing"
    Full k-fold cross-validation requires training `k` complete models. For a foundation model
    costing millions of GPU-hours, that is absurd. LLMs use a **single fixed held-out
    validation set**, and evaluation happens on separate benchmark suites.

    Know cross-validation as a **classical ML** technique. If a question describes
    cross-validating a large language model's pretraining, something is wrong with the premise
    — and that distinction is itself exam-worthy.

---

## 8. Comparing models honestly

Tasks 1.5 and 2.2 both mention model comparison. A defensible comparison has six components.

**1. Identical conditions.** Same splits, same preprocessing, same random seed where possible.
If model A saw a different train/test partition than model B, the comparison is void.

**2. A metric that matches the goal.** Accuracy is the wrong metric on imbalanced data — a
model predicting "negative" always scores 99% on a 1%-positive dataset while catching nothing.
Use F1, PR-AUC, or recall when the positive class is rare and matters. The choice between
precision and recall is a **business decision** about which error costs more. See
[Evaluation Metrics](../domain-3/evaluation-metrics.md).

**3. A baseline.** Always. Majority-class predictor, random, or a simple logistic regression on
raw features. A sophisticated model that cannot beat "always predict the most common class" is
not a model, and you would be astonished how often that happens undetected.

**4. Variance awareness.** Report mean ± standard deviation across folds or seeds. A 0.3%
improvement inside a 2% standard deviation is nothing. Where it matters, use a proper
statistical comparison — a paired t-test or Wilcoxon signed-rank across folds, or McNemar's
test for two classifiers on the same test set.

**5. Cost accounting.** Latency, memory, GPU-hours, dollars per thousand tokens, engineering
complexity. The best model on a leaderboard is frequently unshippable, and in LLM work the
cost difference between candidates can be 50×.

**6. Error analysis.** Look at *where* each model fails, not just how often. Two models with
identical F1 can fail on completely different cases, and one of those failure sets may be far
more damaging than the other. Aggregate metrics hide this by construction — the same argument
that motivates [disaggregated evaluation](../domain-5/bias-fairness.md) for fairness.

**Parsimony.** When two models perform equivalently, prefer the simpler, cheaper, more
interpretable one. Occam's razor is an engineering principle as well as a philosophical one:
simpler models are easier to debug, cheaper to serve, and less likely to break in novel ways.

---

## 9. Recap

- Machine learning infers rules from data and answers; the rule is only as good as the
  examples, and you cannot inspect it directly.
- **Supervised** needs labels; **unsupervised** finds structure; **reinforcement** learns from
  reward; **self-supervised** manufactures labels from the data itself — which is what made
  LLM pretraining possible at trillion-token scale.
- **Scaling** matters for distance- and gradient-based models, not for trees. **One-hot** for
  unordered categories; **ordinal** only when the order is real.
- **TF-IDF** = frequency × rarity; a word appearing in every document gets an IDF of exactly
  zero.
- **Overfitting** = low training error, high validation error, large gap. Fix with
  regularization, dropout, early stopping, more data — not with more capacity.
- Bias² + variance + irreducible noise; capacity trades bias for variance. Learning curves tell
  you which problem you have, and therefore whether more data will help.
- **Leakage** is the most common cause of results that are too good to be true. Fit every
  transform inside the fold; split time series chronologically; deduplicate.
- **k-fold CV** gives a lower-variance estimate **and a standard deviation** — the second is
  what lets you claim one model beats another. Stratify for classification, group for
  correlated rows, never shuffle time series.
- Compare models on identical splits, against a baseline, with variance reported and cost
  accounted for.
