# Statistics & Performance Metrics

*Covers tasks 2.2 ("compare models using statistical performance metrics, such as loss functions
or proportion of explained variance") and 2.5 ("identify relationships and trends or any factors
that could affect the results of research").*

Note the precision of those task statements. NVIDIA names **loss functions** and **proportion of
explained variance** (which is R²) explicitly, and task 2.5 is essentially asking "what could
make your conclusions wrong?" — which is a checklist worth memorising.

---

## 1. Describing a distribution

### Measures of centre

| Measure | What it is | When it lies |
| --- | --- | --- |
| **Mean** | The arithmetic average | **Dragged by outliers.** One billionaire in a sample of 100 people destroys the average income |
| **Median** | The middle value when sorted | Rarely lies. **Robust** — prefer it for skewed data |
| **Mode** | The most frequent value | The only option for categorical data |

!!! tip "Mean vs. median, in one diagnostic"
    ```text
    mean = 45,000    median = 28,000
    ```
    Mean far above median → the distribution is **right-skewed**, and a small number of large
    values is pulling the average up. Report the **median**, and consider a log transform before
    modeling.

    This is why **latency is always reported in percentiles**, never as an average. A mean
    latency of 400 ms is entirely compatible with 95% of users seeing 200 ms and 5% seeing 8
    seconds — and it is that 5% who file support tickets.

### Measures of spread

| Measure | What it is |
| --- | --- |
| **Variance** | Average squared deviation from the mean |
| **Standard deviation** | Square root of variance — same units as the data, so interpretable |
| **IQR** (Q3 − Q1) | The range of the middle 50%. **Robust to outliers**; the basis of the box plot |
| **Range** | Max − min. Determined entirely by the two most extreme points, so almost useless |

### Shape

- **Skewness** — asymmetry. **Right (positive) skew** means a long tail of large values, and it is
  extremely common in real data: incomes, response times, document lengths, request volumes.
- **Kurtosis** — tail heaviness. High kurtosis means more extreme values than a normal
  distribution would produce.
- **Percentiles** — position in the distribution. **p95 and p99 are what users actually
  experience**, which is why they belong on every latency dashboard.

---

## 2. Distributions worth recognising

**Normal (Gaussian).** Symmetric and bell-shaped; roughly 68% of values within 1 standard
deviation, 95% within 2, 99.7% within 3. Many statistical tests assume it.

**Uniform.** All values equally likely.

**Long-tailed / power law.** Extremely common in real data and in text specifically: word
frequencies follow **Zipf's law** (the *n*-th most common word occurs about 1/*n* as often as the
most common). Document lengths, user activity and request sizes are all long-tailed. **The mean
of a power-law distribution is close to meaningless** — it is dominated by the tail and is not
representative of any typical value.

**Bimodal.** Two peaks, almost always a sign of two mixed subpopulations. The correct response is
to split them and analyse separately, not to report a mean that describes neither group.

**Central limit theorem.** The distribution of the **sample mean** approaches normal as the sample
grows, *regardless of the underlying distribution*. This is what makes A/B test statistics valid
even when the underlying metric is wildly skewed — and it is why A/B tests need adequate sample
sizes, since the approximation is poor for small *n* on very skewed data.

---

## 3. Relationships between variables (task 2.5)

### Correlation

| Coefficient | What it measures | Use when |
| --- | --- | --- |
| **Pearson** *r* | **Linear** relationship | Continuous, roughly normal, no severe outliers |
| **Spearman** ρ | **Monotonic** relationship (computed on ranks) | Non-linear but monotonic, ordinal data, or outliers present |
| **Kendall** τ | Rank concordance | Small samples, many ties |

`r` ranges from −1 to +1: −1 is a perfect inverse relationship, 0 no *linear* relationship, +1
perfect direct.

### The two things correlation cannot tell you

!!! danger "1. Correlation is not causation"
    Three alternative explanations for any correlation between X and Y, all as consistent with
    the data as "X causes Y":

    - **Y causes X** (reverse causation)
    - **A third variable Z causes both** (confounding) — the classic being ice cream sales and
      drowning deaths, both caused by summer
    - **Coincidence** — with enough variables tested, some will correlate by chance

    **Only a randomised controlled experiment establishes causation**, because randomisation
    breaks the link between the treatment and every possible confounder. That is exactly what an
    [A/B test](../domain-3/experiment-design.md) is.

!!! danger "2. r = 0 does not mean independent"
    Pearson's *r* measures **linear** relationships only. A perfect U-shaped relationship — where
    Y is entirely determined by X — produces r ≈ 0:

    ```text
      Y │  ●                    ●
        │    ●                ●
        │      ●            ●
        │        ●   ●   ●
        └────────────────────────► X

      r ≈ 0, and yet Y is a perfect function of X
    ```

    **Always plot the scatter.** Anscombe's quartet is the canonical demonstration: four datasets
    with identical means, variances and correlation coefficients that look completely different
    when plotted.

### Other traps in this territory

**Simpson's paradox** — a trend present in every subgroup **reverses** when the groups are pooled.
This is not a curiosity; it happens in real analyses regularly, and it means you must **segment
before concluding**. It is also structurally the same argument that motivates
[disaggregated evaluation](../domain-5/bias-fairness.md) for fairness.

**Multicollinearity** — highly correlated predictors make coefficients unstable and
uninterpretable (predictions may still be fine, but you cannot say "feature X contributes this
much"). Detect with a correlation heatmap or VIF; fix by dropping, combining or regularising.

---

## 4. Loss functions (task 2.2)

The blueprint names loss functions explicitly as a means of **comparing models**.

| Loss | Formula | Character |
| --- | --- | --- |
| **MSE (L2)** | `mean((y − ŷ)²)` | Penalises large errors **quadratically**; outlier-sensitive; smooth everywhere |
| **RMSE** | `√MSE` | Same units as the target — much easier to interpret |
| **MAE (L1)** | `mean(|y − ŷ|)` | **Robust to outliers**; all errors weighted equally |
| **Huber** | Quadratic near 0, linear in the tails | The compromise between MSE and MAE |
| **Cross-entropy** | `−Σ y log ŷ` | Classification and **language modeling** |
| **Hinge** | `max(0, 1 − y·ŷ)` | SVMs |
| **Contrastive / triplet** | — | Embedding and retrieval training |

### MSE vs. MAE, worked

Suppose two models make these errors on five examples:

```text
Model A errors:  [2, 2, 2, 2, 2]        Model B errors:  [0, 0, 0, 0, 10]

MAE(A) = 10/5 = 2.0                     MAE(B) = 10/5 = 2.0        ← identical
MSE(A) = 20/5 = 4.0                     MSE(B) = 100/5 = 20.0      ← 5× worse
```

Both models have the same total absolute error. **MAE says they are equivalent. MSE says B is
five times worse.**

Which is right depends entirely on your problem:

- If **one large error is much worse than several small ones** — a delivery estimate off by three
  hours, a dosage calculation — use **MSE**.
- If **outliers are real data you do not want to chase** — a few genuinely extreme values that the
  model should not distort itself to fit — use **MAE**.

This is a design decision, not a statistical one, and it appears as a question.

---

## 5. R² — the proportion of explained variance

The blueprint names it in exactly those words.

$$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2} $$

Read the two sums:

- `SS_tot` is the error you would make by **always predicting the mean** — the baseline.
- `SS_res` is the error your model actually makes.

So R² is *"what fraction of the baseline error did I eliminate?"*

| R² | Meaning |
| --- | --- |
| **1.0** | The model explains all variance — perfect fit |
| **0.7** | The model explains 70% of the variance in the target |
| **0.0** | The model does exactly as well as predicting the mean — useless |
| **< 0** | The model is **worse than predicting the mean** |

!!! warning "Negative R² is possible and is a red flag"
    It cannot happen on the training data for ordinary least squares, but it happens routinely on
    **held-out data** — and it means your model is actively worse than a constant. If you see it,
    something is badly wrong: distribution shift, severe overfitting, or a bug.

**Adjusted R²** penalises adding predictors. Plain R² can only **increase** when you add a
feature, even a column of random numbers — so it always favours the more complex model. Use
**adjusted R² when comparing models with different numbers of features**.

---

## 6. Comparing models rigorously

The six components of a defensible comparison, extending
[ML Fundamentals](../domain-1/ml-fundamentals.md#8-comparing-models-honestly):

**1. Identical conditions.** Same splits, same preprocessing, same seed where possible.

**2. A baseline.** Mean or majority predictor, or a simple linear model. Always.

**3. Variance reported.** Mean ± standard deviation across folds or seeds:

```text
Model A:  0.810 ± 0.026
Model B:  0.830 ± 0.030

The means differ by 0.020. The fold-to-fold variation is 0.026–0.030.
→ You cannot conclude that B is better. Reporting "B beats A" here is
  reporting noise as a result.
```

**4. A statistical test where it matters.** A **paired t-test** or **Wilcoxon signed-rank** across
folds (paired, because both models saw the same folds). **McNemar's test** for two classifiers on
the same test set.

**5. More than one metric.** Quality, latency, memory, cost, fairness. In LLM work the cost
difference between candidates can be 50×, which frequently outweighs a small quality difference.

**6. Error analysis.** *Where* does each model fail? Two models with identical F1 can fail on
completely different, and differently important, cases.

**Parsimony.** Equal performance → prefer the simpler, cheaper, more interpretable model.

---

## 7. Factors that can invalidate results (task 2.5)

The task statement asks you to "identify relationships and trends **or any factors that could
affect the results of research**". Here is that checklist.

| Factor | What goes wrong |
| --- | --- |
| **Data leakage** | Information from validation/test reaches training. **The most common cause of results that are too good to be true** |
| **Selection bias** | The sample does not represent the population you will deploy on |
| **Survivorship bias** | Only the successes are in the data — you analyse the planes that came back |
| **Confounding variables** | An unmeasured cause of both variables under study |
| **Sampling / non-response bias** | Who chose to be in the data is not random |
| **Label noise** | Low [inter-annotator agreement](../domain-3/rlhf-alignment.md#inter-annotator-agreement) caps achievable accuracy |
| **Distribution shift** | Training data no longer resembles production data |
| **Temporal effects** | Seasonality, day-of-week, trend — an experiment run over one week may not generalise |
| **Small sample size** | Underpowered: a real effect goes undetected (Type II error) |
| **Multiple comparisons** | Test twenty metrics and one will look significant by chance |
| **Benchmark contamination** | The test set was in the model's training data |
| **Simpson's paradox** | Aggregate and subgroup conclusions disagree |

!!! danger "The heuristic worth internalising"
    **Results that are surprisingly good are a bug report, not a breakthrough.**

    Near-perfect accuracy on a genuinely hard problem almost always means leakage, duplicate rows
    across splits, or a feature that accidentally encodes the target. Go and find it *before* you
    present it — finding it yourself is much better than having someone else find it afterwards.

---

## 8. Recap

- **Median and IQR are robust; mean and standard deviation are not.** Report **percentiles** for
  latency.
- Mean ≫ median → **right skew**. Power-law data (word frequencies, Zipf's law) makes the mean
  nearly meaningless.
- **Pearson = linear, Spearman = monotonic/rank.** `r = 0` means no *linear* relationship, **not**
  independence — always plot the scatter.
- **Correlation is not causation**; only randomised experiments establish it. Watch for
  **Simpson's paradox** and segment before concluding.
- **MSE punishes large errors quadratically; MAE is robust to outliers.** Which you want is a
  design decision about consequences.
- **R² = proportion of explained variance.** 0 means no better than the mean; negative means
  worse. Use **adjusted R²** when comparing models with different feature counts.
- Compare on identical splits, against a baseline, **with variance reported** — a gap smaller than
  the standard deviation is noise.
- Memorise the invalidation checklist, and treat surprisingly good results as leakage until
  proven otherwise.
