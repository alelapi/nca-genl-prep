# Statistics & Performance Metrics

*Covers tasks 2.2 (compare models using statistical performance metrics, such as loss
functions or proportion of explained variance) and 2.5 (identify relationships and trends
or any factors that could affect the results of research).*

## Descriptive statistics

| Measure | What it tells you | Caution |
| --- | --- | --- |
| **Mean** | Average | Dragged by outliers |
| **Median** | Middle value | **Robust** — prefer it for skewed data |
| **Mode** | Most frequent | The only option for categorical data |
| **Variance / standard deviation** | Spread around the mean | Same units as the data for std |
| **IQR** (Q3 − Q1) | Spread of the middle 50% | Robust to outliers; basis of the box plot |
| **Skewness** | Asymmetry | Right/positive skew: a long tail of large values |
| **Kurtosis** | Tail heaviness | High kurtosis → more extreme values than a normal distribution |
| **Percentiles** | Position in the distribution | **p95/p99 latency** is what users actually experience |

!!! tip "Mean vs. median, in one sentence"
    If mean ≫ median, the distribution is right-skewed and a few large values are pulling
    the average — report the median. This is why **latency is always reported in
    percentiles**, never as an average.

## Distributions

- **Normal (Gaussian)** — symmetric; ~68/95/99.7% of values within 1/2/3 standard
  deviations. Many statistical tests assume it.
- **Uniform** — all values equally likely.
- **Long-tailed / power law** — extremely common in real data: word frequencies
  (**Zipf's law**), document lengths, user activity. The mean is nearly meaningless here.
- **Bimodal** — two peaks, usually a sign of two mixed subpopulations. Worth splitting.

**Central limit theorem** — the distribution of the *sample mean* approaches normal as the
sample grows, regardless of the underlying distribution. This is what makes A/B test
statistics work even on skewed metrics.

## Relationships between variables (task 2.5)

**Correlation** measures how two variables move together.

| Coefficient | Measures | Use |
| --- | --- | --- |
| **Pearson** *r* | **Linear** relationship | Continuous, roughly normal, no severe outliers |
| **Spearman** ρ | **Monotonic** relationship (on ranks) | Non-linear but monotonic, ordinal, outlier-resistant |
| **Kendall** τ | Rank concordance | Small samples, many ties |

`r` ranges −1 to +1: −1 perfect inverse, 0 no *linear* relationship, +1 perfect direct.

!!! danger "Two things r cannot tell you"
    1. **Correlation is not causation.** A third **confounding** variable may drive both;
       the direction may be reversed; or it may be coincidence.
    2. **r = 0 does not mean independence** — it means no *linear* relationship. A perfect
       U-shaped relationship has r ≈ 0. Always plot the scatter.

**Simpson's paradox** — a trend present in every subgroup can reverse when the groups are
pooled. The reason you must segment your analysis before concluding anything.

**Multicollinearity** — highly correlated predictors make coefficients unstable and
uninterpretable (though predictions may be fine). Detect with a correlation heatmap or
VIF; fix by dropping, combining or regularising.

**Establishing causation** requires a **randomised controlled experiment** — which is
exactly what an [A/B test](../domain-3/experiment-design.md) is.

## Loss functions (task 2.2)

The blueprint names loss functions explicitly as a means of **comparing models**.

| Loss | Formula | Character |
| --- | --- | --- |
| **MSE (L2)** | $\frac{1}{n}\sum(y-\hat{y})^2$ | Penalises large errors quadratically; outlier-sensitive; differentiable everywhere |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same units as the target — easier to interpret |
| **MAE (L1)** | $\frac{1}{n}\sum\lvert y-\hat{y}\rvert$ | Robust to outliers; all errors weighted equally |
| **Huber** | Quadratic near 0, linear in the tails | The compromise between MSE and MAE |
| **Cross-entropy** | $-\sum y\log\hat{y}$ | The classification and **language-modeling** loss |
| **Hinge** | $\max(0, 1-y\hat{y})$ | SVMs |
| **Contrastive / triplet** | — | Embedding and retrieval training |

!!! tip "MSE vs. MAE"
    If a few large errors are much worse than many small ones (or your outliers are real
    and important), use **MSE**. If your data has outliers you do not want to chase, use
    **MAE**. This trade-off is a likely exam question.

## R² — the proportion of explained variance

The blueprint names it in exactly those words.

$$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2} $$

- **R² = 1** — the model explains all variance (perfect fit).
- **R² = 0** — the model does no better than predicting the mean.
- **R² < 0** — the model is *worse* than predicting the mean. Possible on held-out data,
  and a red flag.

**Adjusted R²** penalises adding predictors, because plain R² can only increase when you
add features — even useless random ones. Use adjusted R² when comparing models with
different numbers of features.

## Comparing models rigorously

1. **Same splits, same preprocessing, same metric** for every candidate.
2. **A baseline** — mean/majority predictor, or a simple linear model.
3. **Report variance**: mean ± std across CV folds or random seeds. A 0.2-point gap
   inside a 1.5-point standard deviation is noise.
4. **Statistical comparison** where it matters — paired t-test or Wilcoxon signed-rank
   across folds, McNemar's test for two classifiers on the same test set.
5. **More than one metric** — quality, latency, memory, cost, fairness.
6. **Error analysis** — *where* does each model fail? Two models with equal F1 can fail on
   completely different, and differently important, cases.
7. **Parsimony** — equal performance, prefer the simpler and cheaper model.

## Factors that can invalidate results (task 2.5)

A checklist worth memorising, because the task statement asks for exactly this:

- **Data leakage** — the most common cause of results too good to be true.
- **Selection bias** — the sample does not represent the population.
- **Survivorship bias** — only the successes are in the data.
- **Confounding variables** — an unmeasured cause of both variables.
- **Sampling bias / non-response bias.**
- **Label noise** and low [inter-annotator agreement](../domain-3/rlhf-alignment.md#human-annotation-quality).
- **Distribution shift** between training data and production.
- **Temporal effects** — seasonality, day-of-week, trend.
- **Small sample size** — underpowered, so a real effect goes undetected.
- **Multiple comparisons** — test twenty metrics and one will look significant by chance.
- **Benchmark contamination** — the test set was in the training data.

!!! warning "Results that are too good"
    Near-perfect accuracy on a hard problem almost always means leakage, duplicates across
    splits, or a target variable accidentally encoded in a feature — not a breakthrough.
    Check before celebrating.

## Key takeaways

- Median and IQR are robust; mean and std are not. Report **percentiles** for latency.
- Pearson = linear, Spearman = monotonic/rank-based. **r = 0 ≠ independent.**
- **Correlation is not causation**; only randomised experiments establish it.
- MSE punishes large errors, MAE is outlier-robust; cross-entropy is the classification
  and language-modeling loss.
- **R² = proportion of explained variance**; adjusted R² when comparing models with
  different feature counts.
- Compare models on identical splits with a baseline, report variance, and check for
  leakage before believing a strong result.
