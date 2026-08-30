# EDA & Visualization

*Covers tasks 2.1 ("awareness of the process of extracting insights from large datasets using
data mining, data visualization and similar techniques") and 2.4 ("create graphs, charts or other
visualizations to convey the results of data analysis using specialized software").*

---

## 1. What EDA is for

**Exploratory data analysis** is the phase where you build an understanding of the data before
modeling it.

Its output is **not charts**. Its output is **decisions**: which features matter, what needs
cleaning, whether your assumptions hold, what the model will struggle with, and whether the
question you are trying to answer is even answerable with this data.

Skipping it is how projects end up training excellent models on subtly broken data.

### A working checklist

1. **Shape and structure** — rows, columns, dtypes, memory footprint.
2. **Univariate distributions** — centre, spread, skew, modality, for each variable.
3. **Missingness** — how much, and is it random or patterned?
4. **Outliers** — present, and are they errors or genuine extremes?
5. **Bivariate relationships** — each feature against the target, and features against each other.
6. **Correlation and multicollinearity.**
7. **Class balance** for classification; **temporal trends** for time series.
8. **Data quality** — duplicates, inconsistent categories, impossible values.

```python
df.shape
df.info()
df.describe(include="all")
df.isna().mean().sort_values(ascending=False)     # missingness per column
df["target"].value_counts(normalize=True)         # class balance
df.corr(numeric_only=True)                        # correlation matrix
```

That `value_counts(normalize=True)` line is the one that most often changes the plan: discovering
the positive class is 0.4% of the data means [accuracy is
useless](../domain-3/evaluation-metrics.md#the-accuracy-trap) and you need a different metric,
different sampling, and probably different expectations.

---

## 2. Choosing the right chart (task 2.4)

The most examinable part of this page: **which visualization answers which question.**

| The question you are asking | The chart |
| --- | --- |
| How is one numeric variable distributed? | **Histogram**, density plot |
| What is the spread, and are there outliers — especially across groups? | **Box plot**, violin plot |
| How do two numeric variables relate? | **Scatter plot** |
| How do categories compare? | **Bar chart** |
| How does something change over time? | **Line chart** |
| How do many variables correlate? | **Heatmap** of the correlation matrix |
| What are all the pairwise relationships in a small feature set? | **Pair plot** / scatter matrix |
| What are the parts of a whole? | Stacked bar (usually better than a pie chart) |
| How do two categorical variables relate? | Grouped bar, mosaic plot, cross-tab heatmap |
| Where is the density when points overlap heavily? | Hexbin, 2-D density plot |
| What structure exists in high-dimensional data (e.g. embeddings)? | **t-SNE / UMAP** projection |
| Where does my classifier make mistakes? | **Confusion matrix** heatmap |
| How does my classifier behave across thresholds? | **ROC curve**, precision–recall curve |
| Is my model overfitting? | **Learning curves** — training vs. validation loss per epoch |

!!! tip "The three to know instantly"
    **Histogram** → the distribution of one variable.
    **Scatter plot** → the relationship between two.
    **Box plot** → spread, median, quartiles and outliers, especially compared across groups.

    These three cover the majority of chart-selection questions.

### Reading a box plot

Worth being explicit about, since it encodes five numbers at once:

```text
        outliers ●   ●
                 │
            ┌────┴────┐
    ├───────┤    │    ├───────┤
            └────┬────┘
   whisker   Q1  │   Q3   whisker
                median

  box     = IQR (middle 50% of the data)
  whiskers= typically 1.5 × IQR beyond the box
  points  = outliers beyond that
```

Its strength is comparison: five box plots side by side show you instantly which group has the
highest median, which is most variable, and which has outliers.

---

## 3. Visualizing embeddings

Directly relevant to LLM work: embeddings live in 384–4096 dimensions and cannot be plotted
directly.

| Method | Character | Safe as model input? |
| --- | --- | --- |
| **PCA** | Linear, deterministic, fast. Preserves global variance structure | **Yes** |
| **t-SNE** | Non-linear, stochastic. Excellent at revealing local clusters. Slow. Layout depends on the `perplexity` parameter | **No** |
| **UMAP** | Non-linear, faster than t-SNE, preserves more global structure | **No** |

!!! danger "The t-SNE trap — a reliable exam question"
    **t-SNE and UMAP are for visualization only.** Three specific consequences:

    1. **The axes are meaningless.** They are not "dimension 1" and "dimension 2" of anything.
    2. **Distances between clusters are not interpretable.** Two clusters appearing far apart in a
       t-SNE plot may be adjacent in the original space. The algorithm optimises local
       neighbourhoods, not global geometry.
    3. **It is not deterministic.** Different runs, and different `perplexity` values, produce
       visibly different layouts of the same data.

    So: never feed t-SNE output into a model, and never claim two groups are "far apart" because
    they look far apart. If you need dimensionality reduction **as a preprocessing step**, use
    **PCA**.

---

## 4. The plotting stack (task 2.4: "specialized software")

| Tool | Identity |
| --- | --- |
| **Matplotlib** | The foundational Python plotting library. Total control, verbose API |
| **Seaborn** | Statistical charts on top of Matplotlib; understands DataFrames; good defaults |
| **Plotly** | Interactive, web-based — zoom, hover, dashboards |
| **Bokeh / Altair** | Other interactive / declarative options |
| **pandas `.plot()`** | Quick Matplotlib charts straight off a DataFrame |
| **Tableau / Power BI / Looker** | BI tools for business dashboards |
| **TensorBoard / Weights & Biases** | Training curves, experiment tracking, model comparison |
| **cuxfilter** | RAPIDS' GPU-accelerated dashboarding for large datasets |

```python
import seaborn as sns, matplotlib.pyplot as plt

sns.histplot(df["latency_ms"], bins=50, kde=True)                  # distribution
sns.boxplot(data=df, x="model", y="score")                         # compare groups
sns.scatterplot(data=df, x="tokens_in", y="latency_ms", hue="model")  # relationship
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")  # correlations
plt.show()
```

Note the relationship: **Seaborn is a layer on Matplotlib**, not a competitor. You use Seaborn
for the chart and Matplotlib to adjust it.

---

## 5. Principles of an honest chart

Trustworthy AI applies to how you present results, not only to how you build models. A misleading
chart is a misleading claim.

- **Label the axes and state the units.** Always. An unlabeled axis is an unsupported assertion.
- **Start bar-chart y-axes at zero.** Truncating the axis visually exaggerates small differences —
  a 2% difference can be made to look like a doubling. (Line charts showing change over time may
  legitimately not start at zero, because the reader is looking at the slope.)
- **Show variability.** Error bars, confidence intervals, or the individual points. A bare mean
  hides exactly the information needed to judge whether a difference is real — the same argument
  as [reporting mean ± std](statistics.md#6-comparing-models-rigorously).
- **One message per chart.** If it takes a paragraph to explain what to look at, it is two charts.
- **Choose colour deliberately** — sequential for magnitude, diverging for a meaningful midpoint,
  categorical for classes. Use colourblind-safe palettes, and **never encode meaning in colour
  alone**.
- **Avoid 3-D effects and many-slice pie charts.** Both make quantities genuinely harder to
  compare.
- **State the sample size.** "Group B performs better" means something very different at n=8 than
  at n=8,000.

---

## 6. Data mining techniques (task 2.1)

"Extracting insights from large datasets" — the named families:

| Technique | What it finds |
| --- | --- |
| **Clustering** | Natural groupings (k-means, DBSCAN, hierarchical) |
| **Classification** | A supervised label |
| **Regression** | A continuous relationship |
| **Association rule mining** | Co-occurrence patterns — "customers who bought X also bought Y" |
| **Anomaly detection** | Rare, unusual records |
| **Dimensionality reduction** | Compressed representations (PCA) |
| **Topic modeling** | Latent themes in a text corpus (LDA) |
| **Sequence / pattern mining** | Ordered patterns over time |

!!! tip "The modern answer for text corpora"
    Classical topic modeling (LDA) has largely been superseded for the question *"what is in this
    corpus?"* by a simpler pipeline:

    ```text
    embed every document ──► cluster the embeddings ──► label each cluster with an LLM
    ```

    Faster, usually higher quality, and the cluster labels are human-readable sentences rather
    than bags of keywords. It uses exactly the machinery from
    [Embeddings](../domain-1/embeddings.md).

---

## 7. Recap

- EDA produces **decisions, not decoration**: distributions, missingness, outliers, relationships,
  class balance, data quality.
- **Histogram** = distribution of one variable. **Scatter** = relationship between two.
  **Box plot** = spread, median and outliers across groups. **Heatmap** = correlation matrix.
- **t-SNE and UMAP are visualization only** — meaningless axes, uninterpretable cluster distances,
  non-deterministic. Use **PCA** if you need reduction as model input.
- **Matplotlib** (foundation) → **Seaborn** (statistical layer) → **Plotly** (interactive);
  **TensorBoard/W&B** for training curves; **cuxfilter** for GPU dashboards.
- Honest charts: labelled axes, zero-based bars, **visible variability**, stated sample size,
  colourblind-safe palettes.
- For "what is in this text corpus?", embed → cluster → label with an LLM.
