# EDA & Visualization

*Covers tasks 2.1 (extracting insights from large datasets using data mining and
visualization) and 2.4 (create graphs, charts or other visualizations using specialized
software).*

## What exploratory data analysis is for

**EDA** is the phase where you build an understanding of the data before modeling it. Its
outputs are not charts — they are decisions: which features matter, what needs cleaning,
which assumptions hold, and what the model is likely to struggle with.

A working checklist:

1. **Shape and structure** — rows, columns, dtypes, memory.
2. **Univariate distributions** — centre, spread, skew, modality, for each variable.
3. **Missingness** — how much, and is it random or patterned?
4. **Outliers** — present, and are they errors or real extremes?
5. **Bivariate relationships** — feature ↔ target, feature ↔ feature.
6. **Correlation and multicollinearity.**
7. **Class balance** for classification, and **temporal trends** for time series.
8. **Data quality** — duplicates, inconsistent categories, impossible values.

```python
df.shape; df.info(); df.describe(include="all")
df.isna().mean().sort_values(ascending=False)
df["target"].value_counts(normalize=True)
df.corr(numeric_only=True)
```

## Choosing the right chart (task 2.4)

The most examinable part of this page: **which visualization for which question**.

| Question | Chart |
| --- | --- |
| Distribution of one numeric variable | **Histogram**, density plot |
| Distribution + outliers, comparison across groups | **Box plot**, violin plot |
| Relationship between two numeric variables | **Scatter plot** |
| Comparison across categories | **Bar chart** |
| Change over time / trend | **Line chart** |
| Correlation across many variables | **Heatmap** of the correlation matrix |
| Pairwise relationships in a small feature set | **Pair plot** / scatter matrix |
| Part-to-whole | Stacked bar (usually better than a pie chart) |
| Two categorical variables | Grouped bar, mosaic plot, cross-tab heatmap |
| Density with many overlapping points | Hexbin, 2-D density |
| High-dimensional structure (e.g. embeddings) | **t-SNE / UMAP** projection to 2-D |
| Classifier errors | **Confusion matrix** heatmap |
| Classifier threshold behaviour | **ROC curve**, precision–recall curve |
| Training progress | **Learning curves** (train vs. validation loss per epoch) |

!!! tip "The three you should be able to name instantly"
    **Histogram** → distribution of one variable. **Scatter plot** → relationship between
    two. **Box plot** → spread, median, quartiles and outliers, especially across groups.

## Visualizing embeddings

Directly relevant to LLM work: embeddings live in 384–4096 dimensions and cannot be
plotted directly.

- **PCA** — linear, fast, preserves global variance structure. Also usable as a
  preprocessing step.
- **t-SNE** — non-linear, excellent at revealing local clusters. Slow; the layout depends
  on **perplexity**; **distances between clusters are not meaningful**.
- **UMAP** — non-linear, faster than t-SNE, preserves more global structure.

!!! danger "The trap"
    **t-SNE and UMAP are for visualization only.** Their axes have no meaning, distances
    between clusters are not interpretable, and different runs give different layouts.
    Never feed their output into a model, and never claim two clusters are "far apart"
    because they look far apart.

## The plotting stack (task 2.4: "specialized software")

| Tool | Identity |
| --- | --- |
| **Matplotlib** | The foundational Python plotting library. Total control, verbose API |
| **Seaborn** | Statistical charts on top of Matplotlib; understands DataFrames; good defaults |
| **Plotly** | Interactive, web-based; zoom, hover, dashboards |
| **Bokeh / Altair** | Other interactive/declarative options |
| **pandas `.plot()`** | Quick Matplotlib charts straight off a DataFrame |
| **Tableau / Power BI / Looker** | BI tools for business dashboards |
| **TensorBoard / Weights & Biases** | Training curves, experiment tracking, model comparison |
| **cuxfilter** | RAPIDS' GPU-accelerated dashboarding for large datasets |

```python
import seaborn as sns, matplotlib.pyplot as plt

sns.histplot(df["latency_ms"], bins=50, kde=True)
sns.boxplot(data=df, x="model", y="score")
sns.scatterplot(data=df, x="tokens_in", y="latency_ms", hue="model")
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()
```

## Principles of an honest chart

Trustworthy AI applies to how you present results too:

- **Label the axes and state the units.** Always.
- **Start bar-chart y-axes at zero** — truncating exaggerates differences. Line charts
  showing change over time may legitimately not start at zero.
- **Show variability** — error bars, confidence intervals, or the individual points. A
  bare mean hides everything that matters.
- **One message per chart.**
- **Choose colour deliberately** — sequential for magnitude, diverging for a meaningful
  midpoint, categorical for classes. Use colourblind-safe palettes; never encode meaning
  in colour alone.
- **Do not use 3-D effects or pie charts with many slices.** They mislead.
- **State the sample size.**

## Data mining techniques (task 2.1)

"Extracting insights from large datasets" — the named techniques:

| Technique | Finds |
| --- | --- |
| **Clustering** | Natural groupings (k-means, DBSCAN, hierarchical) |
| **Classification** | A supervised label |
| **Regression** | A continuous relationship |
| **Association rule mining** | Co-occurrence patterns ("customers who bought X also bought Y") |
| **Anomaly detection** | Rare, unusual records |
| **Dimensionality reduction** | Compressed representations (PCA) |
| **Topic modeling** | Latent themes in a text corpus (LDA, or clustering embeddings) |
| **Sequence/pattern mining** | Ordered patterns over time |

For text corpora specifically, the modern approach to "what is in this data?" is to
**embed the documents, cluster the embeddings, and label each cluster with an LLM** —
faster and usually better than classical topic modeling.

## Key takeaways

- EDA produces decisions, not decoration: distributions, missingness, outliers,
  relationships, balance.
- Histogram = distribution; scatter = relationship; box plot = spread and outliers;
  heatmap = correlation matrix.
- **t-SNE/UMAP for visualising embeddings only** — never as model input, and cluster
  distances are meaningless.
- Matplotlib (foundation) → Seaborn (statistical) → Plotly (interactive); TensorBoard/W&B
  for training curves.
- Honest charts: labelled axes, zero-based bars, visible variability, stated sample size.
