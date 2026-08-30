---
title: Data Analysis
---

# Domain 4 — Data Analysis <span class="weight">14%</span>

> *Inspecting, cleansing, transforming, and modeling data with the goal of discovering
> useful information, informing conclusions, and supporting decision-making.*

Roughly **7–8 questions**. Smaller than the first three domains, but easy points: the
material is concrete and the NVIDIA-specific part (**RAPIDS**) is a short, closed list of
facts.

## Task statements

| # | Statement |
| --- | --- |
| 2.1 | Awareness of extracting insights from large datasets using data mining, data visualization and similar techniques |
| 2.2 | Compare models using statistical performance metrics, such as loss functions or proportion of explained variance |
| 2.3 | Conduct data analysis under the supervision of a senior team member |
| 2.4 | Create graphs, charts or other visualizations to convey the results of data analysis using specialized software |
| 2.5 | Identify relationships and trends, or any factors that could affect the results of research |

## Chapters

<div class="grid cards" markdown>

- **[Data Preprocessing & Text Cleaning](data-preprocessing.md)**

    Cleaning, missing values, outliers, tokenization, **stemming vs. lemmatization**,
    vectorizers, deduplication, splitting. *(Task 2.3)*

- **[EDA & Visualization](eda-visualization.md)**

    Exploratory analysis, chart selection, distributions, correlation, the plotting
    stack. *(Tasks 2.1, 2.4)*

- **[Statistics & Performance Metrics](statistics.md)**

    Descriptive statistics, correlation vs. causation, loss functions, R², comparing
    models. *(Tasks 2.2, 2.5)*

- **[Accelerated Data Science (RAPIDS)](rapids.md)**

    cuDF, cuML, cuGraph, cuVS — the GPU equivalents of pandas, scikit-learn and NetworkX.

- **[Quiz](quiz.md)**

    14 exam-style questions.

</div>

## What NVIDIA emphasises here

The suggested reading list for this domain is short and specific:

- **RAPIDS** and **cuML documentation** — GPU-accelerated data science
- **Data exploration**
- **Stemming and lemmatizing with sklearn vectorizers**

Plus the course objectives: *"use cuDF to accelerate pandas, Polars and Dask"*, *"apply
graph algorithms with NetworkX and cuGraph"*, and *"enhance datasets through data
augmentation to improve model accuracy."*

If you learn nothing else in this domain, learn **RAPIDS component names** and
**stemming vs. lemmatization**.
