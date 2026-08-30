# Accelerated Data Science (RAPIDS)

*NVIDIA's suggested readings for Data Analysis are dominated by RAPIDS: the RAPIDS site, the cuML
documentation, and "GPU Accelerated Data Science With RAPIDS". Its course objectives name
**cuDF**, **Polars**, **Dask**, **NetworkX**, **cuGraph**, **XGBoost** and **Triton**.*

This is a short, high-yield page. If you learn nothing else in Domain 4, learn the component
mapping in section 2.

---

## 1. What RAPIDS is, and the problem it solves

The standard Python data science stack — pandas, scikit-learn, NetworkX — is CPU-bound and
largely single-threaded. On a 50 GB dataset, a groupby that should take seconds takes minutes,
and an interactive analysis session becomes a series of coffee breaks.

Meanwhile the GPU sitting in the same machine is idle. It has thousands of cores and enormous
memory bandwidth, and dataframe operations — filters, joins, aggregations, sorts — are *exactly*
the kind of massively parallel work it is good at.

**RAPIDS** is NVIDIA's open-source suite that runs that stack **on GPUs**, keeping data resident
in GPU memory across the whole pipeline and — crucially — **mirroring the APIs people already
know**.

That last point is the actual design achievement. A GPU dataframe library with a novel API would
be a research project nobody adopts. A GPU dataframe library that accepts your existing pandas
code is a drop-in upgrade.

---

## 2. The component mapping — memorise this

```text
 CPU stack                GPU equivalent (RAPIDS)
 ─────────────            ───────────────────────
 pandas          ──────►  cuDF
 scikit-learn    ──────►  cuML
 NetworkX        ──────►  cuGraph
 NumPy           ──────►  CuPy
 (vector search) ──────►  cuVS / RAFT
 Dask            ──────►  Dask-cuDF        (multi-GPU / multi-node)
 Spark           ──────►  RAPIDS Accelerator for Apache Spark
 GeoPandas       ──────►  cuSpatial
 dashboards      ──────►  cuxfilter
```

| Library | Replaces | What it does |
| --- | --- | --- |
| **cuDF** | pandas | GPU DataFrames: load, filter, join, groupby, aggregate, sort |
| **cuML** | scikit-learn | GPU machine learning: regression, random forest, k-means, DBSCAN, PCA, t-SNE, UMAP, k-NN |
| **cuGraph** | NetworkX | GPU graph analytics: PageRank, BFS, centrality, connected components, community detection |
| **cuVS / RAFT** | — | GPU vector search primitives — used by FAISS-GPU and Milvus |
| **CuPy** | NumPy | GPU N-dimensional arrays |
| **cuSpatial** | GeoPandas | Geospatial operations |
| **cuxfilter** | — | GPU-backed interactive dashboards |
| **Dask-cuDF** | Dask | Scale cuDF across multiple GPUs and nodes |

!!! tip "The exam framing"
    *"Accelerate an existing pandas / scikit-learn / NetworkX workflow on GPUs with minimal code
    changes"* → **cuDF / cuML / cuGraph** respectively.

    The distractors are always the other RAPIDS libraries, so the mapping is the whole question.

---

## 3. Zero-code-change acceleration

The most notable recent development, and worth knowing because it changes the adoption story
entirely.

```python
# pandas, accelerated — the script itself is unchanged
%load_ext cudf.pandas       # in Jupyter
import pandas as pd         # transparently GPU-backed from here

# or from the command line:
# python -m cudf.pandas my_existing_script.py
```

```bash
# NetworkX, accelerated
NETWORKX_BACKEND_PRIORITY=cugraph python my_graph_script.py
```

The important property: **`cudf.pandas` falls back to CPU pandas automatically** for any
operation it does not support. Your script keeps working — it just runs the supported parts on
the GPU. That removes the traditional adoption risk, where migrating meant rewriting and hoping
everything was covered.

**Polars** also has a GPU engine powered by cuDF, and **Spark** has the RAPIDS Accelerator — both
named in NVIDIA's course objectives.

---

## 4. When RAPIDS helps — and when it does not

This nuance is examined, because the naive answer ("GPUs are faster") is wrong often enough to
make a good question.

**It helps when:**

- The dataset is **large** — roughly hundreds of MB up to many GB — and fits in GPU memory (or is
  partitioned across GPUs with Dask-cuDF).
- Operations are **vectorised and parallel**: joins, groupbys, aggregations, sorts, distance
  computations, matrix operations.
- The pipeline can stay **entirely on the GPU** from load to model.

**It does not help when:**

- The dataset is **small.** Kernel-launch overhead and the cost of moving data to the GPU dominate,
  and CPU pandas wins outright. For a 30 MB CSV, RAPIDS is slower.
- The work is **inherently sequential** or dominated by Python-level loops with custom logic that
  cannot be vectorised.
- Data must move **back and forth across the PCIe bus** repeatedly.
- The data **does not fit in GPU memory** and cannot be partitioned.

!!! important "Host-to-device transfer is the classic performance killer"
    ```text
    ❌ SLOW                              ✓ FAST
    load on CPU                          load directly into cuDF
    → transfer to GPU                    → filter on GPU
    → filter                             → join on GPU
    → transfer back to CPU               → aggregate on GPU
    → join                               → train with cuML
    → transfer to GPU again              → only the final result leaves
    → aggregate
    ```

    Each PCIe round trip can cost more time than the computation it enables. **The value of
    RAPIDS is not any single library — it is keeping the data resident on the GPU across the
    entire pipeline.** A pipeline that bounces between CPU and GPU can easily be slower than pure
    pandas.

---

## 5. XGBoost

Named directly in NVIDIA's course objective: *"utilize a wide variety of machine learning
algorithms, including XGBoost, for different data science problems."*

**What it is:** gradient-boosted decision trees. An ensemble built **sequentially**, where each
new tree is trained to correct the errors of the accumulated ensemble so far.

**Why it still matters in an LLM-era exam:** it remains the **strongest default for tabular
data**, routinely beating neural networks on structured, feature-based problems. Deep learning
won decisively on text, images and audio; on spreadsheets, gradient-boosted trees are still the
answer.

It has native GPU acceleration (`device="cuda"`) and integrates with RAPIDS and Dask.

Hyperparameters to recognise: `n_estimators` (number of trees), `max_depth`, `learning_rate`
(`eta`), `subsample`, `colsample_bytree`, `reg_lambda`.

!!! tip "Bagging vs. boosting — a likely question"
    | | **Random forest = BAGGING** | **XGBoost = BOOSTING** |
    | --- | --- | --- |
    | Trees are trained | **In parallel**, independently, on bootstrap samples | **Sequentially**, each correcting the last |
    | Combination | Averaged / majority vote | Weighted sum |
    | Primarily reduces | **Variance** | **Bias** |
    | Overfitting risk | Low — averaging is stabilising | Higher — needs regularization and early stopping |
    | Parallelisable | Trivially | Only within each tree |

    The mnemonic: **bagging averages many independent opinions; boosting is a sequence of
    corrections.**

---

## 6. Graph analytics with cuGraph

Named in two separate NVIDIA course objectives (*"learn and apply powerful graph algorithms to
analyze complex networks with NetworkX and cuGraph"*).

| Algorithm | The question it answers |
| --- | --- |
| **PageRank** | Which nodes are most important, judged by link structure? |
| **Betweenness / degree centrality** | Which nodes are the key bridges / most connected? |
| **BFS / SSSP** | What are the shortest paths? |
| **Connected components** | Which nodes form isolated subgroups? |
| **Louvain / Leiden** | What communities exist in this network? |
| **Jaccard / overlap similarity** | Which nodes are similar by shared neighbours? |

Why graphs matter in an LLM context: **knowledge graphs** and
[GraphRAG](../domain-1/rag.md#4-advanced-patterns), where retrieval traverses relationships rather
than matching flat chunks — which is what makes multi-hop questions answerable. Plus the classical
applications: fraud rings, recommendation, citation networks, entity resolution.

---

## 7. The end-to-end GPU pipeline

NVIDIA's course *Accelerating End-to-End Data Science Workflows* pitches exactly this shape, and
it is worth recognising as a diagram:

```text
   ┌────────────┐    ┌──────────────────┐    ┌────────────────┐
   │   cuDF     │───►│  cuML / XGBoost  │───►│     Triton     │
   │ load,      │    │  train, evaluate │    │ deploy & serve │
   │ wrangle    │    │                  │    │                │
   └─────┬──────┘    └────────┬─────────┘    └────────────────┘
         │                    │
   ┌─────▼──────┐      ┌──────▼──────┐
   │  cuGraph   │      │  cuxfilter  │
   │  graph     │      │  visualise  │
   └────────────┘      └─────────────┘

        ◄──────── all of this stays in GPU memory ────────►
```

The point is the annotation at the bottom, not the boxes.

---

## 8. Recap

- **cuDF → pandas. cuML → scikit-learn. cuGraph → NetworkX. CuPy → NumPy. cuVS → vector search.**
  Memorise this mapping — it is the most likely question in Domain 4.
- **`cudf.pandas`** and the NetworkX cuGraph backend give **zero-code-change** GPU acceleration
  with automatic CPU fallback.
- RAPIDS pays off on **large, vectorisable** workloads. On small data, or with constant
  host↔device transfers, it is **slower** than CPU pandas.
- **Keep the whole pipeline on the GPU** — PCIe round trips can cost more than the computation.
- **XGBoost = boosting** (sequential, reduces bias, still the tabular default);
  **random forest = bagging** (parallel, reduces variance).
- **cuGraph** for PageRank, centrality, shortest paths and community detection — and for
  GraphRAG.
