# Accelerated Data Science (RAPIDS)

*NVIDIA's suggested readings for Data Analysis are dominated by RAPIDS: the RAPIDS site,
the cuML documentation, and "GPU Accelerated Data Science With RAPIDS". Its course
objectives name **cuDF**, **Polars**, **Dask**, **NetworkX**, **cuGraph**, **XGBoost** and
**Triton**. This is a short, high-yield page — learn the component names.*

## What RAPIDS is

**RAPIDS** is NVIDIA's open-source suite of libraries that runs the standard Python data
science stack **on GPUs**, keeping data in GPU memory across the whole pipeline and
deliberately mirroring the APIs people already know.

```text
 CPU stack            GPU equivalent (RAPIDS)
 ─────────────        ───────────────────────
 pandas         ───►  cuDF
 scikit-learn   ───►  cuML
 NetworkX       ───►  cuGraph
 (vector search)───►  cuVS / RAFT
 NumPy          ───►  CuPy
 Dask           ───►  Dask-cuDF (multi-GPU / multi-node)
 Spark          ───►  RAPIDS Accelerator for Apache Spark
 signal/image   ───►  cuSignal / cuCIM
 dashboards     ───►  cuxfilter
```

The design goal is **API compatibility**: code written against pandas or scikit-learn
often runs on GPU with an import change or none at all.

## The components

| Library | Replaces | Purpose |
| --- | --- | --- |
| **cuDF** | pandas | GPU DataFrames: load, filter, join, groupby, aggregate |
| **cuML** | scikit-learn | GPU machine learning: regression, random forest, k-means, DBSCAN, PCA, t-SNE, UMAP, k-NN |
| **cuGraph** | NetworkX | GPU graph analytics: PageRank, BFS, connected components, centrality, community detection |
| **cuVS / RAFT** | — | GPU-accelerated vector search and nearest-neighbour primitives (used by FAISS-GPU and Milvus) |
| **CuPy** | NumPy | GPU N-dimensional arrays |
| **cuSpatial** | GeoPandas | Geospatial operations |
| **cuxfilter** | — | GPU-backed interactive dashboards |
| **Dask-cuDF** | Dask | Scale cuDF across multiple GPUs and nodes |

## Zero-code-change acceleration

The most exam-worthy recent development: RAPIDS can accelerate existing code **without
rewriting it**.

```python
# pandas, accelerated — no code changes to the script itself
%load_ext cudf.pandas          # in Jupyter
import pandas as pd            # transparently GPU-backed

# or from the command line:
# python -m cudf.pandas my_script.py
```

```python
# NetworkX, accelerated
# NETWORKX_BACKEND_PRIORITY=cugraph python my_graph_script.py
```

`cudf.pandas` falls back to CPU pandas automatically for any operation it does not
support, so scripts keep working. **Polars** also has a GPU engine powered by cuDF, and
**Spark** has the RAPIDS Accelerator — both named in NVIDIA's course objectives.

## When RAPIDS helps — and when it does not

**It helps when:**

- The dataset is large (roughly hundreds of MB up to many GB) and fits in GPU memory
  (or is spread across GPUs with Dask-cuDF).
- Operations are vectorised and parallel: joins, groupbys, aggregations, sorts,
  distance computations, matrix operations.
- The pipeline can stay **entirely on the GPU** end to end.

**It does not help when:**

- The dataset is small — kernel-launch and transfer overhead dominate, and pandas wins.
- The work is inherently sequential or Python-loop-heavy with custom logic.
- Data must be moved back and forth across the PCIe bus repeatedly. **Host↔device transfer
  is the classic performance killer**; keep data resident on the GPU.
- The data does not fit in GPU memory and cannot be partitioned.

!!! tip "The exam framing"
    "Accelerate an existing pandas/scikit-learn/NetworkX data science workflow on GPUs
    with minimal code changes" → **RAPIDS** (cuDF / cuML / cuGraph respectively).

## XGBoost

Named directly in NVIDIA's course objective: *"utilize a wide variety of machine learning
algorithms, including XGBoost, for different data science problems."*

- **Gradient-boosted decision trees** — an ensemble built sequentially, where each new tree
  corrects the errors of the accumulated ensemble.
- Still the **strongest default for tabular data**, routinely beating neural networks there.
- Has native **GPU acceleration** (`device="cuda"`) and integrates with RAPIDS and Dask.
- Key hyperparameters to recognise: `n_estimators`, `max_depth`, `learning_rate`
  (`eta`), `subsample`, `colsample_bytree`, `reg_lambda`.

!!! note "Bagging vs. boosting"
    **Random forest = bagging** — many independent trees trained in parallel on bootstrap
    samples, averaged. Reduces **variance**.
    **XGBoost = boosting** — trees trained sequentially, each fixing the previous errors.
    Reduces **bias**, and is more prone to overfitting without regularisation.

## Graph analytics with cuGraph

Named in two separate NVIDIA course objectives (*"learn and apply powerful graph algorithms
to analyze complex networks with NetworkX and cuGraph"*).

| Algorithm | Answers |
| --- | --- |
| **PageRank** | Which nodes are most important, by link structure |
| **Betweenness / degree centrality** | Which nodes are the most connected or the key bridges |
| **BFS / SSSP** | Shortest paths |
| **Connected components** | Which nodes form isolated subgroups |
| **Louvain / Leiden** | Community detection |
| **Jaccard / Overlap similarity** | Node similarity by shared neighbours |

Applications relevant to LLM work: knowledge graphs (and **GraphRAG**), fraud rings,
recommendation, citation networks, entity resolution.

## The end-to-end GPU pipeline

NVIDIA's course *Accelerating End-to-End Data Science Workflows* pitches exactly this:

```text
cuDF (load & wrangle) ──► cuML / XGBoost (train) ──► Triton (deploy & serve)
        │                        │
   cuGraph (graph)          cuxfilter (visualise)
```

The value is not any single library — it is **never leaving GPU memory** between stages.
Each round trip to host memory costs more than the computation it enables.

## Key takeaways

- **cuDF → pandas. cuML → scikit-learn. cuGraph → NetworkX. CuPy → NumPy. cuVS → vector
  search.** Memorise this mapping.
- `cudf.pandas` and the NetworkX cuGraph backend give **zero-code-change** GPU
  acceleration, with automatic CPU fallback.
- RAPIDS pays off on **large, vectorisable** workloads; small data and constant
  host↔device transfers erase the benefit.
- **XGBoost** (boosting, sequential) is the tabular-data default; random forest is bagging
  (parallel).
- Keep the entire pipeline on the GPU — cuDF → cuML → Triton.
