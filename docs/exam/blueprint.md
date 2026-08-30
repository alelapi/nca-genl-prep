# Official Blueprint

The five domains and their task statements, as published in NVIDIA's
[NCA-GENL Exam Study Guide (PDF)](https://dam-cdn.nvd.orangelogic.com/AssetLink/ik3amm6oy7871nv371600i4h7drvex5m.pdf).
Each statement is annotated with the page in this course that covers it.

---

## 1. Core Machine Learning and AI Knowledge — 30%

> *Knowledge of algorithms, conventions, and techniques that allow computers to learn
> from and make predictions or decisions based on data.*

| # | Task statement | Covered in |
| --- | --- | --- |
| 1.1 | Assist in deployment and evaluation of model scalability, performance and reliability | [Deployment & Serving](../domain-2/deployment.md) |
| 1.2 | Awareness of extracting insights from large datasets (data mining, visualization) | [EDA & Visualization](../domain-4/eda-visualization.md) |
| 1.3 | Build LLM use cases such as RAG, chatbots and summarizers | [RAG](../domain-1/rag.md), [LLM App Architecture](../domain-2/llm-app-architecture.md) |
| 1.4 | Curate and embed content datasets for RAGs | [Embeddings](../domain-1/embeddings.md), [RAG](../domain-1/rag.md) |
| 1.5 | Fundamentals of ML (feature engineering, model comparison, cross-validation) | [ML Fundamentals](../domain-1/ml-fundamentals.md) |
| 1.6 | Capabilities of Python NLP packages (spaCy, NumPy, vector databases, …) | [Python NLP Stack](../domain-1/python-nlp-stack.md) |
| 1.7 | Read research papers to identify emerging LLM trends | [LLMs & Foundation Models](../domain-1/llm-landscape.md) |
| 1.8 | Select and use models to create text embeddings | [Embeddings](../domain-1/embeddings.md) |
| 1.9 | Use prompt engineering principles to achieve desired results | [Prompt Engineering](../domain-1/prompt-engineering.md) |
| 1.10 | Use Python packages (spaCy, NumPy, Keras, …) for traditional ML analyses | [Python NLP Stack](../domain-1/python-nlp-stack.md) |

---

## 2. Software Development — 24%

> *Create, maintain, and test software.*

| # | Task statement | Covered in |
| --- | --- | --- |
| 4.1 | Assist in deployment and evaluation of model scalability, performance and reliability | [Deployment & Serving](../domain-2/deployment.md) |
| 4.2 | Build LLM use cases such as RAGs, chatbots and summarizers | [LLM App Architecture](../domain-2/llm-app-architecture.md) |
| 4.3 | Capabilities of Python natural language packages | [Python NLP Stack](../domain-1/python-nlp-stack.md) |
| 4.4 | Identify system data, hardware or software components required to meet user needs | [Hardware & Memory Sizing](../domain-2/hardware-sizing.md) |
| 4.5 | Monitor functioning of data collection, experiments and other software processes | [Deployment & Serving](../domain-2/deployment.md) |
| 4.6 | Use Python packages to implement traditional ML analyses | [Python NLP Stack](../domain-1/python-nlp-stack.md) |
| 4.7 | Write software components or scripts under supervision | [Labs](../labs/index.md) |

The suggested readings for this domain reveal the real emphasis: **TensorRT**,
**NeMo best practices**, **LLM customization**, **INT8 quantization / QAT**, **NCCL and
AllReduce**, **distributed training**, **scaling laws** and **BERT pretraining**. Those
are covered in [Inference Optimization](../domain-2/inference-optimization.md),
[Customization & PEFT](../domain-2/customization.md) and
[Distributed Training](../domain-2/distributed-training.md).

---

## 3. Experimentation — 22%

> *The study of how to perform, evaluate, and interpret experiments, including AI model
> evaluation and the use of human subjects in labeling or reinforcement learning from
> human feedback (RLHF).*

!!! warning "A quirk in the official PDF"
    In the published study guide, the five task statements printed under
    *Experimentation* (3.1–3.5) are **identical to the ones under Data Analysis**
    (2.1–2.5) — apparently a copy-and-paste slip. The domain's real scope is
    unambiguous from its **description**, its **course objectives** and especially its
    **suggested reading list**. Prepare for the topics below, not for a repeat of
    Data Analysis.

| Derived from | Topic | Covered in |
| --- | --- | --- |
| *"How to Conduct A/B Testing in ML"* | A/B testing, experiment design | [Experiment Design](../domain-3/experiment-design.md) |
| *"Cross-Validation in Machine Learning"* | Validation strategy, data splits | [Experiment Design](../domain-3/experiment-design.md) |
| *"Zero-Shot Testing"* | Zero-/one-/few-shot evaluation | [Zero- & Few-Shot](../domain-3/zero-few-shot.md) |
| *"GLUE"*, *"Benchmarking Elementary Language Tasks"* | Benchmarks and leaderboards | [Evaluation Metrics](../domain-3/evaluation-metrics.md) |
| *"Machine Translation methods"*, *"Speech and Language Processing"* | Task metrics (BLEU, ROUGE, perplexity) | [Evaluation Metrics](../domain-3/evaluation-metrics.md) |
| *"Evaluating RAG Applications"* | RAG-specific evaluation | [RAG Evaluation](../domain-3/rag-evaluation.md) |
| *"Hallucinations in Large Language Models"* | Failure modes and mitigation | [Hallucinations](../domain-3/hallucinations.md) |
| *"Inference Optimization"* | Latency/throughput experiments | [Inference Optimization](../domain-2/inference-optimization.md) |
| Domain description | RLHF and human labeling | [Alignment & RLHF](../domain-3/rlhf-alignment.md) |

---

## 4. Data Analysis — 14%

> *Inspecting, cleansing, transforming, and modeling data with the goal of discovering
> useful information, informing conclusions, and supporting decision-making.*

| # | Task statement | Covered in |
| --- | --- | --- |
| 2.1 | Awareness of extracting insights from large datasets (data mining, visualization) | [EDA & Visualization](../domain-4/eda-visualization.md) |
| 2.2 | Compare models using statistical performance metrics (loss functions, explained variance) | [Statistics & Metrics](../domain-4/statistics.md) |
| 2.3 | Conduct data analysis under supervision | [Data Preprocessing](../domain-4/data-preprocessing.md) |
| 2.4 | Create graphs, charts or other visualizations using specialized software | [EDA & Visualization](../domain-4/eda-visualization.md) |
| 2.5 | Identify relationships, trends or factors affecting results | [Statistics & Metrics](../domain-4/statistics.md) |

Its suggested readings add the **RAPIDS** stack (cuDF, cuML, cuGraph), **data
exploration**, and **stemming/lemmatization with scikit-learn vectorizers** —
see [Accelerated Data Science](../domain-4/rapids.md) and
[Data Preprocessing](../domain-4/data-preprocessing.md).

---

## 5. Trustworthy AI — 10%

> *Creation and assessment of ethical, energy-conscious, and reliable AI systems…
> designed and applied in a manner that's transparent, fair, and verifiable.*

| # | Task statement | Covered in |
| --- | --- | --- |
| 5.1 | Describe the ethical principles of trustworthy AI | [Principles](../domain-5/principles.md) |
| 5.2 | Describe the balance between data privacy and data consent | [Privacy & Consent](../domain-5/privacy-consent.md) |
| 5.3 | Describe how to use NVIDIA and other technologies to improve AI trustworthiness | [Guardrails & Security](../domain-5/guardrails-security.md) |
| 5.4 | Describe how to minimize bias in AI systems | [Bias & Fairness](../domain-5/bias-fairness.md) |

---

## The job description behind the exam

NVIDIA frames the certification around a concrete role. Questions are written from
this perspective, so it is worth reading once:

- Collaborate with an AI team to **design, code, test, debug and document** applications.
- **Integrate new AI language models** into existing systems.
- Assist in **assessing and resolving performance issues**.
- **Perform prompt engineering** and **select models**.
- **Define, curate, label and annotate LLM datasets**.
- Perform **experimentation**: A/B testing, evaluating prompts, evaluating models,
  producing proofs of concept.

Recommended background: a CS/SE/AI degree, knowledge of **Python, C and AI frameworks
(PyTorch, TensorFlow)**, and a solid understanding of **neural networks and deep
learning models**.
