# Deployment & Serving

*Covers tasks 1.1 / 4.1 (assist in deployment and evaluation of model scalability,
performance and reliability) and 4.5 (monitor functioning of data collection, experiments
and other software processes).*

## The three qualities the blueprint names

The task statement is precise — **scalability, performance and reliability** — and each
maps to concrete practice.

| Quality | Question it answers | Levers |
| --- | --- | --- |
| **Scalability** | Does it hold up as load grows? | Horizontal replicas, autoscaling, batching, load balancing, multi-GPU parallelism |
| **Performance** | Is it fast and efficient enough? | Latency (TTFT/TPOT), throughput, GPU utilisation, quantization, TensorRT |
| **Reliability** | Does it keep working? | Health checks, retries with backoff, timeouts, circuit breakers, fallback models, graceful degradation, canary rollouts |

## NVIDIA Triton Inference Server

The serving layer of the NVIDIA stack.

- **Multi-framework**: TensorRT, PyTorch (TorchScript), TensorFlow, ONNX Runtime, Python,
  OpenVINO — one server, many backends.
- **Dynamic batching**: combines individual requests into batches server-side, with a
  configurable max delay. Large throughput gains for free.
- **Concurrent model execution**: multiple models, or multiple instances of one model, on
  the same GPU.
- **Model repository + versioning**: load, unload and hot-swap model versions without
  restarting.
- **Model ensembles / business-logic scripting**: chain preprocessing → model →
  postprocessing inside the server.
- **HTTP/REST and gRPC endpoints**, plus a C API.
- **Prometheus metrics** out of the box: request counts, latency percentiles, queue time,
  GPU utilisation.
- Runs on Kubernetes, in the cloud, on-prem, or at the edge.

## NVIDIA NIM

**NVIDIA Inference Microservices** — prebuilt containers that package an optimized model
plus its runtime behind a **standard, OpenAI-compatible API**.

Why it appears on the exam: it is the "deploy quickly, anywhere, without doing the
optimization yourself" answer. NIM containers include TensorRT-LLM engines tuned for the
target GPU, and run in your own cloud, data centre or workstation — so data never leaves
your control.

## NeMo Framework

NVIDIA's end-to-end framework for **building and customizing** generative AI models —
the step *before* serving:

- **Data curation** (with NeMo Curator), **pretraining**, **SFT**, **PEFT/LoRA**,
  **alignment**, **evaluation**, and **export/deploy** to TensorRT-LLM and Triton.
- Built on PyTorch; uses **Megatron-LM** for large-scale distributed training.
- Companion products: **NeMo Guardrails** (safety and topic rails), **NeMo Retriever**
  (embedding and reranking microservices for RAG), **NeMo Evaluator**.

!!! tip "Product mapping cheat sheet"
    | Need | Product |
    | --- | --- |
    | Curate and dedupe a training corpus | **NeMo Curator** |
    | Pretrain / fine-tune / align a model | **NeMo Framework** (Megatron-LM underneath) |
    | Optimize an LLM for inference | **TensorRT-LLM** (TensorRT for general DL models) |
    | Serve models in production | **Triton Inference Server** |
    | Deploy a ready-made optimized endpoint fast | **NIM** |
    | Add safety/topic rails to an LLM app | **NeMo Guardrails** |
    | Embeddings + reranking for RAG | **NeMo Retriever** |
    | GPU-accelerated dataframes and classical ML | **RAPIDS** (cuDF, cuML, cuGraph) |

## Deployment patterns

- **Blue/green** — run the new version alongside the old, switch traffic atomically,
  roll back instantly.
- **Canary** — send 1–5% of traffic to the new version, watch metrics, ramp gradually.
- **Shadow (mirroring)** — send a copy of production traffic to the new model without
  returning its responses. Zero user risk; excellent for validating a new model on real
  inputs.
- **A/B test** — split traffic deliberately to compare outcome metrics. See
  [Experiment Design](../domain-3/experiment-design.md).

**Autoscaling** for GPU inference is different from stateless web services: model loading
takes tens of seconds to minutes, so scale on **queue depth / batch saturation**, keep a
warm pool, and set generous scale-down cooldowns.

## Monitoring and observability (task 4.5)

Four layers, all of which can be asked about:

**1. Infrastructure** — GPU utilisation, GPU memory, temperature/power, host CPU/RAM,
network. (`nvidia-smi`, **DCGM** exporter → Prometheus/Grafana.)

**2. Service** — requests/sec, error rate, latency percentiles (p50/p95/p99), queue
depth, batch size, timeout rate. Never monitor the mean alone; **p99 is where users
suffer**.

**3. Model / LLM-specific** — input and output token counts, cost per request,
time-to-first-token, truncation rate, refusal rate, tool-call failure rate, retrieval
hit rate, guardrail trigger rate, cache hit rate.

**4. Quality** — this is the layer that distinguishes ML systems from ordinary services:

- **Data drift** — the input distribution shifts away from what the model was built for.
- **Concept drift** — the relationship between input and correct output changes.
- **Performance decay** — accuracy falls as the world moves on.
- **User feedback** — thumbs up/down, escalation rate, task completion.
- **Sampled human review** and **continuous offline evaluation** on a fixed eval set.

!!! warning "Drift is silent"
    Nothing errors. Latency is fine, the dashboard is green, and the answers are quietly
    getting worse. Detecting drift requires explicitly comparing the current input/output
    distribution to a training or reference baseline, and re-running an eval set on a
    schedule.

**Logging and tracing**: capture the full request trace — prompt, retrieved chunks, model
version, parameters, output, latency, cost. This is what makes an incident debuggable.
Redact PII before it is stored, and set a retention policy.

## Reliability practices

- **Timeouts** on every model call; **retries with exponential backoff** on transient
  failures only.
- **Circuit breakers** so a failing dependency does not cascade.
- **Fallbacks**: a smaller/faster model, a cached response, or an honest error message.
- **Rate limiting and quotas** per user or tenant.
- **Health and readiness probes** — readiness must not pass until the model is loaded.
- **Graceful degradation** — if retrieval is down, answer without RAG *and say so*,
  rather than failing outright.

## MLOps / LLMOps

- Version **models, prompts, datasets and indexes** — all four.
- **CI/CD** should run the evaluation suite and block on regressions, exactly like tests.
- **Model registry** (MLflow, W&B, NGC) with lineage: which data, which code, which
  hyperparameters produced this artefact.
- **Reproducibility**: pin seeds, library versions, container images and model revisions.

## Key takeaways

- **Triton serves; TensorRT-LLM optimizes; NeMo builds; NIM packages; Guardrails protect.**
- Scalability = autoscaling + batching + replicas; performance = latency/throughput
  tuning; reliability = timeouts, retries, fallbacks, canaries.
- Monitor infrastructure, service, model **and quality**; track p95/p99, not just the mean.
- **Data drift and concept drift are silent failures** — detect them by comparing
  distributions and re-running eval sets on a schedule.
- Shadow deployment validates a new model on real traffic with zero user risk.
