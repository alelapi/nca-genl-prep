# Deployment & Serving

*Covers tasks 1.1 / 4.1 ("assist in deployment and evaluation of model scalability, performance
and reliability") and 4.5 ("monitor functioning of data collection, experiments and other
software processes").*

---

## 1. The three qualities the blueprint names

The task statement is unusually precise — **scalability, performance and reliability** — and
each maps to concrete engineering.

| Quality | The question it answers | Levers |
| --- | --- | --- |
| **Scalability** | Does it hold up as load grows? | Horizontal replicas, autoscaling, batching, load balancing, multi-GPU parallelism |
| **Performance** | Is it fast and efficient enough? | Latency (TTFT/TPOT), throughput, GPU utilisation, quantization, TensorRT |
| **Reliability** | Does it keep working when things go wrong? | Health checks, timeouts, retries, circuit breakers, fallbacks, graceful degradation, canary rollouts |

These are genuinely different properties, and a system can be excellent at one and terrible at
another. A deployment that serves 10,000 requests per second (scalable) with a p99 latency of 30
seconds (not performant) that falls over when the vector database hiccups (not reliable) has
solved exactly one third of the problem.

---

## 2. NVIDIA Triton Inference Server

The serving layer of the NVIDIA stack, and the answer to any question about **hosting** models
in production.

**What it does:**

- **Multi-framework.** TensorRT, PyTorch (TorchScript), TensorFlow, ONNX Runtime, OpenVINO and
  Python backends — one server, many model types. This matters operationally: you run one piece
  of infrastructure instead of a different serving stack per framework.
- **Dynamic batching.** Triton combines individual requests arriving close together into a batch
  server-side, transparently to clients. Given that inference is
  [memory-bandwidth bound](inference-optimization.md), this is a large throughput gain for
  essentially no effort.
- **Concurrent model execution.** Multiple models, or multiple instances of the same model, on
  one GPU — so a small model does not leave the GPU idle.
- **Model repository and versioning.** Load, unload and hot-swap versions without restarting the
  server. Version policies let you serve v1 and v2 simultaneously, which is what makes canary
  and A/B deployments practical.
- **Model ensembles and business-logic scripting.** Chain preprocessing → model →
  postprocessing inside the server, so the client sends raw input and receives a final answer,
  with no network hop between stages.
- **HTTP/REST and gRPC endpoints**, plus a C API for embedding.
- **Prometheus metrics** out of the box: request counts, latency percentiles, **queue time**,
  batch sizes, GPU utilisation.
- Runs on Kubernetes, in the cloud, on-premises, or at the edge.

That "queue time" metric is more useful than it sounds — it separates *"the model is slow"* from
*"the model is fine but requests are waiting"*, which are completely different problems with
completely different fixes.

---

## 3. NVIDIA NIM

**NVIDIA Inference Microservices** — prebuilt containers packaging an optimized model plus its
runtime behind a **standard, OpenAI-compatible API**.

The reasoning behind it: getting peak inference performance out of a given model on a given GPU
requires TensorRT-LLM engine building, quantization calibration, batching configuration and
kernel tuning. That is specialist work, it must be redone per model and per GPU, and most teams
should not be doing it. NIM ships the result pre-built.

Why it appears on the exam: it is the answer to *"deploy an optimized model quickly, anywhere,
without doing the optimization work yourself"*. And because it runs **in your own cloud, data
centre or workstation**, data never leaves your perimeter — which is often the deciding factor
for regulated industries.

---

## 4. NeMo Framework

NVIDIA's end-to-end framework for **building and customizing** models — the step *before*
serving.

Covers **data curation** (with NeMo Curator), **pretraining**, **SFT**, **PEFT/LoRA**,
**alignment**, **evaluation**, and **export** to TensorRT-LLM and Triton. Built on PyTorch, using
**Megatron-LM** underneath for large-scale distributed training.

The companion products:

- **NeMo Curator** — data curation and deduplication at scale
- **NeMo Guardrails** — safety and topic rails, see [Domain 5](../domain-5/guardrails-security.md)
- **NeMo Retriever** — embedding and reranking microservices for RAG
- **NeMo Evaluator** — model and RAG evaluation

!!! tip "The product map — likely free marks"
    | Need | Product |
    | --- | --- |
    | Curate, dedupe, PII-strip a training corpus | **NeMo Curator** |
    | Pretrain, fine-tune or align a model | **NeMo Framework** (Megatron-LM underneath) |
    | Optimize an LLM for inference | **TensorRT-LLM** (TensorRT for general DL models) |
    | Serve models in production | **Triton Inference Server** |
    | Deploy a ready-optimized endpoint fast | **NIM** |
    | Add safety/topic rails to an LLM app | **NeMo Guardrails** |
    | Embeddings and reranking for RAG | **NeMo Retriever** |
    | GPU-accelerated dataframes and classical ML | **RAPIDS** (cuDF, cuML, cuGraph) |
    | Multi-GPU collective communication | **NCCL** |

    And the pair most often swapped in distractors: **TensorRT optimizes; Triton serves.**

---

## 5. Deployment patterns

| Pattern | How it works | Best for |
| --- | --- | --- |
| **Blue/green** | Run the new version alongside the old, switch all traffic at once | Instant rollback; needs double the capacity briefly |
| **Canary** | Route 1–5% of traffic to the new version, watch metrics, ramp gradually | Limiting blast radius on a real but small population |
| **Shadow (mirroring)** | Send a **copy** of production traffic to the new model and **discard its responses** | **Zero user risk.** Validating a new model on real inputs before anyone sees it |
| **A/B test** | Deliberately split traffic to compare outcome metrics | Deciding *which is better*, not just *whether it works* |

The distinction between **shadow** and **canary** is worth being precise about, because it
appears in questions:

- **Shadow** — nobody sees the new model's output. You learn whether it crashes, how fast it is,
  and how its answers compare offline. You learn **nothing** about how users react.
- **Canary** — a small number of real users see it. You learn about user reaction, at the cost
  of exposing them to risk.

*"Validate on real production traffic with zero risk to users"* → **shadow**, every time.

### Autoscaling GPU inference is different

Standard web autoscaling assumes a new instance is ready in seconds. GPU model serving does not:
loading a 14 GB model from storage into VRAM takes tens of seconds to minutes, so by the time
your new replica is ready the traffic spike is over.

Practical adjustments:

- **Scale on queue depth or batch saturation**, not CPU utilisation.
- **Keep a warm pool** — some idle headroom is cheaper than dropped requests.
- **Set generous scale-down cooldowns** so you do not terminate a replica you will need again in
  90 seconds.
- Consider **model caching** on local NVMe to shorten cold starts.

---

## 6. Monitoring and observability (task 4.5)

Four layers, and the fourth is what distinguishes ML systems from ordinary services.

### Layer 1 — Infrastructure

GPU utilisation, GPU memory, temperature and power, host CPU and RAM, network. Collected via
`nvidia-smi` and, in production, the **DCGM** exporter feeding Prometheus and Grafana.

The most informative single number is **GPU utilisation**: consistently low utilisation while
requests queue means you are bottlenecked somewhere else — data loading, tokenization, the
network, or batching that is not configured.

### Layer 2 — Service

Requests per second, error rate, latency percentiles, queue depth, batch size, timeout rate.

!!! warning "Never monitor the mean alone"
    A mean latency of 400 ms is compatible with 95% of users getting 200 ms and 5% getting 8
    seconds. **The p99 is where users actually suffer**, and it is the number that generates
    support tickets. Track p50, p95 and p99 — the *shape* tells you the cause. A high p99 with a
    normal p50 usually means queueing, cold starts, or a few very long prompts.

### Layer 3 — Model / LLM-specific

Metrics that only exist because this is an LLM:

- Input and output **token counts** (and therefore **cost per request**)
- **Time to first token**
- **Truncation rate** — how often output hits the max-token cap. A high rate means answers are
  being cut off mid-sentence
- **Refusal rate** — a sudden rise suggests over-cautious guardrails or a prompt regression
- **Retrieval hit rate** and reranker score distribution
- **Guardrail trigger rate**
- **Cache hit rate**
- **Tool-call failure rate** for agents

### Layer 4 — Quality

The layer that has no analogue in normal software engineering.

- **Data drift** — the input distribution moves away from what the system was built for. New
  product names, a new customer segment, a new language.
- **Concept drift** — the *correct answer* changes even though the inputs look the same. A policy
  was updated; the right response to an identical question is now different.
- **Performance decay** — accuracy falls as the world moves on.
- **User feedback** — thumbs up/down, escalation rate, task completion, conversation abandonment.
- **Sampled human review** and **continuous offline evaluation** against a fixed eval set.

!!! danger "Drift is a silent failure"
    Nothing errors. Latency is normal. Every dashboard is green. And the answers are quietly
    getting worse.

    This is the failure mode that standard monitoring **structurally cannot detect**, because
    there is nothing to detect — no exception, no status code, no latency change. Detecting it
    requires explicitly comparing the current input/output distribution against a reference
    baseline and **re-running an evaluation set on a schedule**.

    Questions asking "which failure would standard service monitoring miss?" are asking about
    drift.

### Logging and tracing

Capture the full request trace: prompt, retrieved chunks and their scores, model version, prompt
version, decoding parameters, output, latency, token counts, cost. Without this you cannot debug
a bad answer after the fact — and "the model said something wrong yesterday" is the most common
incident report an LLM system produces.

Two obligations: **redact PII before storing**, and set a **retention policy**. Prompt logs are
one of the most under-appreciated sources of personal-data leakage. See
[Privacy & Consent](../domain-5/privacy-consent.md).

---

## 7. Reliability practices

- **Timeouts** on every model call. An LLM call with no timeout can hang for minutes.
- **Retries with exponential backoff** — on *transient* failures only. Retrying a 400 Bad Request
  just multiplies load.
- **Circuit breakers** so a failing dependency stops being called instead of cascading.
- **Fallbacks** — a smaller/faster model, a cached response, or an honest error. Decide in
  advance rather than during an incident.
- **Rate limiting and quotas** per user or tenant, so one client cannot exhaust the GPU pool.
- **Health and readiness probes.** Readiness must not pass until the model is **loaded** —
  otherwise Kubernetes routes traffic to a replica that will time out every request.
- **Graceful degradation.** If retrieval is down, answer without RAG **and say so**, rather than
  failing entirely or — worse — silently answering ungrounded.

---

## 8. MLOps / LLMOps

An LLM application has **four** versioned dependencies where ordinary software has one:

```text
  code   +   model version   +   prompt version   +   index version
```

Any of them changing silently changes behaviour. The discipline follows:

- **Version all four**, pin them in production, and record which combination produced any given
  output.
- **CI runs the evaluation suite** and blocks on regressions, exactly like tests. A prompt change
  is a code change.
- **Model registry** (MLflow, Weights & Biases, NGC) with lineage: which data, which code, which
  hyperparameters produced this artefact.
- **Reproducibility**: pin seeds, library versions, container images and model revisions.

!!! tip "The practice that separates real systems from demos"
    Treat the **prompt and the index as production artefacts**, not configuration. They have as
    much influence on behaviour as the model, they change more often, and they are the two that
    teams routinely edit without review or testing.

---

## 9. Recap

- **Scalability, performance, reliability** are distinct properties with distinct levers.
- **Triton serves. TensorRT-LLM optimizes. NeMo builds. NIM packages. Guardrails protect.**
- Triton gives you dynamic batching, concurrent model execution, versioning, HTTP/gRPC and
  Prometheus metrics across multiple frameworks.
- **Shadow deployment** validates on real traffic with zero user risk; **canary** exposes a small
  real population.
- GPU autoscaling must account for slow model loading — scale on queue depth, keep warm capacity.
- Monitor four layers: infrastructure, service (**p95/p99, not the mean**), LLM-specific metrics,
  and **quality**.
- **Data drift and concept drift are silent** — no errors, no latency change. Detect by comparing
  distributions and re-running eval sets on a schedule.
- Reliability: timeouts, backoff, circuit breakers, fallbacks, readiness probes that wait for
  model load, and graceful degradation that tells the user.
- Version **code, model, prompt and index** — all four.
