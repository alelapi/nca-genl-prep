# Hardware & Memory Sizing

*Covers task 4.4: "identify system data, hardware, or software components required to meet user
needs."*

This page is mostly arithmetic, and it is arithmetic worth being able to do in your head. A
surprising number of exam scenarios reduce to "will this fit?", and being able to answer in ten
seconds converts several questions from guesses into certainties.

---

## 1. The fundamental calculation

Everything starts with **bytes per parameter**.

| Precision | Bytes per parameter |
| --- | --- |
| FP32 | 4 |
| FP16 / BF16 | 2 |
| FP8 / INT8 | 1 |
| INT4 | 0.5 |

### Inference memory

$$ \text{weights memory} \approx \text{parameters} \times \text{bytes per parameter} $$

Then add the **KV cache** and some activation working space — a practical rule is **+20–30%
overhead** on top of the weights, plus the KV cache computed separately.

| Model | FP16 | INT8 | INT4 |
| --- | --- | --- | --- |
| 1B | ~2 GB | ~1 GB | ~0.5 GB |
| 7B | ~14 GB | ~7 GB | ~3.5 GB |
| 13B | ~26 GB | ~13 GB | ~6.5 GB |
| 34B | ~68 GB | ~34 GB | ~17 GB |
| 70B | ~140 GB | ~70 GB | ~35 GB |

!!! tip "The shortcut to memorise"
    **In FP16, a model needs roughly 2 GB of GPU memory per billion parameters** — weights only,
    before the KV cache. Halve it for INT8, quarter it for INT4.

    From that one number you can answer most sizing questions instantly:

    - A 7B model fits comfortably on a 24 GB consumer card.
    - A 13B model in FP16 (26 GB) does **not** fit on 24 GB — quantize it, or use a bigger card.
    - A 70B model in FP16 (140 GB) needs **two 80 GB GPUs** with tensor parallelism.
    - A 70B model at INT4 (35 GB) fits on a single 48 GB card.

### The KV cache

$$ \text{KV bytes} = 2 \times \text{layers} \times \text{seq len} \times \text{kv heads} \times \text{head dim} \times \text{batch} \times \text{bytes} $$

The leading 2 accounts for K and V. Note the two terms that scale linearly and will hurt you:
**batch size** and **sequence length**.

Worked, for Llama-2-7B in FP16 (32 layers, 32 heads, head dim 128, no GQA):

```text
per token, per layer:    2 × 32 × 128 × 2 bytes   =   16 KB
per token, all layers:   16 KB × 32               =  512 KB
4,096-token context:     512 KB × 4,096           ≈    2 GB   per sequence

weights:                                              14 GB
+ 16 concurrent users at 4k context (16 × 2 GB):      32 GB
                                                    ────────
                                                      46 GB
```

**The KV cache exceeds the weights.** This is the number that surprises people, and it is why
long-context, high-concurrency serving is a memory problem rather than a compute problem.

With **GQA** at 8 KV heads instead of 32, that 32 GB becomes **8 GB** — which is precisely why
every modern model uses it.

---

## 2. Training memory — and why PEFT exists

Training needs vastly more than inference. For mixed-precision full fine-tuning with Adam:

| Component | Bytes per parameter | Why |
| --- | --- | --- |
| Parameters (FP16) | 2 | The weights used in the forward pass |
| Gradients (FP16) | 2 | One gradient per parameter |
| FP32 master weights | 4 | Updates are accumulated in full precision to avoid rounding away small changes |
| Adam momentum `m` | 4 | First moment estimate |
| Adam variance `v` | 4 | Second moment estimate |
| **Total state** | **≈ 16** | |
| Activations | varies | Depends on batch size, sequence length, layers — **often the largest term** |

Apply it:

```text
7B model, full fine-tuning:    7e9 × 16 bytes  =  112 GB   of state alone
                                                  + activations
                                                  ✗ exceeds a single 80 GB H100
```

!!! important "This table is the reason LoRA and QLoRA exist"
    With LoRA, the base weights are **frozen**: no gradients, no FP32 master copy, no Adam
    moments for 99%+ of parameters. You store the weights for the forward pass and full training
    state only for the tiny adapter.

    ```text
    7B full fine-tuning  ≈ 112 GB   multiple A100/H100s
    7B LoRA  (FP16 base) ≈  20 GB   one 24 GB card, tightly
    7B QLoRA (4-bit base)≈   6 GB   comfortable on consumer hardware
    ```

    Recognising a question as "the memory arithmetic says full fine-tuning is impossible here"
    is usually enough to pick the right answer. See [Customization & PEFT](customization.md).

**Reducing training memory** without changing strategy: activation checkpointing, mixed
precision, gradient accumulation, ZeRO/FSDP sharding, CPU offloading. All covered in
[Distributed Training](distributed-training.md).

---

## 3. What actually matters about a GPU

| Property | Why it matters |
| --- | --- |
| **VRAM capacity** | The hard limit on what fits. Always check first |
| **Memory bandwidth (HBM)** | LLM decoding is bandwidth-bound, so this often determines tokens/second more than raw FLOPS |
| **Tensor Cores** | Dedicated matrix-multiply units; the source of FP16/BF16/FP8/INT8 speedups |
| **NVLink / NVSwitch** | High-bandwidth GPU↔GPU interconnect. **Essential for tensor parallelism** |
| **Compute capability** | Determines which precisions are available — FP8 requires Hopper or newer |
| **MIG (Multi-Instance GPU)** | Partition one GPU into isolated instances — good for many small models |

!!! note "Bandwidth over FLOPS — the counterintuitive one"
    For **inference**, a GPU with more memory bandwidth often beats one with more FLOPS, because
    decode is bandwidth-bound (see
    [Inference Optimization](inference-optimization.md#why-llm-inference-is-slow--the-fact-that-explains-everything-else)).
    For **training**, which is compute-heavy, FLOPS matter more.

    That is the reasoning behind H200 over H100 for inference workloads: same compute, more
    memory and more bandwidth.

Datacenter generations, oldest to newest: **Volta → Turing → Ampere (A100) → Hopper (H100/H200)
→ Blackwell (B200/GB200)**.

---

## 4. A sizing decision framework

Given a scenario, work through these in order:

**1. What is the task, really?** Classification, NER or embeddings may not need an LLM at all —
a 110M-parameter encoder can be 100× cheaper than a 7B decoder and just as accurate for
classification. This is the highest-leverage question and the one most often skipped.

**2. Does it fit?** `parameters × bytes` + KV cache. If not: quantize, choose a smaller model,
or add GPUs with tensor parallelism.

**3. Latency or throughput?** Interactive chat → optimise TTFT, small batches, streaming. Batch
processing → maximise batch size and GPU utilisation. See
[Inference Optimization](inference-optimization.md).

**4. What concurrency and context length?** Multiply them: this drives the KV cache, which drives
VRAM, which drives how many GPUs you need. A 32k-context product at 50 concurrent users is a
fundamentally different sizing problem from a 2k-context product at the same concurrency.

**5. Training or inference?** Training multiplies memory by roughly 8–10× over inference.

**6. Data residency and cost model?** On-prem or VPC (GPU-hours, cheap at high sustained volume,
data stays inside) vs. hosted API (per-token, cheap at low or spiky volume, data leaves). See
[Privacy & Consent](../domain-5/privacy-consent.md).

### CPU vs. GPU

GPUs win where the work is massively parallel matrix arithmetic — training and LLM inference.
CPUs remain perfectly appropriate for small classical models, TF-IDF pipelines, data wrangling,
and low-volume small-model inference where a GPU would sit idle between requests.

For classical ML on **large tabular data**, **RAPIDS** delivers the GPU speedup without leaving
the pandas/scikit-learn API — see [Accelerated Data Science](../domain-4/rapids.md).

---

## 5. The rest of the system (task 4.4)

The blueprint says "system data, hardware, **or software** components", so the GPU is not the
whole answer.

**Storage.** Training data throughput matters more than people expect — a slow data loader
starves expensive GPUs, and a cluster running at 40% utilisation because of I/O is an expensive
mistake. Use fast local NVMe, memory-mapped datasets, prefetching and multiple loader workers.

**CPU and host RAM.** Data loading, tokenization and preprocessing are CPU-bound. An
under-provisioned host bottlenecks the GPUs it is meant to feed.

**Network.** Multi-node training needs **InfiniBand** or high-speed RoCE. Standard 10 GbE cannot
sustain gradient synchronisation for a large model, and the training run will spend most of its
time waiting on the network.

**Software.** Docker with the NVIDIA Container Toolkit; Kubernetes with the GPU Operator; base
images from **NGC** (NVIDIA GPU Cloud); a matched CUDA/driver/framework stack. Version mismatches
between CUDA, cuDNN, the driver and the framework are a classic and time-consuming failure.

---

## 6. Recap

- **~2 GB per billion parameters in FP16** for inference weights. Halve per precision step down.
- **KV cache** scales linearly with batch size and sequence length, and at production concurrency
  it commonly **exceeds the model weights**. GQA is the structural fix.
- Full fine-tuning needs **~16 bytes per parameter** of state — 112 GB for a 7B model. This is
  the arithmetic that makes **LoRA/QLoRA** necessary rather than merely convenient.
- Check in order: **VRAM capacity** first, then **memory bandwidth** (decode is bandwidth-bound),
  then compute.
- **NVLink is required for practical tensor parallelism**; pipeline parallelism tolerates slower
  links.
- Match hardware to objective: latency (small batches, streaming) vs. throughput (large batches,
  continuous batching).
- Do not forget storage throughput, host CPU, and InfiniBand for multi-node training — any of
  them can leave expensive GPUs idle.
