# Hardware & Memory Sizing

*Covers task 4.4: "identify system data, hardware, or software components required to
meet user needs."*

## GPU memory arithmetic

The single most useful calculation in the whole domain.

**Bytes per parameter:**

| Precision | Bytes |
| --- | --- |
| FP32 | 4 |
| FP16 / BF16 | 2 |
| FP8 / INT8 | 1 |
| INT4 | 0.5 |

### Inference

$$ \text{weights memory} \approx \text{params} \times \text{bytes per param} $$

Then add the **KV cache** and activation working space; a practical rule is to add
**~20–30% overhead** on top of the weights, plus the KV cache.

| Model | FP16 | INT8 | INT4 |
| --- | --- | --- | --- |
| 7B | ~14 GB | ~7 GB | ~3.5 GB |
| 13B | ~26 GB | ~13 GB | ~6.5 GB |
| 70B | ~140 GB | ~70 GB | ~35 GB |

!!! tip "The shortcut to memorise"
    **In FP16, a model needs roughly 2 GB of GPU memory per billion parameters** — before
    the KV cache. Halve it for INT8, quarter it for INT4.

    That is why a 7B model fits comfortably on a 24 GB card, a 70B model in FP16 needs
    multiple GPUs (2× 80 GB), and INT4 quantization brings 70B down to a single 48 GB card.

**KV cache size:**

$$ 2 \times \text{layers} \times \text{seq len} \times \text{kv heads} \times \text{head dim} \times \text{batch} \times \text{bytes} $$

The leading 2 covers K and V. Note it scales **linearly with batch size and sequence
length** — which is why long-context, high-concurrency serving is memory-bound and why
GQA/MQA (fewer KV heads) matters so much.

### Training

Training needs far more than inference — typically **12–20× the parameter count in
bytes** for full fine-tuning with Adam:

| Component | Cost (FP16 compute, FP32 master, Adam) |
| --- | --- |
| Parameters (FP16) | 2 bytes/param |
| Gradients (FP16) | 2 bytes/param |
| FP32 master weights | 4 bytes/param |
| Adam optimizer states (m, v) | 8 bytes/param |
| **Subtotal** | **~16 bytes/param** |
| Activations | Depends on batch size, sequence length, layers — often the largest term |

So a 7B model full-fine-tuned needs roughly **112 GB+** just for states — well beyond a
single 80 GB GPU. This is precisely why **LoRA/QLoRA** exist: with the base frozen, you
carry no gradients or optimizer states for 99%+ of the parameters, and QLoRA quantizes
the frozen base to 4 bits on top of that.

## GPU characteristics that matter

| Property | Why it matters |
| --- | --- |
| **VRAM capacity** | Hard limit on what fits. The first thing to check. |
| **Memory bandwidth (HBM)** | LLM decoding is bandwidth-bound — this often determines tokens/sec more than FLOPS. |
| **Tensor Cores** | Dedicated matrix-multiply units; the source of FP16/BF16/FP8/INT8 speedups. |
| **NVLink / NVSwitch** | High-bandwidth GPU↔GPU interconnect. Essential for tensor parallelism. |
| **Multi-Instance GPU (MIG)** | Partition one GPU into isolated instances — good for many small models. |
| **Compute capability** | Determines which precisions and kernels are available (e.g. FP8 needs Hopper or newer). |

Datacenter generations, oldest to newest: **Volta → Turing → Ampere (A100) → Hopper
(H100/H200) → Blackwell (B200/GB200)**.

## Sizing decision framework

Given a described workload, ask in this order:

1. **What is the task?** Classification/NER/embeddings → a small encoder model may be
   enough (and 100× cheaper than an LLM).
2. **Does the model fit?** `params × bytes` + KV cache. If not: quantize, choose a smaller
   model, or add GPUs with tensor parallelism.
3. **Latency or throughput?** Interactive chat → optimise TTFT, small batches, streaming.
   Batch processing → maximise batch size and throughput.
4. **What concurrency?** Concurrency × context length drives KV cache, which drives VRAM.
5. **Training or inference?** Training multiplies memory by ~8–10× over inference.
6. **Data residency and cost?** On-prem/VPC vs. hosted API; GPU-hours vs. per-token.

!!! note "CPU vs. GPU"
    GPUs win when the work is massively parallel matrix arithmetic — training and LLM
    inference. CPUs remain fine for small classical models, TF-IDF pipelines, data
    wrangling and low-volume small-model inference. For classical ML on large tabular
    data, **RAPIDS** brings the GPU speedup without leaving the pandas/scikit-learn API —
    see [Accelerated Data Science](../domain-4/rapids.md).

## Other components in the system

- **Storage** — training data throughput matters; a slow data loader starves expensive
  GPUs. Use fast local NVMe, memory-mapped datasets, prefetching and multiple loader
  workers.
- **CPU and RAM** — data loading, tokenization and preprocessing are CPU-bound; an
  under-provisioned host bottlenecks the GPU.
- **Network** — multi-node training needs InfiniBand or high-speed RoCE; Ethernet at
  10 GbE will not keep up with gradient synchronisation.
- **Containers and orchestration** — Docker with the NVIDIA Container Toolkit, Kubernetes
  with the GPU Operator, images from **NGC** (NVIDIA GPU Cloud).

## Key takeaways

- **~2 GB per billion parameters in FP16** for inference weights; halve per precision step down.
- Full fine-tuning needs **~16 bytes per parameter** for states alone — the reason LoRA
  and QLoRA exist.
- KV cache scales with batch × sequence length; it can exceed the weights at long context.
- Decoding is **memory-bandwidth bound**, so bandwidth often matters more than raw FLOPS.
- Match the hardware to the objective: latency (small batches, streaming) vs. throughput
  (large batches, continuous batching).
