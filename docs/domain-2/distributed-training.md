# Distributed Training

*NVIDIA's suggested readings for Software Development include **NCCL**, **AllReduce**,
**Ring-AllReduce** and "Technologies behind distributed deep learning". Expect at least
one or two questions.*

## Why distribute at all

Three independent reasons, and the exam tests whether you can tell them apart:

1. **The data is too large** to train on in reasonable time → **data parallelism**.
2. **The model does not fit** in one GPU's memory → **model parallelism** (tensor and/or
   pipeline).
3. **Both** → 3D parallelism, which is how frontier models are actually trained.

## Data parallelism

The default and simplest strategy.

- **Replicate the full model** on every GPU.
- Split each mini-batch into shards, one per GPU.
- Each GPU computes gradients on its shard **independently**.
- **Synchronise the gradients across all GPUs** — this is the **AllReduce** step.
- Every GPU applies the same averaged update, so replicas stay identical.

Requirement: the model plus optimizer states must fit on a single GPU.

In PyTorch this is **DDP (DistributedDataParallel)**. The **effective batch size** is
`per_gpu_batch × num_gpus`, and the learning rate normally needs scaling up to match
(with warmup).

## Collective communication and AllReduce

A **collective operation** involves all participating processes. The ones to know:

| Operation | Effect |
| --- | --- |
| **Broadcast** | One rank's data is copied to all ranks |
| **Reduce** | Combine (sum/max/…) data from all ranks into **one** rank |
| **AllReduce** | Combine data from all ranks; **every** rank gets the result |
| **Gather / AllGather** | Collect data from all ranks into one / into all |
| **Scatter** | Distribute distinct chunks from one rank to all |
| **ReduceScatter** | Reduce, then each rank keeps a different slice of the result |

**AllReduce is the core primitive of data-parallel training**: sum the gradients across
all GPUs and hand the sum back to everyone.

### Ring-AllReduce

The bandwidth-optimal algorithm, and the one NVIDIA's reading list highlights.

GPUs are arranged in a logical ring. The gradient vector is split into *N* chunks (for
*N* GPUs), and the algorithm runs in two phases:

1. **ReduceScatter** — *N−1* steps. At each step, every GPU sends one chunk to its
   neighbour and accumulates the chunk it receives. After the phase, each GPU holds one
   *fully reduced* chunk.
2. **AllGather** — *N−1* steps. The fully reduced chunks are passed around the ring until
   every GPU has all of them.

**Why it matters:** each GPU sends and receives roughly `2 × (N−1)/N × data_size` bytes —
**independent of the number of GPUs** for a fixed data size. A naive
gather-to-one-master approach makes the master a bottleneck that worsens linearly with
*N*. Ring-AllReduce is what makes data parallelism scale.

### NCCL

**NVIDIA Collective Communications Library** — the optimized implementation of these
collectives for NVIDIA GPUs.

- Topology-aware: automatically uses **NVLink/NVSwitch** inside a node and
  **InfiniBand/RoCE** between nodes.
- Multi-GPU and multi-node.
- The communication backend behind PyTorch DDP/FSDP, Megatron-LM, NeMo, Horovod and
  DeepSpeed.
- Overlaps communication with computation to hide latency.

!!! tip "One-liner"
    **NCCL is the library; AllReduce is the operation; Ring-AllReduce is the algorithm.**

## Model parallelism

When the model itself will not fit.

**Tensor parallelism (intra-layer)** — split *individual* weight matrices across GPUs.
Each GPU computes a slice of the same layer, and results are combined with AllReduce/AllGather.
Communication happens **many times per layer**, so it demands very fast interconnect
(NVLink) and is normally kept **within a single node**. Pioneered by **Megatron-LM**.

**Pipeline parallelism (inter-layer)** — assign *different layers* to different GPUs;
activations flow from stage to stage. Communication is small (only at stage boundaries),
so it works **across nodes**. Its weakness is the **pipeline bubble** — idle time while
stages wait — mitigated by splitting batches into **micro-batches** (GPipe, 1F1B
interleaved schedules).

**Expert parallelism** — for MoE models, place different experts on different GPUs and
route tokens to them.

**Sequence/context parallelism** — split the sequence dimension, for very long contexts.

| | Tensor parallel | Pipeline parallel |
| --- | --- | --- |
| Splits | Within a layer | Across layers |
| Communication | High frequency, high volume | Low frequency, at boundaries |
| Best placement | Inside one node (NVLink) | Across nodes |
| Main inefficiency | Communication overhead | Pipeline bubbles |

**3D parallelism** = data × tensor × pipeline, combined. This is how models with hundreds
of billions of parameters are trained.

## Memory-optimization techniques

- **ZeRO (DeepSpeed) / FSDP (PyTorch)** — shard optimizer states (stage 1), gradients
  (stage 2) and parameters (stage 3) across data-parallel ranks instead of replicating
  them. Dramatic memory savings with data-parallel simplicity; parameters are gathered
  just in time for each layer's forward/backward.
- **Activation (gradient) checkpointing** — do not store all activations for the backward
  pass; recompute them. Trades ~30% extra compute for a large memory saving.
- **Mixed precision (AMP)** — compute in FP16/BF16 with an FP32 master copy and loss
  scaling. ~2× faster, ~half the activation memory.
- **Gradient accumulation** — run several micro-batches before stepping the optimizer, to
  simulate a large batch on limited memory.
- **CPU/NVMe offloading** — park optimizer states off-GPU. Slow, but it works.

## Scaling laws in practice

NVIDIA lists *"Deep Learning Scaling Is Predictable, Empirically"*. Two practical
consequences:

- **Strong scaling** — fixed total problem size, more GPUs. Limited by communication
  overhead and Amdahl's law; efficiency drops as GPU count rises.
- **Weak scaling** — problem size grows with GPU count. Scales much better; this is how
  large-model training is actually organised.

Loss improves as a **power law** in compute, data and parameters, which is what makes
multimillion-dollar training runs plannable rather than speculative.

## Key takeaways

- **Data parallel** = replicate the model, split the data, **AllReduce** the gradients.
- **Tensor parallel** = split within a layer (chatty, keep it inside a node).
  **Pipeline parallel** = split across layers (cheap communication, watch the bubble).
- **Ring-AllReduce** makes communication volume independent of GPU count; **NCCL** is
  NVIDIA's implementation.
- Model does not fit → tensor/pipeline parallelism, ZeRO/FSDP, activation checkpointing,
  mixed precision.
- Frontier training combines all three axes: **3D parallelism**.
