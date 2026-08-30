# Distributed Training

*NVIDIA's suggested readings for Software Development include **NCCL**, **AllReduce**,
**Ring-AllReduce**, "Technologies behind distributed deep learning" and "Deep Learning Scaling
Is Predictable, Empirically". Expect one or two questions, and they will be about telling the
parallelism strategies apart.*

---

## 1. Three different reasons to use more than one GPU

The exam tests whether you can distinguish these, because each has a different answer.

**Reason 1: training would take too long.** The model fits fine on one GPU; there is just too
much data to get through. → **Data parallelism.**

**Reason 2: the model does not fit.** A 70B model needs ~140 GB in FP16 just for weights, and
full fine-tuning needs far more. No single GPU has that. → **Model parallelism** (tensor and/or
pipeline).

**Reason 3: both.** → 3D parallelism, which is how every frontier model is actually trained.

Getting this triage right resolves most questions in this section: *"the model does not fit in
GPU memory"* is never answered by data parallelism, because data parallelism **replicates** the
model on every GPU.

---

## 2. Data parallelism

The simplest strategy and the default.

```text
                    mini-batch of 256 examples
                              │
        ┌─────────────┬───────┴───────┬─────────────┐
        ▼             ▼               ▼             ▼
     64 ex.        64 ex.          64 ex.        64 ex.
   ┌────────┐    ┌────────┐     ┌────────┐    ┌────────┐
   │ GPU 0  │    │ GPU 1  │     │ GPU 2  │    │ GPU 3  │
   │ FULL   │    │ FULL   │     │ FULL   │    │ FULL   │
   │ model  │    │ model  │     │ model  │    │ model  │
   └───┬────┘    └───┬────┘     └───┬────┘    └───┬────┘
       │             │              │             │
     grads         grads          grads         grads
       └─────────────┴──────┬───────┴─────────────┘
                            ▼
                    ALL-REDUCE (sum and share)
                            │
        ┌─────────────┬─────┴─────────┬─────────────┐
        ▼             ▼               ▼             ▼
   every GPU applies the SAME averaged update → replicas stay identical
```

The steps:

1. **Replicate** the full model on every GPU.
2. **Split** each mini-batch into shards, one per GPU.
3. Each GPU computes gradients on its shard **independently** — this part is embarrassingly
   parallel.
4. **Synchronise** the gradients across all GPUs. This is the **AllReduce** step, and it is the
   only communication.
5. Every GPU applies the same averaged update, so all replicas remain bit-identical.

**The requirement:** the model plus its optimizer states must fit on a single GPU. If it does
not, data parallelism alone cannot help you.

In PyTorch this is **DDP (DistributedDataParallel)**. Note that the **effective batch size**
becomes `per_gpu_batch × num_gpus`, which usually means the learning rate needs scaling up (with
warmup) to match.

---

## 3. Collective communication and AllReduce

A **collective operation** is one that involves all participating processes together. These
names come up, so recognise them:

| Operation | Effect |
| --- | --- |
| **Broadcast** | One rank's data is copied to all ranks |
| **Reduce** | Combine (sum, max, …) data from all ranks into **one** rank |
| **AllReduce** | Combine data from all ranks; **every** rank receives the result |
| **Gather / AllGather** | Collect data from all ranks into one rank / into all ranks |
| **Scatter** | Distribute distinct chunks from one rank to all |
| **ReduceScatter** | Reduce, then each rank keeps a different slice of the result |

**AllReduce is the core primitive of data-parallel training**: sum the gradients across all
GPUs and give the sum back to everyone, so every replica applies the same update.

### Ring-AllReduce — why it scales

This is the algorithm NVIDIA's reading list highlights, and it is worth understanding because
the naive alternative fails in an instructive way.

**The naive approach:** every GPU sends its gradients to a designated master, which sums them
and broadcasts the result. With `N` GPUs and gradient size `D`, the master must receive
`(N−1)·D` bytes and send `(N−1)·D` bytes. **The master's load grows linearly with GPU count**,
so it becomes a hard bottleneck. Doubling your cluster does not halve your training time.

**Ring-AllReduce** removes the master entirely. GPUs are arranged in a logical ring, and the
gradient vector is split into `N` chunks. The algorithm runs in two phases:

```text
Phase 1 — REDUCE-SCATTER  (N−1 steps)
Each GPU sends one chunk to its neighbour and accumulates the chunk it receives.
After N−1 steps, each GPU holds ONE fully-summed chunk.

    GPU0 ──► GPU1 ──► GPU2 ──► GPU3 ──┐
      ▲                                │
      └────────────────────────────────┘

Phase 2 — ALL-GATHER  (N−1 steps)
The fully-summed chunks are passed around the ring until every GPU has all of them.
```

**The payoff:** each GPU sends and receives roughly

$$ 2 \times \frac{N-1}{N} \times D \;\approx\; 2D \text{ bytes} $$

which is **effectively independent of `N`**. Going from 8 GPUs to 800 does not increase
per-GPU communication volume. It increases the number of *steps* (latency) but not the *bytes*
(bandwidth), and bandwidth is the binding constraint at scale.

That property is the entire reason large-scale data-parallel training is feasible.

### NCCL

**NVIDIA Collective Communications Library** — the optimized implementation of these collectives
for NVIDIA GPUs.

- **Topology-aware.** It automatically discovers the interconnect and uses **NVLink/NVSwitch**
  within a node and **InfiniBand/RoCE** between nodes, choosing algorithms accordingly.
- **Multi-GPU and multi-node.**
- It is the communication backend behind PyTorch DDP and FSDP, Megatron-LM, NeMo, DeepSpeed and
  Horovod. You rarely call it directly; you use it constantly.
- **Overlaps communication with computation** — gradients for later layers are ready first
  during the backward pass, so their AllReduce can start while earlier layers are still
  computing. This hides much of the communication cost.

!!! tip "The one-liner"
    **NCCL is the library. AllReduce is the operation. Ring-AllReduce is the algorithm.**

---

## 4. Model parallelism

When the model itself will not fit on one GPU, you must split the model rather than the data.
There are two ways to cut it, and knowing which is which is the most likely exam question in
this section.

### Tensor parallelism (intra-layer)

Split **individual weight matrices** across GPUs. Each GPU computes a *slice* of the same layer,
and the partial results are combined.

```text
        A single 4096 × 4096 weight matrix, split across 4 GPUs

        ┌──────┬──────┬──────┬──────┐
   x ──►│ GPU0 │ GPU1 │ GPU2 │ GPU3 │──► partial results
        └──────┴──────┴──────┴──────┘
                     │
              AllReduce / AllGather to combine
                     │
                     ▼
              full layer output
```

The critical property: **communication happens multiple times per layer**. Every split matrix
multiplication needs its partial results combined before the next operation. With 80 layers,
that is hundreds of collective operations per forward-backward pass.

Consequence: tensor parallelism **demands extremely fast interconnect** and is therefore
normally confined **within a single node**, where GPUs are connected by NVLink (hundreds of
GB/s) rather than by network (tens of GB/s). Pioneered by **Megatron-LM**.

### Pipeline parallelism (inter-layer)

Assign **different layers** to different GPUs. Activations flow from stage to stage.

```text
   GPU0          GPU1          GPU2          GPU3
 layers 1–20 ► layers 21–40 ► layers 41–60 ► layers 61–80
```

Communication is small — only the activations at stage boundaries — so this works **across
nodes** over ordinary networking.

Its weakness is the **pipeline bubble**: while GPU0 processes the first micro-batch, GPUs 1–3
have nothing to do. Then GPU1 works while GPU0 waits for the next batch and GPUs 2–3 still idle.

```text
NAIVE (one big batch):                  WITH MICRO-BATCHES:
GPU0  ████░░░░░░░░                      GPU0  ████████░░
GPU1  ░░░░████░░░░                      GPU1  ░░████████
GPU2  ░░░░░░░░████                      GPU2  ░░░░██████
GPU3  ░░░░░░░░░░░░████                  GPU3  ░░░░░░████
      huge bubbles — most GPUs idle           bubble shrinks substantially
```

The mitigation is to split each batch into **micro-batches** so stages are kept fed (GPipe, and
the more efficient 1F1B interleaved schedules).

### Side by side

| | **Tensor parallel** | **Pipeline parallel** |
| --- | --- | --- |
| Splits | **Within** a layer | **Across** layers |
| Communication | High frequency, high volume | Low frequency, at stage boundaries |
| Best placement | **Inside one node** (NVLink) | **Across nodes** |
| Main inefficiency | Communication overhead | **Pipeline bubbles** |
| Mitigation | Fast interconnect | Micro-batching |

**Expert parallelism** — for MoE models, place different experts on different GPUs and route
tokens to them.

**Sequence / context parallelism** — split along the sequence dimension for very long contexts,
where activations rather than weights dominate memory.

### 3D parallelism

Frontier models combine all three axes:

```text
      DATA parallel  ×  TENSOR parallel  ×  PIPELINE parallel

e.g. 512 GPUs = 8 data-parallel groups
                × 8 tensor-parallel (within a node, over NVLink)
                × 8 pipeline stages (across nodes)
```

The placement is deliberate: the chattiest dimension (tensor) goes on the fastest link.

---

## 5. Memory-optimization techniques

These reduce memory without changing the parallelism strategy, and they often remove the need
for model parallelism entirely.

### ZeRO / FSDP

Plain data parallelism is wasteful: every GPU stores an **identical** copy of the optimizer
states, gradients and parameters. With 8 GPUs you are storing 8 redundant copies.

**ZeRO** (DeepSpeed) and **FSDP** (PyTorch) shard those across the data-parallel ranks instead
of replicating them, in three progressive stages:

| Stage | Shards | Memory reduction |
| --- | --- | --- |
| **ZeRO-1** | Optimizer states | ~4× |
| **ZeRO-2** | + gradients | ~8× |
| **ZeRO-3** | + parameters | ~N× (linear in GPU count) |

At stage 3, parameters are **gathered just in time** for each layer's forward and backward pass
and released immediately after. You pay extra communication to buy enormous memory savings —
while keeping the programming simplicity of data parallelism.

### The others

**Activation checkpointing (gradient checkpointing).** Do not store every intermediate
activation for the backward pass; store a few checkpoints and **recompute** the rest on demand.
Costs roughly 30% extra compute, saves a large fraction of activation memory. Often the single
most effective knob when you are just short of fitting.

**Mixed precision (AMP).** FP16/BF16 compute with an FP32 master copy. ~2× faster, about half
the activation memory. See [Inference Optimization](inference-optimization.md#3-quantization).

**Gradient accumulation.** Run several micro-batches, accumulating gradients, before stepping
the optimizer. Simulates a large batch on limited memory — at the cost of proportionally more
time per step.

**CPU / NVMe offloading.** Park optimizer states off-GPU. Slow, but it turns "impossible" into
"overnight".

---

## 6. Scaling behaviour

NVIDIA lists *"Deep Learning Scaling Is Predictable, Empirically"*, and two practical concepts
follow.

**Strong scaling** — fix the total problem size, add GPUs, and see how much faster it goes.
Efficiency **falls** as GPU count rises, because the communication and synchronisation
overheads stay while the per-GPU work shrinks. Amdahl's law sets a hard ceiling: whatever
fraction of the work is inherently serial bounds your maximum speedup regardless of hardware.

**Weak scaling** — grow the problem size *with* the GPU count, keeping per-GPU work constant.
This scales far better, and it is how large-model training is actually organised: you do not
train the same model faster on more GPUs, you train a bigger model on more GPUs.

And the underlying reason any of this is worth doing: model loss improves as a **power law** in
compute, data and parameters, which makes a multimillion-dollar training run a projectable
investment rather than a gamble. See [LLMs & Foundation Models](../domain-1/llm-landscape.md).

---

## 7. Recap

- **Three reasons to distribute**: too slow (data parallel), does not fit (model parallel), both
  (3D). "Does not fit" is **never** answered by data parallelism, which replicates the model.
- **Data parallelism**: replicate the model, split the data, **AllReduce** the gradients.
- **Ring-AllReduce** makes per-GPU communication volume effectively independent of GPU count,
  removing the master bottleneck. **NCCL** is NVIDIA's implementation.
- **Tensor parallel** splits *within* a layer — chatty, keep it inside a node on NVLink.
  **Pipeline parallel** splits *across* layers — cheap communication, watch the **bubble**,
  mitigate with micro-batches.
- **ZeRO/FSDP** shards optimizer states (1), gradients (2) and parameters (3) instead of
  replicating them.
- **Activation checkpointing** trades ~30% compute for a large memory saving; **mixed
  precision** gives ~2× speed at half the activation memory; **gradient accumulation** simulates
  large batches.
- **Weak scaling** works far better than strong scaling, which is why more GPUs means bigger
  models rather than faster identical ones.
