# Inference Optimization

*NVIDIA's suggested readings for this domain lead with **TensorRT**, **INT8 quantization with
QAT**, and **inference optimization**. This is prime exam territory, and it is also where an
NVIDIA certification is most likely to test NVIDIA-specific vocabulary.*

---

## 1. The two metrics, and why they conflict

Everything in this page trades between two numbers.

**Latency** — how long one request takes. For LLMs it splits into two components that behave
completely differently:

- **TTFT (time to first token)** — how long until the user sees anything. Dominated by
  *prefill*: processing the entire prompt.
- **TPOT / ITL (time per output token)** — how fast tokens then stream out. Dominated by
  *decode*.

**Throughput** — total tokens or requests per second across all users. This is what determines
your cost per token, because a GPU costs the same per hour whether it is busy or idle.

!!! important "They trade against each other, and this drives many exam answers"
    Batching more requests together raises **throughput** (the GPU does more useful work per
    weight-read) and raises **per-request latency** (your request waits for others).

    - An **interactive chatbot** optimises latency: small batches, streaming, low TTFT.
    - An **overnight document-processing job** optimises throughput: maximum batch size, nobody
      is watching.

    Scenario questions almost always hinge on identifying which one the described workload
    actually cares about.

---

## 2. Why LLM inference is slow — the fact that explains everything else

This is the single most useful idea on the page, and it is counterintuitive.

**LLM token generation is memory-bandwidth bound, not compute bound.**

Here is why. To generate one token, the GPU must read **every weight in the model** from
high-bandwidth memory (HBM) into the compute cores. For a 7B model in FP16 that is 14 GB of
data movement. And the arithmetic it then performs with those weights is comparatively tiny —
each weight is used in roughly one multiply-accumulate.

```text
Generating ONE token from a 7B FP16 model:

  data read from HBM:     14 GB
  arithmetic performed:   ~14 GFLOPs

  An H100 delivers ~3,300 GB/s of memory bandwidth and ~1,000 TFLOPS of FP16 compute.

  time bound by memory:   14 GB / 3,300 GB/s   ≈  4.2 ms
  time bound by compute:  14 GFLOP / 1e15 FLOPS ≈  0.014 ms

  → memory is ~300× the bottleneck. The compute units are idle almost the whole time.
```

Two enormous consequences follow, and they explain most of the optimizations in this page:

**Quantization is a speed optimization, not just a memory one.** Halving the precision halves
the bytes that must be read, which roughly doubles generation speed. That is why INT8 and FP8
matter so much more for LLMs than they did for older models.

**Batching is nearly free.** If you read the weights once and use them for 32 sequences instead
of 1, you get roughly 32× the throughput for approximately the same memory traffic. The compute
units were idle anyway. This is why continuous batching produces such large gains.

### Prefill vs. decode

| Phase | What happens | Bound by |
| --- | --- | --- |
| **Prefill** | The whole prompt is processed in one parallel pass; the KV cache is built | **Compute** — thousands of tokens at once genuinely saturates the cores |
| **Decode** | One token generated at a time, autoregressively | **Memory bandwidth** — as above |

This asymmetry is why a long prompt costs you TTFT while a long *output* costs you total
generation time, and why the two are optimised differently.

---

## 3. Quantization

Reduce the numeric precision used to store weights (and sometimes activations).

| Precision | Bits | Relative size | Notes |
| --- | --- | --- | --- |
| FP32 | 32 | 1× | Full precision; the training baseline |
| **TF32** | 32-bit container, 10-bit mantissa | 1× | NVIDIA Ampere+; automatic matmul speedup with no code change |
| **FP16** | 16 | 0.5× | Half precision. Narrow exponent range — can overflow during training |
| **BF16** | 16 | 0.5× | Same exponent range as FP32, fewer mantissa bits. **More robust for training** |
| **FP8** | 8 | 0.25× | Hopper/Blackwell; increasingly the inference standard |
| **INT8** | 8 | 0.25× | Classic TensorRT target; requires calibration |
| **INT4 / NF4** | 4 | 0.125× | Aggressive; used by QLoRA and edge deployment |

### FP16 vs. BF16 — a distinction worth understanding

Both are 16 bits, but they allocate those bits differently:

```text
FP16:  1 sign │ 5 exponent  │ 10 mantissa      range ≈ ±65,504
BF16:  1 sign │ 8 exponent  │  7 mantissa      range ≈ ±3.4e38  (same as FP32)
```

BF16 sacrifices precision for **range**. In training, gradients can be extremely small and
activations extremely large; FP16 overflows or underflows and produces `NaN`, requiring
careful loss scaling to avoid. BF16 has FP32's dynamic range and simply does not have that
problem, which is why modern training defaults to it.

### What quantization buys and costs

**Buys:** a smaller memory footprint (bigger models fit on the same GPU), less memory traffic
(**faster**, per section 2), higher throughput, lower energy.

**Costs:** some accuracy loss, growing as precision drops. INT8 is usually near-lossless with
care; INT4 has a real, measurable quality cost.

**The methods:**

- **PTQ (post-training quantization)** — quantize an already-trained model. A small
  **calibration dataset** (a few hundred representative samples) is run through the model to
  measure the actual range of each tensor, so scale factors can be chosen sensibly. Minutes of
  work.
- **QAT (quantization-aware training)** — simulate quantization during training so the model
  learns robust weights. Expensive, recovers accuracy. See
  [Customization](customization.md#5-quantization-aware-training).
- **Weight-only quantization** — quantize the weights, keep activations in FP16. Since weights
  dominate memory traffic during decode, this captures most of the benefit with less accuracy
  risk. **GPTQ** and **AWQ** are the standard methods, and this is the common choice for LLMs.

### Mixed precision (training, not inference)

A related idea that is easy to confuse with quantization. In mixed-precision **training**:

- Keep an **FP32 master copy** of the weights.
- Perform the forward and backward passes in **FP16/BF16** on Tensor Cores.
- Use **loss scaling** (multiply the loss by a large constant before backprop, divide after) to
  stop small gradients underflowing to zero in FP16. Not needed with BF16.

Roughly 2× faster with about half the activation memory.

---

## 4. The KV cache

### The problem

During autoregressive generation, producing token 101 requires attending over the keys and
values of tokens 1–100. Those were already computed when producing token 100. Recomputing them
every step makes generating an `n`-token sequence O(n³) in total.

### The fix

**Cache the K and V tensors.** Each new token computes only its own `q`, `k`, `v`, appends its
`k` and `v` to the cache, and attends over the cached history. Generation drops to O(n) per
step.

This is not optional in any serious serving stack. But it is **expensive in memory**, and the
arithmetic is worth being able to do:

$$ \text{KV bytes} = 2 \times \text{layers} \times \text{seq len} \times \text{kv heads} \times \text{head dim} \times \text{batch} \times \text{bytes} $$

Worked, for Llama-2-7B in FP16 (32 layers, 32 heads, head dim 128):

```text
per token, per layer:   2 × 32 × 128 × 2 bytes  =  16 KB
per token, all layers:  16 KB × 32              = 512 KB
4,096-token context:    512 KB × 4,096          ≈   2 GB   for ONE sequence

model weights:                                    14 GB
16 concurrent users at 4k context:                32 GB   ← more than the weights
```

!!! important "The practical consequence"
    At production concurrency and long context, **the KV cache is usually what limits how many
    users a GPU can serve — not the model weights.** Every technique below exists because of
    this table.

### Managing it

- **GQA / MQA** — heads share K/V projections. Rerun the calculation with 8 KV heads instead of
  32 and the cache drops from 2 GB to **512 MB**. Structural, free at inference, and standard in
  modern models.
- **PagedAttention** — manage the cache in fixed-size blocks like OS virtual memory. Eliminates
  fragmentation (previously you had to pre-allocate for the maximum possible length) and enables
  much larger batches. The central idea in vLLM.
- **KV cache quantization** — store the cache in FP8 or INT8, halving or quartering it again.
- **Cache eviction / sliding window** — drop distant tokens for very long conversations.

---

## 5. Batching

| Strategy | Behaviour | Problem |
| --- | --- | --- |
| **Static** | Wait for *N* requests, run them together, return when all finish | Sequences that finish early leave their slots **idle** until the longest one completes |
| **Dynamic** | Group requests arriving within a short time window | Better utilisation; still suffers the same tail problem |
| **In-flight / continuous** | Schedule at **token granularity** — insert a new request as soon as any sequence finishes | **The state of the art** |

The waste in static batching is worth seeing:

```text
STATIC BATCHING                         CONTINUOUS BATCHING
───────────────                         ───────────────────
seq A ████░░░░░░░░  (done, idle)        seq A ████
seq B ████████████                      seq E     ████████
seq C ██████░░░░░░  (done, idle)        seq B ████████████
seq D ███░░░░░░░░░  (done, idle)        seq C ██████
                                        seq F       ██████████
      ───► time                         seq D ███
                                        seq G    ███████
  ~50% of slot-time wasted                    ───► time

                                        slots refilled the moment they free
```

Continuous batching is implemented by **TensorRT-LLM** and **vLLM**, and it typically delivers
several times the throughput of static batching at high concurrency.

---

## 6. The other techniques

**Speculative decoding.** A small, fast **draft** model proposes the next several tokens; the
large model verifies them all in a **single forward pass** and accepts the longest correct
prefix.

```text
draft model (1B) proposes:   "the"  "cat"  "sat"  "on"  "the"
large model (70B) verifies all five in ONE pass
                              ✓      ✓      ✓      ✗
accepted: "the cat sat", then generate "in" from the large model
```

The economics work precisely because of section 2: verifying five tokens costs almost the same
as generating one, since both are dominated by reading the weights once. Crucially, the output
distribution is **mathematically identical** to what the large model alone would produce — it is
a pure latency win with no quality cost. That property is what gets tested.

**Kernel fusion.** Merge several operations into one GPU kernel so intermediate results stay in
registers instead of round-tripping to memory. Fewer kernel launches, far less memory traffic.
This is a large part of what TensorRT does automatically.

**FlashAttention.** Computes exactly the same attention, but tiles the computation so the `n×n`
matrix is never materialised in HBM. Same math, dramatically less memory traffic, and memory
that scales linearly rather than quadratically in sequence length.

**CUDA Graphs.** Capture a sequence of kernel launches once and replay it, eliminating per-launch
CPU overhead. Matters when kernels are small and numerous, which is exactly the decode phase.

**Prompt / prefix caching.** Reuse the KV cache for a shared prompt prefix — a long system
prompt or few-shot block that is identical across requests. Pure win, no behaviour change.

**Model choice.** The largest single lever, and the one most often overlooked. A well-chosen 8B
model with good RAG can be 10× cheaper than a 70B model at equivalent task quality. Before
optimising, ask whether you need the model you are optimising.

---

## 7. The NVIDIA inference stack

| Component | What it is |
| --- | --- |
| **TensorRT** | A **compiler and runtime** that optimizes a trained network for a specific GPU: layer and tensor fusion, precision calibration (FP16/INT8/FP8), kernel auto-tuning, memory reuse. Produces a serialised **engine** |
| **TensorRT-LLM** | The LLM-specific library on top: optimized attention kernels, **in-flight batching**, paged KV cache, tensor parallelism, quantization (FP8, INT8, INT4 AWQ/GPTQ) |
| **ONNX / ONNX Runtime** | An **open interchange format** — export from PyTorch or TensorFlow, run on many runtimes. Named explicitly in NVIDIA's reading list |
| **Triton Inference Server** | The **serving** layer. See [Deployment](deployment.md) |
| **NIM** | Prepackaged, containerised, optimized model endpoints with a standard API |
| **CUDA / cuDNN / CUTLASS** | The underlying GPU compute libraries |

### What "compiling" a model actually means

TensorRT takes a trained network and applies transformations that preserve the mathematics while
changing how it executes:

- **Layer fusion** — a convolution followed by a bias add followed by a ReLU becomes one kernel
  instead of three, avoiding two round-trips to memory.
- **Precision calibration** — determine per-tensor scale factors so INT8 can be used without
  unacceptable accuracy loss.
- **Kernel auto-tuning** — benchmark several implementations of each operation *on your actual
  GPU* and pick the fastest. This is why an engine is tied to a specific GPU architecture.
- **Memory optimisation** — reuse buffers across layers whose lifetimes do not overlap.

The output is a **serialised engine**, which is why TensorRT engines are not portable across GPU
generations and must be rebuilt.

!!! danger "TensorRT vs. Triton — do not confuse them"
    **TensorRT optimizes a model. Triton serves models.**

    They are complementary steps in one workflow: you compile with TensorRT-LLM, then deploy the
    resulting engine on Triton. A question about "optimizing inference performance" wants
    TensorRT; one about "serving multiple models over HTTP/gRPC with dynamic batching and
    versioning" wants Triton.

    This pair appears as a swapped-answer distractor constantly.

---

## 8. An optimization checklist

The order matters — each step is cheaper than the one after it.

1. **Right-size the model.** Consider distillation, or a smaller model with better RAG.
2. **Quantize.** PTQ first; QAT if accuracy suffers.
3. **Compile** with TensorRT-LLM.
4. **Enable continuous batching and paged KV cache.**
5. **Use a GQA/MQA model**, and quantize the KV cache at long context.
6. **Cache prompt prefixes** and repeated responses.
7. **Stream tokens** to the client — this improves *perceived* latency more than anything else
   on this list.
8. **Only then scale horizontally.** Adding GPUs to an unoptimised deployment multiplies an
   avoidable cost.

---

## 9. Recap

- **Latency vs. throughput** is the central trade-off; batching improves one and hurts the other.
  Identify which the workload cares about.
- **LLM decoding is memory-bandwidth bound** — every weight is read per token. This is why
  quantization speeds things up and why batching is nearly free.
- **Prefill is compute-bound; decode is bandwidth-bound.**
- **BF16** has FP32's exponent range and is more robust for training than FP16.
- The **KV cache** turns O(n²) decoding into O(n) but often exceeds the weights in memory at
  production concurrency — hence GQA/MQA, PagedAttention and cache quantization.
- **Continuous (in-flight) batching** refills slots at token granularity and is the modern
  serving default.
- **Speculative decoding** cuts latency with a **mathematically identical output distribution**.
- **TensorRT/TensorRT-LLM optimize. Triton serves. ONNX interchanges.**
