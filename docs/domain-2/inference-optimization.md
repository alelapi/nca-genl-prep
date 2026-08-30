# Inference Optimization

*NVIDIA's suggested readings for this domain lead with **TensorRT**, **INT8 quantization
with QAT**, and **inference optimization**. This is prime exam territory.*

## The two metrics that matter

- **Latency** — time for one request. Split into **TTFT** (time to first token, dominated
  by the *prefill* pass over the prompt) and **TPOT/ITL** (time per output token, the
  *decode* phase).
- **Throughput** — tokens or requests per second across all users. What determines cost
  per token.

!!! important "They trade against each other"
    Larger batches raise throughput and *raise* per-request latency. A chatbot optimises
    latency; an overnight document-processing job optimises throughput. Scenario questions
    almost always hinge on which one the described workload cares about.

## Why LLM inference is slow

Generation is **memory-bandwidth bound**, not compute bound. For each new token the GPU
must read the entire model's weights from HBM, then do relatively little arithmetic with
them. This single fact explains why quantization (fewer bytes to read) and batching
(amortise the read across many sequences) are the two biggest wins.

**Prefill vs. decode:**

| Phase | What happens | Bound by |
| --- | --- | --- |
| **Prefill** | Process the whole prompt in parallel, build the KV cache | Compute |
| **Decode** | Generate one token at a time, autoregressively | Memory bandwidth |

## Quantization

Reduce the numeric precision of weights (and sometimes activations).

| Precision | Bits | Relative size | Notes |
| --- | --- | --- | --- |
| FP32 | 32 | 1× | Full precision; training baseline |
| **TF32** | 19-bit mantissa/exp in 32-bit container | 1× | NVIDIA Ampere+; automatic speedup for matmuls |
| **FP16** | 16 | 0.5× | Half precision; narrow range, can overflow |
| **BF16** | 16 | 0.5× | Same exponent range as FP32, less mantissa — **more robust than FP16 for training** |
| **FP8** | 8 | 0.25× | Hopper/Blackwell; increasingly standard for inference |
| **INT8** | 8 | 0.25× | Classic TensorRT target; needs calibration |
| **INT4 / NF4** | 4 | 0.125× | Aggressive; used by QLoRA and edge deployment |

**Benefits:** smaller memory footprint (fits bigger models on the same GPU), less memory
traffic (faster, because inference is bandwidth-bound), higher throughput, lower energy.

**Cost:** some accuracy loss, growing as precision drops.

- **PTQ (post-training quantization)** — quantize an already-trained model using a small
  **calibration dataset** to determine per-tensor/per-channel scaling ranges. Minutes of
  work.
- **QAT (quantization-aware training)** — simulate quantization during training so the
  model adapts. More expensive, recovers accuracy. See [Customization](customization.md#quantization-aware-training-qat).
- **Weight-only quantization** — quantize weights, keep activations in FP16. The common
  choice for LLMs since weights dominate memory traffic. Methods: GPTQ, AWQ.

!!! tip "Exam framing"
    *"Reduce memory footprint and increase inference speed with minimal accuracy loss"* →
    **quantization**. *"Recover the accuracy lost by INT8 quantization"* → **QAT**.

**Mixed precision** in *training* is a related but distinct idea: keep an FP32 master
copy of the weights, compute in FP16/BF16, and use **loss scaling** to stop small
gradients underflowing. Roughly 2× faster with ~half the memory, on Tensor Cores.

## KV caching

During autoregressive decoding, the keys and values of all previous tokens are needed at
every step. Recomputing them is O(n²); caching them makes each step O(n).

- **Always on** in any serious serving stack.
- The cache is **large**: it scales with `batch × sequence_length × layers × heads ×
  head_dim × 2 (K and V) × bytes_per_element`. At long context it can exceed the weights.
- **PagedAttention** (vLLM) manages the cache in fixed-size blocks like OS virtual
  memory, eliminating fragmentation and enabling much higher batch sizes.
- **MQA/GQA** shrink it structurally by sharing K/V across heads.
- **KV cache quantization** (FP8/INT8) shrinks it further.

## Batching

| Strategy | Behaviour |
| --- | --- |
| **Static batching** | Wait for *N* requests, run them together, wait for the slowest to finish. Simple, wasteful — finished sequences idle. |
| **Dynamic batching** | Group requests arriving within a time window. Better utilisation. |
| **In-flight / continuous batching** | Insert new requests into the batch as soon as any sequence finishes, at token granularity. **The state of the art**, and what TensorRT-LLM and vLLM implement. Large throughput gains at high concurrency. |

## Other techniques

- **Speculative decoding** — a small **draft** model proposes several tokens; the large
  model verifies them in one forward pass, accepting the correct prefix. Output is
  **identical in distribution** to the large model alone, at lower latency.
- **Kernel fusion** — merge several operations into one kernel to cut memory round-trips
  and launch overhead. What TensorRT does automatically.
- **FlashAttention** — IO-aware exact attention; avoids materialising the n×n matrix.
- **Graph capture (CUDA Graphs)** — replay a captured sequence of kernels to eliminate
  per-launch CPU overhead.
- **Tensor/pipeline parallelism at inference** — split a model too large for one GPU
  across several. See [Distributed Training](distributed-training.md).
- **Prompt/prefix caching** — reuse the KV cache of a shared prompt prefix across requests.
- **Model choice** — the largest single lever. A well-chosen 8B model can be 10× cheaper
  than a 70B for the same task quality.

## The NVIDIA inference stack

| Component | What it is |
| --- | --- |
| **TensorRT** | Compiler + runtime that optimizes a trained neural network for a specific GPU: layer/kernel fusion, precision calibration (FP16/INT8/FP8), kernel auto-tuning, memory reuse. Produces a serialised **engine**. |
| **TensorRT-LLM** | LLM-specific library on top of TensorRT: optimized attention kernels, **in-flight batching**, paged KV cache, tensor parallelism, quantization (FP8, INT8, INT4 AWQ/GPTQ). |
| **ONNX / ONNX Runtime** | An **open interchange format** for models, letting you export from PyTorch/TensorFlow and run on many runtimes. NVIDIA's reading list names ONNX explicitly. |
| **Triton Inference Server** | The serving layer — see [Deployment](deployment.md). |
| **NIM** | Prepackaged, containerised, optimized model endpoints with a standard API. |
| **CUDA / cuDNN / CUTLASS** | The underlying GPU compute libraries. |

!!! note "TensorRT vs. Triton — do not confuse them"
    **TensorRT optimizes a model.** **Triton serves models.** You typically compile with
    TensorRT-LLM and then deploy the engine on Triton. A question naming "optimize
    inference performance" wants TensorRT; one naming "serve multiple models with
    dynamic batching over HTTP/gRPC" wants Triton.

## An optimization checklist

1. Right-size the model (and consider distillation).
2. Quantize — PTQ first, QAT if accuracy suffers.
3. Compile with TensorRT-LLM.
4. Enable in-flight batching and paged KV cache.
5. Use GQA/MQA models, and quantize the KV cache at long context.
6. Cache prompt prefixes and repeated responses.
7. Stream tokens to the client.
8. Only then, scale horizontally.

## Key takeaways

- LLM decoding is **memory-bandwidth bound** — that is why quantization and batching win.
- Latency ↔ throughput is a real trade-off; batching improves throughput, hurts per-request latency.
- **KV cache** turns O(n²) decoding into O(n) and often dominates memory; PagedAttention,
  GQA and KV quantization manage it.
- **In-flight/continuous batching** is the modern serving default.
- **Speculative decoding** cuts latency without changing output quality.
- **TensorRT/TensorRT-LLM optimize; Triton serves; ONNX interchanges.**
