# Domain 2 Quiz — Software Development

18 exam-style questions. Target: **15/18**.

---

**1.** Which NVIDIA component optimizes a trained model for fast inference on a specific
GPU, and which one serves models in production?

- A. Triton optimizes; TensorRT serves
- B. TensorRT optimizes; Triton serves
- C. NeMo optimizes; NIM serves
- D. RAPIDS optimizes; Megatron serves

??? success "Answer"
    **B.** TensorRT (and TensorRT-LLM) is a compiler/runtime performing layer fusion,
    precision calibration and kernel auto-tuning to produce an optimized engine. Triton
    Inference Server hosts models and handles HTTP/gRPC, dynamic batching, versioning and
    metrics.

---

**2.** A 70B-parameter model must be served in FP16. Approximately how much GPU memory do
the weights alone require?

- A. ~35 GB
- B. ~70 GB
- C. ~140 GB
- D. ~280 GB

??? success "Answer"
    **C.** FP16 = 2 bytes per parameter → 70 × 2 = 140 GB, before KV cache and activations.
    The rule of thumb is **~2 GB per billion parameters in FP16**.

---

**3.** In LoRA, what happens to the original pretrained weight matrix `W`?

- A. It is replaced by the low-rank matrices
- B. It is frozen; only the low-rank matrices `A` and `B` are trained
- C. It is quantized to 4 bits and then trained
- D. It is retrained at a lower learning rate

??? success "Answer"
    **B.** LoRA freezes `W` and learns `ΔW = BA` with rank `r ≪ min(d,k)`. C describes
    **QLoRA**, where the frozen base is additionally quantized to 4-bit — but even there
    the base weights are not trained.

---

**4.** Which parallelism strategy splits individual weight matrices across GPUs and
therefore requires very high-bandwidth interconnect?

- A. Data parallelism
- B. Pipeline parallelism
- C. Tensor parallelism
- D. Expert parallelism

??? success "Answer"
    **C.** Tensor (intra-layer) parallelism communicates multiple times per layer, so it
    is normally confined within a node connected by NVLink. Pipeline parallelism
    communicates only at stage boundaries and tolerates inter-node links.

---

**5.** Why is Ring-AllReduce preferred over a naive gather-to-master gradient synchronisation?

- A. It requires no network communication
- B. Per-GPU communication volume is independent of the number of GPUs, avoiding a master bottleneck
- C. It eliminates the need for gradient computation
- D. It only works on a single GPU

??? success "Answer"
    **B.** Ring-AllReduce (ReduceScatter + AllGather around a ring) moves roughly
    `2(N−1)/N × data_size` bytes per GPU — effectively constant in *N* — while a central
    master's load grows linearly with GPU count.

---

**6.** LLM token generation is primarily bound by:

- A. Compute (FLOPS)
- B. Memory bandwidth
- C. Disk I/O
- D. CPU single-thread performance

??? success "Answer"
    **B.** The decode phase reads the entire weight set from HBM per token while doing
    comparatively little arithmetic. That is why quantization and batching are the two
    largest inference wins. (Prefill, by contrast, is compute-bound.)

---

**7.** A team quantizes a model to INT8 with PTQ and sees an unacceptable accuracy drop.
What should they try next?

- A. Increase the batch size
- B. Quantization-aware training (QAT)
- C. Switch from Triton to TensorRT
- D. Add more calibration data to INT4

??? success "Answer"
    **B.** QAT simulates quantization during training so the model learns weights robust
    to reduced precision, typically recovering most or all of the lost accuracy — exactly
    the topic of NVIDIA's suggested reading on FP32-accuracy INT8 inference.

---

**8.** What does in-flight (continuous) batching do that static batching does not?

- A. It compresses the model weights
- B. It inserts new requests into the running batch as soon as any sequence completes
- C. It removes the KV cache
- D. It guarantees deterministic output

??? success "Answer"
    **B.** Static batching wastes GPU cycles while finished sequences wait for the slowest
    in the batch. Continuous batching schedules at token granularity, giving large
    throughput gains under concurrency. Implemented in TensorRT-LLM and vLLM.

---

**9.** Which statement about the KV cache is correct?

- A. It stores the model's weights for faster loading
- B. It caches keys and values of previous tokens, avoiding O(n²) recomputation during decoding
- C. It is only used during training
- D. It has negligible memory footprint

??? success "Answer"
    **B.** It makes each decode step O(n) instead of O(n²). Its footprint scales with
    batch × sequence length × layers × KV heads and can exceed the weights at long
    context — the opposite of negligible.

---

**10.** A team has 5,000 labeled examples and wants the model to adopt their
domain-specific terminology and output format. They have one 24 GB GPU. Best approach?

- A. Pretrain from scratch
- B. Full fine-tuning in FP32
- C. LoRA/QLoRA fine-tuning
- D. RAG over the 5,000 examples

??? success "Answer"
    **C.** A behaviour/style problem with modest data and limited memory is exactly the
    PEFT case. Full fine-tuning (B) would need ~16 bytes/param of state. RAG (D) addresses
    knowledge gaps, not output style.

---

**11.** Speculative decoding improves latency by:

- A. Reducing the model's parameter count permanently
- B. Having a small draft model propose tokens that the large model verifies in one pass
- C. Skipping the softmax layer
- D. Serving stale cached responses

??? success "Answer"
    **B.** The large model verifies a proposed span in a single forward pass and accepts
    the correct prefix. Crucially, the output distribution is unchanged — it is a pure
    latency optimization.

---

**12.** Which memory-optimization technique shards optimizer states, gradients and
parameters across data-parallel ranks?

- A. Activation checkpointing
- B. ZeRO / FSDP
- C. Gradient accumulation
- D. Mixed precision

??? success "Answer"
    **B.** ZeRO (DeepSpeed) and FSDP (PyTorch) eliminate the redundant replication
    inherent to plain data parallelism, in three progressive stages. The others save
    memory by different mechanisms: recomputation (A), smaller micro-batches (C), and
    lower precision (D).

---

**13.** A new model version must be validated on real production traffic without any risk
to users. Which deployment pattern fits?

- A. Blue/green
- B. Canary
- C. Shadow (traffic mirroring)
- D. Rolling update

??? success "Answer"
    **C.** Shadow deployment sends a copy of live requests to the new model and discards
    its responses. Canary (B) does expose a small fraction of real users.

---

**14.** Which is a **silent** failure mode that standard service monitoring will not catch?

- A. HTTP 500 error rate spike
- B. GPU out-of-memory
- C. Data drift degrading answer quality
- D. Increased p99 latency

??? success "Answer"
    **C.** Drift produces no errors and no latency change — the system looks healthy while
    answers get worse. Detection requires comparing input/output distributions to a
    baseline and re-running an evaluation set on a schedule.

---

**15.** What is the primary purpose of ONNX?

- A. To serve models over gRPC
- B. To provide an open interchange format so models can move between frameworks and runtimes
- C. To quantize models to INT4
- D. To orchestrate multi-node training

??? success "Answer"
    **B.** ONNX (Open Neural Network Exchange) decouples the training framework from the
    inference runtime. NVIDIA's reading list names it in the context of transitioning AI
    models to deployment.

---

**16.** Approximately how much memory do parameters, gradients, FP32 master weights and
Adam states require per parameter during mixed-precision full fine-tuning?

- A. ~2 bytes
- B. ~4 bytes
- C. ~16 bytes
- D. ~64 bytes

??? success "Answer"
    **C.** 2 (FP16 params) + 2 (gradients) + 4 (FP32 master) + 8 (Adam m and v) ≈ 16
    bytes per parameter — plus activations. This is why full fine-tuning a 7B model needs
    well over 100 GB, and why PEFT dominates in practice.

---

**17.** Which NVIDIA offering packages an optimized model behind a standard,
OpenAI-compatible API in a container you can run in your own environment?

- A. NeMo Curator
- B. NIM
- C. cuGraph
- D. Megatron-LM

??? success "Answer"
    **B.** NVIDIA Inference Microservices (NIM) ship prebuilt, GPU-optimized model
    endpoints that run in your cloud, data centre or workstation, so data stays under your
    control.

---

**18.** A chat application must minimise **perceived** latency. Which change helps most?

- A. Increase the maximum batch size
- B. Stream tokens to the client as they are generated
- C. Add more retrieval results to the prompt
- D. Raise the temperature

??? success "Answer"
    **B.** Streaming lets the user start reading almost immediately; time-to-first-token
    dominates perceived responsiveness. Larger batches (A) actually raise per-request
    latency, and more context (C) increases prefill time.

---

## Scoring

| Score | Verdict |
| --- | --- |
| 16–18 | Ready. |
| 12–15 | Re-read Inference Optimization and Distributed Training. |
| < 12 | Rework the chapter; this is 24% of the exam. |
