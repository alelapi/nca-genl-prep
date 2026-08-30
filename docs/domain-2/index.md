---
title: Software Development
---

# Domain 2 — Software Development <span class="weight">24%</span>

> *Create, maintain, and test software.*

Roughly **12–14 questions**. The task statements sound generic ("write software
components under supervision"), but the **suggested reading list gives the real
syllabus**: TensorRT, NeMo best practices, LLM customization, INT8 quantization and QAT,
NCCL and Ring-AllReduce, distributed deep learning, scaling laws, and BERT pretraining.

This is the most **NVIDIA-product-flavoured** domain on the exam.

## Chapters

<div class="grid cards" markdown>

- **[LLM Application Architecture](llm-app-architecture.md)**

    Building chatbots, summarizers and RAG services; components, state, structured
    output, testing. *(Tasks 4.2, 4.7)*

- **[Model Customization & PEFT](customization.md)**

    The customization ladder: prompting → RAG → PEFT/LoRA → full fine-tuning →
    pretraining. Distillation, quantization-aware training. *(Suggested reading: "Mastering LLM Techniques: Customization")*

- **[Inference Optimization](inference-optimization.md)**

    Quantization, KV cache, batching, TensorRT-LLM, ONNX, speculative decoding.
    *(Suggested reading: TensorRT, INT8 QAT, inference optimization)*

- **[Deployment & Serving](deployment.md)**

    Triton, NIM, NeMo, autoscaling, observability, monitoring. *(Tasks 4.1, 4.5)*

- **[Distributed Training](distributed-training.md)**

    Data/tensor/pipeline parallelism, NCCL, Ring-AllReduce, ZeRO, mixed precision.
    *(Suggested reading: NCCL, AllReduce, distributed deep learning)*

- **[Hardware & Memory Sizing](hardware-sizing.md)**

    How much GPU memory a model needs, for training and for inference. *(Task 4.4)*

- **[Quiz](quiz.md)**

    18 exam-style questions.

</div>

## The NVIDIA stack in one diagram

```text
   DATA              BUILD / CUSTOMIZE          OPTIMIZE            SERVE
 ─────────          ──────────────────       ─────────────      ─────────────
 NeMo Curator  ──►  NeMo Framework      ──►  TensorRT-LLM  ──►  Triton Server
 (curate,           (pretrain, SFT,          TensorRT           NIM microservices
  dedupe)            PEFT, align,            ONNX Runtime       (packaged models,
                     evaluate, export)       quantization        OpenAI-style API)
                          │                                            │
 RAPIDS  ────────────────►│                                            │
 (cuDF/cuML/cuGraph)      │                                    NeMo Guardrails
                     Megatron-LM                               (safety rails)
                  (large-scale training)
                                                               NeMo Retriever
                                                               (embed + rerank
                                                                for RAG)
```

If a question names an NVIDIA product, the answer is almost always a matter of putting
it in the right box above.
