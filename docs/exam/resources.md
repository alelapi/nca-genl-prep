# Official Resources

Everything NVIDIA itself points at. The suggested-reading lists below are reproduced
from the official study guide because exam questions are demonstrably written from
this material — if a topic appears here, it is fair game.

## Primary sources

- [NCA-GENL certification page](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/)
- [Official Exam Study Guide (PDF)](https://dam-cdn.nvd.orangelogic.com/AssetLink/ik3amm6oy7871nv371600i4h7drvex5m.pdf)
- [Exam registration (Certiverse)](https://www.certiverse.com/#/checkout/nvidia/store-exam/NCA-GENL)
- [NVIDIA Deep Learning Institute (DLI)](https://www.nvidia.com/en-us/training/)

## Recommended NVIDIA training

| Course | Format | Length | Price |
| --- | --- | --- | --- |
| Getting Started With Deep Learning | Self-paced | 8 h | $90 |
| Fundamentals of Deep Learning | Instructor-led workshop | 8 h | $500 |
| Accelerating End-to-End Data Science Workflows | Self-paced | 8 h | $90 |
| Introduction to Transformer-Based Natural Language Processing | Self-paced | 6 h | $30 |
| Building LLM Applications With Prompt Engineering | Self-paced / workshop | 8 h | $90 / $500 |
| Rapid Application Development With Large Language Models | Self-paced / workshop | 8 h | $90 / $500 |

!!! tip "If you only buy one"
    **Introduction to Transformer-Based NLP** ($30, 6 h) is the cheapest course with the
    highest overlap against the heaviest domain. **Building LLM Applications With Prompt
    Engineering** is the next best value if you want a second.

    None of them are required. This course covers the same ground.

## Suggested reading — Core ML & AI (30%)

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the transformer paper
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [What Are Foundation Models?](https://blogs.nvidia.com/blog/what-are-foundation-models/) — NVIDIA Blog
- [Generative AI — What Is It and How Does It Work?](https://www.nvidia.com/en-us/glossary/generative-ai/)
- Autoregressive models; activation functions; training hidden units with backpropagation
- [Demystifying Diffusion-Based Models](https://developer.nvidia.com/blog/) — NVIDIA Technical Blog
- End-to-End AI for NVIDIA-Based PCs: transitioning AI models with **ONNX**
- Implementing deep learning methods and feature engineering for text data

## Suggested reading — Data Analysis (14%)

- [RAPIDS](https://rapids.ai/) — GPU-accelerated data science
- [cuML documentation](https://docs.rapids.ai/api/cuml/stable/)
- GPU-accelerated data science with RAPIDS
- Data exploration fundamentals
- Stemming and lemmatizing with scikit-learn vectorizers

## Suggested reading — Experimentation (22%)

- How to conduct A/B testing in machine learning
- Inference optimization
- Zero-shot testing
- [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/) — free online
- Machine translation methods
- Hallucinations in large language models
- [GLUE benchmark](https://gluebenchmark.com/) — General Language Understanding Evaluation
- Evaluating RAG applications
- Cross-validation in machine learning
- Benchmarking elementary language tasks

## Suggested reading — Software Development (24%)

- [TensorRT — Get Started](https://developer.nvidia.com/tensorrt)
- [NeMo Framework best practices](https://docs.nvidia.com/nemo-framework/user-guide/latest/)
- [Mastering LLM Techniques: Customization](https://developer.nvidia.com/blog/) — NVIDIA Technical Blog
- Achieving FP32 accuracy for INT8 inference using **quantization-aware training** with TensorRT
- [NCCL: accelerated multi-GPU collective communications](https://developer.nvidia.com/nccl)
- Technologies behind distributed deep learning: **AllReduce**
- Visual intuition on **Ring-AllReduce** for distributed deep learning
- [Big Data? Datasets to the Rescue!](https://huggingface.co/learn/nlp-course) — Hugging Face NLP Course
- Deep learning scaling is predictable, empirically

## Suggested reading — Trustworthy AI (10%)

- [Trustworthy AI](https://www.nvidia.com/en-us/ai-data-science/trustworthy-ai/) — NVIDIA
- [What Is Trustworthy AI?](https://blogs.nvidia.com/blog/what-is-trustworthy-ai/) — NVIDIA Blog
- [What Is Retrieval-Augmented Generation, aka RAG?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) — NVIDIA Blog

## Free extras worth your time

- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) — the single best free companion to Domain 1
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — if attention has not clicked yet
- [NVIDIA Technical Blog — Generative AI](https://developer.nvidia.com/blog/category/generative-ai/)
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) — read the README, it is exam-relevant for Domain 5
