# Study Plan

Two schedules. Pick the one that matches your starting point, then book the exam
**before** you start — a date on the calendar is the single most effective study tool.

## 6-week plan (~6–8 h/week)

Recommended if you are new to LLMs or have not trained a model recently.

=== "Week 1 — Foundations"

    - [ ] Read [Exam Guide](index.md) and [Blueprint](blueprint.md)
    - [ ] [ML Fundamentals](../domain-1/ml-fundamentals.md)
    - [ ] [Neural Networks & Deep Learning](../domain-1/neural-networks.md)
    - [ ] [Lab 0: Setup](../labs/00-setup.md) and [Lab 1: spaCy & NumPy](../labs/01-spacy-numpy.md)
    - [ ] Read *Attention Is All You Need* (skim; you will re-read it in week 2)

=== "Week 2 — Transformers & LLMs"

    - [ ] [Transformers & Attention](../domain-1/transformers.md)
    - [ ] [LLMs & Foundation Models](../domain-1/llm-landscape.md)
    - [ ] [Lab 2: Transformers with Hugging Face](../labs/02-transformers-hf.md)
    - [ ] Re-read *Attention Is All You Need*, this time following the diagram
    - [ ] NVIDIA DLI: *Introduction to Transformer-Based NLP* (6 h, $30) — optional but well matched

=== "Week 3 — Embeddings, RAG, Prompting"

    - [ ] [Embeddings & Vector Search](../domain-1/embeddings.md)
    - [ ] [RAG](../domain-1/rag.md)
    - [ ] [Prompt Engineering](../domain-1/prompt-engineering.md)
    - [ ] [Python NLP Stack](../domain-1/python-nlp-stack.md)
    - [ ] [Lab 3](../labs/03-embeddings-vector-search.md) and [Lab 4: RAG](../labs/04-rag.md)
    - [ ] [Domain 1 quiz](../domain-1/quiz.md) — target 85%

=== "Week 4 — Software Development"

    - [ ] [LLM Application Architecture](../domain-2/llm-app-architecture.md)
    - [ ] [Customization & PEFT](../domain-2/customization.md)
    - [ ] [Inference Optimization](../domain-2/inference-optimization.md)
    - [ ] [Deployment & Serving](../domain-2/deployment.md)
    - [ ] [Distributed Training](../domain-2/distributed-training.md) and [Hardware Sizing](../domain-2/hardware-sizing.md)
    - [ ] [Lab 6: LoRA](../labs/06-peft-lora.md)
    - [ ] [Domain 2 quiz](../domain-2/quiz.md)

=== "Week 5 — Experimentation & Data"

    - [ ] [Experiment Design](../domain-3/experiment-design.md)
    - [ ] [Evaluation Metrics](../domain-3/evaluation-metrics.md) — memorise the metric table
    - [ ] [Zero- & Few-Shot](../domain-3/zero-few-shot.md), [RAG Evaluation](../domain-3/rag-evaluation.md)
    - [ ] [RLHF & Alignment](../domain-3/rlhf-alignment.md), [Hallucinations](../domain-3/hallucinations.md)
    - [ ] All of [Domain 4](../domain-4/index.md)
    - [ ] [Lab 5: Evaluation](../labs/05-evaluation.md)
    - [ ] [Domain 3](../domain-3/quiz.md) and [Domain 4](../domain-4/quiz.md) quizzes

=== "Week 6 — Trust, review, mocks"

    - [ ] All of [Domain 5](../domain-5/index.md) — it is only 10% but it is the easiest 10% to bank
    - [ ] [Mock Exam 1](../practice/mock-exam-1.md) under timed conditions
    - [ ] Review every wrong answer back to its source page
    - [ ] [Mock Exam 2](../practice/mock-exam-2.md)
    - [ ] [Flashcards](../practice/flashcards.md) on the two days before
    - [ ] [Exam-day strategy](exam-day.md), then sit the exam

## 2-week sprint (~15 h/week)

For practitioners who already build with LLMs and mainly need to map their knowledge
onto NVIDIA's vocabulary.

| Day | Focus |
| --- | --- |
| 1 | Blueprint + [Mock Exam 1](../practice/mock-exam-1.md) **cold**, to find your gaps |
| 2–3 | Domain 1 (heaviest — 30%) |
| 4–5 | Domain 2, focus on TensorRT/Triton/NeMo/NIM vocabulary and quantization |
| 6–7 | Domain 3, focus on the metrics and benchmark table |
| 8 | Domain 4 — especially RAPIDS (cuDF / cuML / cuGraph) |
| 9 | Domain 5 — read NVIDIA's Trustworthy AI pages directly |
| 10 | [Labs](../labs/index.md) 3, 4 and 5 back to back |
| 11 | [Mock Exam 2](../practice/mock-exam-2.md) + full review |
| 12 | Re-do every question you have ever missed |
| 13 | [Flashcards](../practice/flashcards.md) + glossary |
| 14 | Exam |

## How to study, not just what

!!! tip "Active recall beats re-reading"
    After each page, close it and write down — from memory — the three things you would
    put on a cheat sheet. Only then check. Re-reading feels productive and is the
    weakest form of study; retrieval is what builds durable memory.

!!! tip "Explain the distractors"
    On every practice question, do not stop at "C is right". Say out loud *why A, B and
    D are wrong*. NCA-GENL questions are built from plausible near-misses, and this
    habit is what converts a 65% into an 85%.

!!! tip "Weight your time by domain weight"
    30% of the exam is Domain 1. If you have six hours left, roughly two of them belong
    to Core ML & AI, and half an hour — no more — to Trustworthy AI.
