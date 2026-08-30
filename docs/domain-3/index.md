---
title: Experimentation
---

# Domain 3 — Experimentation <span class="weight">22%</span>

> *The study of how to perform, evaluate, and interpret experiments, including AI model
> evaluation and the use of human subjects in labeling or reinforcement learning from
> human feedback (RLHF).*

Roughly **11–13 questions**.

!!! warning "Read this before you study the domain"
    In the published study guide, the task statements printed under *Experimentation*
    (3.1–3.5) are **byte-identical to the Data Analysis statements** (2.1–2.5) — an
    apparent copy-paste error in NVIDIA's PDF.

    The real scope is unmistakable from the domain **description** ("evaluate and
    interpret experiments… RLHF") and from its **suggested reading list**, which is
    entirely distinct from Data Analysis's: A/B testing, cross-validation, zero-shot
    testing, GLUE, machine translation metrics, hallucinations, evaluating RAG
    applications, inference optimization and benchmarking.

    **Study the reading list, not the duplicated bullets.**

## Chapters

<div class="grid cards" markdown>

- **[Experiment Design & A/B Testing](experiment-design.md)**

    Hypotheses, controls, randomisation, sample size, significance, offline vs. online
    evaluation, cross-validation.

- **[Evaluation Metrics & Benchmarks](evaluation-metrics.md)**

    Perplexity, BLEU, ROUGE, METEOR, BERTScore, accuracy/precision/recall/F1, GLUE,
    SuperGLUE, MMLU, HELM, LLM-as-a-judge.

- **[Zero-, One- and Few-Shot Testing](zero-few-shot.md)**

    What each means, how to evaluate them fairly, and where they break down.

- **[Evaluating RAG Systems](rag-evaluation.md)**

    Retrieval metrics (recall@k, MRR, nDCG) vs. generation metrics (faithfulness,
    answer relevance, context precision); the RAG triad.

- **[Alignment, RLHF & Human Labeling](rlhf-alignment.md)**

    Reward models, PPO, DPO, constitutional AI, annotation quality, inter-annotator
    agreement.

- **[Hallucinations](hallucinations.md)**

    Why they happen, the taxonomy, how to detect and reduce them.

- **[Quiz](quiz.md)**

    18 exam-style questions.

</div>

## The mental model

Evaluation happens at three levels, and questions often turn on which level is being
described:

| Level | Question | Example |
| --- | --- | --- |
| **Component** | Is this piece working? | Recall@10 of the retriever; F1 of the classifier |
| **System** | Is the whole pipeline producing good answers? | Faithfulness, answer relevance, human ratings |
| **Product** | Does it improve outcomes for users? | Task completion, deflection rate, CSAT — measured by **A/B test** |

Good offline metrics that do not move product metrics are a warning sign, not a success.
