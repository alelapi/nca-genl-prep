# NCA-GENL Preparation Course

A complete, self-paced course for the **NVIDIA-Certified Associate: Generative AI LLMs**
exam (exam code **NCA-GENL**).

!!! info "Exam at a glance"
    | | |
    | --- | --- |
    | **Exam code** | NCA-GENL |
    | **Level** | Associate (entry-level) |
    | **Questions** | 50–60 multiple choice |
    | **Duration** | 1 hour |
    | **Price** | $125 USD |
    | **Delivery** | Online, remotely proctored (Certiverse) |
    | **Prerequisite** | Basic understanding of generative AI and LLMs |
    | **Validity** | 2 years from issuance |

    Source: [NVIDIA certification page](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/)

## The five domains

Every page in this course maps to a numbered task statement from NVIDIA's official
study guide. Study time should roughly follow the weights.

| # | Domain | Weight | Approx. questions |
| --- | --- | --- | --- |
| 1 | [Core Machine Learning and AI Knowledge](domain-1/index.md) | **30%** | 15–18 |
| 2 | [Software Development](domain-2/index.md) | **24%** | 12–14 |
| 3 | [Experimentation](domain-3/index.md) | **22%** | 11–13 |
| 4 | [Data Analysis](domain-4/index.md) | **14%** | 7–8 |
| 5 | [Trustworthy AI](domain-5/index.md) | **10%** | 5–6 |

## How to use this course

1. **Read [the blueprint](exam/blueprint.md) first.** It is the literal contract of
   what can be asked. Everything else here exists to serve it.
2. **Follow the [6-week study plan](exam/study-plan.md)**, or compress it to 2–3 weeks
   if you already work with LLMs daily.
3. **Work the domain chapters in order.** Each chapter ends with a quiz that mirrors
   the exam's question style.
4. **Do the [labs](labs/index.md).** The exam is conceptual, but the questions are
   written from a practitioner's point of view — code you have actually run is much
   harder to forget.
5. **Take the [mock exams](practice/index.md)** under real conditions: 60 questions,
   60 minutes, no notes. Aim for a consistent 80%+ before booking.

!!! tip "What the exam actually tests"
    NCA-GENL is *not* a coding exam and there is no lab component. It tests whether
    you recognise the right concept, tool or trade-off in a described scenario:
    *"which technique reduces hallucination when the model lacks domain knowledge?"*,
    *"which parallelism strategy applies when a model does not fit on one GPU?"*.
    Breadth beats depth — but knowing *why* an answer is right eliminates
    the two plausible distractors that every question contains.

## What is (and is not) in scope

<div class="grid cards" markdown>

- :material-check-circle: **In scope**

    Transformers and attention, embeddings, RAG, prompt engineering, fine-tuning and
    PEFT/LoRA, evaluation metrics and benchmarks, A/B testing, RLHF, data
    preprocessing, Python NLP/DS libraries, inference optimization, GPU deployment
    (Triton, TensorRT, NeMo, RAPIDS), trustworthy-AI principles.

- :material-close-circle: **Out of scope**

    Writing CUDA kernels, deriving backpropagation by hand, DGX/cluster
    administration, MLOps platform certification, anything requiring you to produce
    working code during the exam.

</div>

---

!!! warning "Disclaimer"
    Unofficial study material, not affiliated with or endorsed by NVIDIA. Exam
    objectives are paraphrased from NVIDIA's publicly published study guide. All
    practice questions in this course are original and are **not** real exam
    questions.
