---
title: Trustworthy AI
---

# Domain 5 — Trustworthy AI <span class="weight">10%</span>

> *Creation and assessment of ethical, energy-conscious, and reliable artificial
> intelligence systems capable of interpreting and integrating various forms of data,
> ensuring that they're designed and applied in a manner that's transparent, fair, and
> verifiable.*

Roughly **5–6 questions**, and the **easiest points on the exam**. There is very little to
compute and a small, closed set of concepts. Do not skip it because the weight is low —
this is where you bank marks cheaply.

## Task statements

| # | Statement | Covered in |
| --- | --- | --- |
| 5.1 | Describe the ethical principles of trustworthy AI | [Principles](principles.md) |
| 5.2 | Describe the balance between data privacy and the importance of data consent | [Privacy & Consent](privacy-consent.md) |
| 5.3 | Describe how to use NVIDIA and other technologies to improve AI trustworthiness | [Guardrails & Security](guardrails-security.md) |
| 5.4 | Describe how to minimize bias in AI systems | [Bias & Fairness](bias-fairness.md) |

## Chapters

<div class="grid cards" markdown>

- **[Principles of Trustworthy AI](principles.md)** — NVIDIA's pillars, transparency, explainability, accountability, sustainability, model cards.
- **[Bias & Fairness](bias-fairness.md)** — where bias enters, fairness definitions, mitigation at every stage.
- **[Privacy, Consent & Data Governance](privacy-consent.md)** — PII, consent, GDPR concepts, memorisation, anonymisation, federated learning.
- **[Guardrails & LLM Security](guardrails-security.md)** — NeMo Guardrails, prompt injection, jailbreaks, the OWASP LLM Top 10.
- **[Quiz](quiz.md)** — 12 exam-style questions.

</div>

## Read the source

The suggested reading list for this domain is only three items, all short and all NVIDIA's
own. Read them directly — the exam's phrasing comes from this language:

- [Trustworthy AI](https://www.nvidia.com/en-us/ai-data-science/trustworthy-ai/) — NVIDIA
- [What Is Trustworthy AI?](https://blogs.nvidia.com/blog/what-is-trustworthy-ai/) — NVIDIA Blog
- [What Is Retrieval-Augmented Generation, aka RAG?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) — NVIDIA Blog

!!! tip "Why RAG is in the *Trustworthy AI* reading list"
    Because grounding is a **trust** mechanism, not just a capability one. RAG makes
    answers **verifiable** (citations), **current**, and less prone to fabrication. If a
    Trustworthy AI question asks how to make an LLM's answers more reliable and auditable,
    **RAG with citations** is very often the intended answer.
