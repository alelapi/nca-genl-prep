# Principles of Trustworthy AI

*Covers task 5.1: "describe the ethical principles of trustworthy AI."*

Note the verb the blueprint uses throughout this domain: **describe**. You are not being asked to
implement a fairness algorithm. You are being asked to recognise the principles, use the right
vocabulary, and identify which principle a described situation engages. That makes this the
cheapest domain to prepare well.

---

## 1. Why this is an engineering topic, not a philosophy topic

It is tempting to treat trustworthy AI as a compliance box. The reason it belongs in a technical
certification is that **the failures are technical and specific**:

- A model that hallucinates confidently causes real harm when a user acts on it.
- A model trained on historical hiring data reproduces historical discrimination **at scale and
  with an appearance of objectivity**.
- A model that memorised its training data can be induced to leak someone's phone number.
- A RAG system without permission filtering will happily quote a document the user was never
  meant to see.

Each of those is a bug with a technical cause and a technical mitigation. The principles below
are a vocabulary for naming which bug you are looking at.

---

## 2. NVIDIA's framing

NVIDIA organises Trustworthy AI around four pillars. **Learn these words** — the exam's answer
options use them directly.

| Pillar | What it means in practice |
| --- | --- |
| **Privacy** | Comply with data protection regulation; protect the personal data used to train and operate models |
| **Safety and security** | Systems operate as intended, resist misuse and attack, and avoid causing harm |
| **Transparency** | Model behaviour, capabilities and limitations are made understandable and auditable |
| **Non-discrimination** | Minimise unwanted bias; work toward equitable outcomes across groups |

Two further themes NVIDIA emphasises:

**Energy efficiency and sustainability.** The domain description's own wording is
*"energy-conscious"*, which makes this examinable rather than decorative.

**Human accountability.** AI supports human decisions; responsibility remains with people. "The
model decided" is not an answer anyone accepts.

---

## 3. The broader principle set

Most frameworks — EU AI Act, OECD, NIST AI RMF — converge on the same list. Recognise all of
these by the question each one answers:

| Principle | The question it answers |
| --- | --- |
| **Fairness / non-discrimination** | Does it treat people equitably? |
| **Transparency** | Can people tell what it is and how it works? |
| **Explainability / interpretability** | Can a **specific decision** be explained? |
| **Accountability** | Who is responsible when it goes wrong? |
| **Privacy** | Is personal data protected and lawfully used? |
| **Safety and robustness** | Is it reliable under stress and adversarial input? |
| **Human oversight / agency** | Can a human intervene, override and appeal? |
| **Beneficence** | Does it actually benefit people? |
| **Sustainability** | Is the environmental cost justified and minimised? |

!!! important "Transparency vs. explainability — a distinction that gets tested"
    **Transparency** is about the **system**. Disclosing what data it was trained on, what it can
    and cannot do, its known limitations, and — crucially — that the user is talking to an AI at
    all.

    **Explainability** is about a **specific output**. Why did the model produce *this* answer,
    or reject *this* loan application?

    Questions often present both as options. Read carefully for whether the stem is asking about
    the system as a whole or about one individual decision.

---

## 4. Explainability for LLMs

Deep models are not intrinsically interpretable — there is no rule to read out. So LLM
explainability is largely **architectural**: you design the system so that its outputs can be
checked, rather than trying to open the model.

| Approach | What it gives you |
| --- | --- |
| **Citations and source attribution** | **RAG's biggest trust contribution.** The user can verify the claim against the source themselves |
| **Chain-of-thought / reasoning traces** | A plausible account of the reasoning — with a large caveat (below) |
| **Confidence signals** | Token probabilities, self-consistency across samples, retrieval scores |
| **Attention visualisation** | Attributes output tokens to input tokens. Suggestive, not causal |
| **Feature attribution (SHAP, LIME)** | Standard for classical ML and small models; expensive and unreliable for large LLMs |
| **Model cards / system cards** | Documented capabilities, limitations, training data and evaluation results |

!!! danger "The self-explanation trap"
    An LLM's stated reasoning is **generated text**, not an introspection log.

    The model produces a plausible-sounding explanation using the same next-token prediction that
    produced the answer. It may be accurate. It may also be a post-hoc rationalisation entirely
    unrelated to whatever computation actually produced the output — and there is research
    demonstrating exactly that, where models give explanations that do not reflect the factor
    actually driving their answers.

    **Never present a model's self-explanation as an audit trail.** If a question offers "ask the
    model to explain its reasoning and log that" as a way to make a system auditable, it is the
    trap answer. The real answer is **RAG with citations**, because a citation can be checked
    against an external source.

---

## 5. Documentation artefacts

**Model card** — the standard disclosure document for a model:

- Intended use, and explicitly **out-of-scope uses**
- Training data summary and provenance
- **Evaluation results disaggregated by subgroup** — not just aggregate accuracy
- Known limitations and biases
- Ethical considerations and recommendations

NVIDIA publishes **Model Cards++** for its own models.

**Datasheet for a dataset** — the equivalent for data: provenance, collection method, consent
basis, composition, known biases, recommended and discouraged uses.

**AI usage policy** — what the organisation permits, what requires review, what is forbidden.

That "disaggregated by subgroup" line is the one that does real work. A single accuracy number
can hide the fact that a model works well for one group and badly for another — see
[Bias & Fairness](bias-fairness.md).

---

## 6. Human oversight

Three levels, distinguished by where the human sits relative to the decision:

| Level | The human's role |
| --- | --- |
| **Human-in-the-loop** | Approves **each output** before it takes effect. Required for high-stakes decisions |
| **Human-on-the-loop** | The system acts; humans **monitor** and can intervene |
| **Human-in-command** | Humans decide **whether the system is deployed at all**, and can switch it off |

**Match the oversight to the risk.** This is the risk-tiering intuition from the EU AI Act, echoed
by most frameworks: the higher the potential harm, the more oversight, documentation and testing
required.

A marketing-copy generator and a medical triage assistant do not warrant the same controls, and
applying identical governance to both is a failure in either direction — either you have
strangled the low-risk use case, or you have under-protected the high-risk one.

---

## 7. Sustainability

The domain description explicitly says *"energy-conscious"*, so this is fair game.

**The costs are real.** Training a large model consumes substantial energy and water. And at
scale, **inference over a model's lifetime can exceed the cost of training it** — a model trained
once and served billions of times spends most of its energy budget after deployment.

**The levers**, most of which you already know from Domain 2:

- **Right-size the model.** A 7B model that does the job costs a fraction of a 70B one, forever.
- **Quantization** — fewer bytes read per token, less energy per token.
- **Distillation** — a small student replacing a large teacher.
- **PEFT instead of full fine-tuning** — orders of magnitude less compute.
- **Reuse pretrained models** rather than training from scratch.
- **Batching** for GPU utilisation — an idle GPU still draws power.
- **Run in low-carbon-intensity regions and at low-carbon times.**
- **Report** energy and carbon alongside accuracy where it matters.

!!! tip "The efficient answer is usually the trustworthy answer"
    Notice that the sustainability levers are the *same* choices as the cost and latency levers.
    Choosing a smaller model, using PEFT rather than full fine-tuning, quantizing, and reusing a
    foundation model rather than pretraining are simultaneously the **cheapest**, the **fastest**
    and the **most sustainable** options.

    Exam answers essentially never reward "train a bigger model from scratch" — and this is one
    more reason why.

---

## 8. Recap

- NVIDIA's pillars: **privacy, safety and security, transparency, non-discrimination** — plus
  **energy consciousness** and **human accountability**.
- **Transparency = the system** (what it is, its data, its limits). **Explainability = a specific
  decision.**
- LLM explainability comes from **citations**, documentation and evaluation — **not** from the
  model explaining itself, which is generated text rather than introspection.
- **Model cards** and **dataset datasheets** are the standard documentation artefacts, and
  subgroup-disaggregated evaluation is what makes them useful.
- Match oversight to risk: **human-in-the-loop** for high-stakes decisions, **on-the-loop** for
  monitoring, **in-command** for the deployment decision itself.
- Efficiency (right-sizing, quantization, distillation, PEFT) is a **sustainability** practice as
  well as a cost one.
