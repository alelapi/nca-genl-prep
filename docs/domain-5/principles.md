# Principles of Trustworthy AI

*Covers task 5.1: "describe the ethical principles of trustworthy AI."*

## NVIDIA's framing

NVIDIA describes Trustworthy AI around a small set of pillars. Learn these words — the
exam's answer options use them.

| Pillar | Meaning |
| --- | --- |
| **Privacy** | Comply with data protection regulation; protect personal data used to train and operate models |
| **Safety and security** | Systems must operate as intended, resist misuse and attack, and avoid causing harm |
| **Transparency** | Make model behaviour, capabilities and limitations understandable and auditable |
| **Non-discrimination** | Minimise unwanted bias; work toward equitable outcomes across groups |

NVIDIA also emphasises **energy efficiency and sustainability** — reflected in the
domain description's phrase *"energy-conscious"* — and **human accountability**: AI
supports human decisions; responsibility stays with people.

## The broader principle set

Most frameworks (EU, OECD, NIST AI RMF) converge on the same list. Recognise all of these:

| Principle | The question it answers |
| --- | --- |
| **Fairness / non-discrimination** | Does it treat people equitably? |
| **Transparency** | Can people tell what it is and how it works? |
| **Explainability / interpretability** | Can a specific decision be explained? |
| **Accountability** | Who is responsible when it goes wrong? |
| **Privacy** | Is personal data protected and lawfully used? |
| **Safety and robustness** | Is it reliable under stress and adversarial input? |
| **Human oversight / agency** | Can a human intervene, override and appeal? |
| **Beneficence** | Does it actually benefit people? |
| **Sustainability** | Is the environmental cost justified and minimised? |

!!! note "Transparency vs. explainability"
    **Transparency** is about the *system*: disclosing what data it was trained on, what it
    can and cannot do, that a user is talking to an AI at all.

    **Explainability** is about a *specific output*: why did the model produce this
    particular answer or decision?

    Questions often present both as options — read carefully for whether the stem is about
    the system or about one decision.

## Explainability for LLMs

Deep models are not intrinsically interpretable, so LLM explainability is largely
architectural rather than mathematical:

- **Citations and source attribution** — RAG's biggest trust contribution. The user can
  check the source themselves.
- **Chain-of-thought / reasoning traces** — with the caveat that stated reasoning is not
  guaranteed to be the model's actual computation.
- **Confidence signals** — token probabilities, self-consistency across samples.
- **Attention visualisation** — attributes output tokens to input tokens. Suggestive, not
  a causal explanation.
- **Feature-attribution methods** — SHAP and LIME, standard for classical ML and
  applicable to smaller models.
- **Model cards and system cards** — documented capabilities, limitations, training data
  and evaluation results.

!!! warning "The self-explanation trap"
    An LLM's stated reasoning is *generated text*, not an introspection log. It can be
    plausible and unrelated to how the answer was actually produced. Never present a
    model's self-explanation as an audit trail.

## Documentation artefacts

- **Model card** — intended use, out-of-scope uses, training data summary, evaluation
  results **broken down by subgroup**, known limitations and biases, ethical
  considerations. NVIDIA publishes **Model Cards++** for its models.
- **Datasheet for a dataset** — provenance, collection method, consent basis,
  composition, known biases, recommended and discouraged uses.
- **AI usage policy** — what the organisation permits, what needs review, what is
  forbidden.

## Human oversight

Three levels worth naming:

- **Human-in-the-loop** — a human approves each output before it takes effect. Required in
  high-stakes settings.
- **Human-on-the-loop** — the system acts, humans monitor and can intervene.
- **Human-in-command** — humans decide whether the system is deployed at all, and can
  switch it off.

The risk-tiering intuition (from the EU AI Act, and echoed by most frameworks): the higher
the potential harm, the more oversight, documentation and testing required. A marketing
copy generator and a medical triage assistant do not warrant the same controls.

## Sustainability

The domain description explicitly says *"energy-conscious"*, so this is fair game:

- Training a large model consumes substantial energy and water; inference at scale can
  exceed training's lifetime footprint.
- Levers: **efficient architectures**, **quantization**, **distillation**, **PEFT instead of
  full fine-tuning**, **reusing pretrained models rather than training from scratch**,
  right-sizing the model to the task, batching for utilisation, and running in
  low-carbon-intensity regions.
- Report energy and carbon alongside accuracy when it matters — several model cards now do.

!!! tip "The efficient answer is usually the trustworthy answer"
    Choosing a smaller model, using PEFT rather than full fine-tuning, and reusing a
    foundation model instead of pretraining are simultaneously the **cheapest**, the
    **fastest** and the **most sustainable** options. Exam answers rarely reward "train a
    bigger model from scratch".

## Key takeaways

- NVIDIA's pillars: **privacy, safety and security, transparency, non-discrimination** —
  plus energy consciousness and human accountability.
- **Transparency = the system; explainability = a specific decision.**
- LLM explainability comes mostly from **citations**, documentation and evaluation — not
  from the model explaining itself.
- **Model cards** and **dataset datasheets** are the standard documentation artefacts.
- Match oversight to risk: human-in-the-loop for high-stakes decisions.
- Efficiency (quantization, distillation, PEFT, right-sizing) is a sustainability practice.
