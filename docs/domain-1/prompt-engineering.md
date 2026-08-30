# Prompt Engineering

*Covers task 1.9. NVIDIA sells an entire course on this
(*Building LLM Applications With Prompt Engineering*), and the job description lists
"perform prompt engineering" as a core responsibility — expect several questions.*

## Definition

**Prompt engineering** is the practice of designing model inputs to reliably elicit the
desired output — **without changing model weights**. It is the cheapest, fastest
adaptation method and should always be attempted before RAG or fine-tuning.

## Anatomy of a good prompt

| Component | Purpose | Example |
| --- | --- | --- |
| **Role / persona** | Sets voice and expertise | "You are a senior Kubernetes engineer." |
| **Task / instruction** | The actual ask, stated imperatively | "Summarise the incident report below." |
| **Context** | Data the model needs (RAG chunks, documents) | `<context>…</context>` |
| **Examples** | Demonstrations of input→output | Few-shot pairs |
| **Format specification** | The exact output shape | "Return JSON with keys `cause`, `impact`, `fix`." |
| **Constraints** | Boundaries | "Max 100 words. Use only the context. If unknown, say so." |

**System vs. user prompt** — the *system* prompt sets persistent behaviour, identity and
rules for the whole conversation; the *user* prompt carries the specific request. Put
durable policy in the system prompt, per-request content in the user message.

## Shot-based prompting

| Technique | Prompt contains | When to use |
| --- | --- | --- |
| **Zero-shot** | Instruction only | Simple, common tasks; keeps prompts short and cheap |
| **One-shot** | Instruction + 1 example | Output format needs demonstrating |
| **Few-shot** | Instruction + 2–10 examples | Nuanced tasks, custom labels, strict formats |

Few-shot works through **in-context learning** — an emergent capability where the model
infers the pattern from the examples with **no gradient updates**. It is not training.

!!! tip "Making few-shot work"
    - Keep the format of every example **byte-identical**; the model imitates structure.
    - **Cover the classes**, including edge cases and a negative example.
    - Watch for **recency and majority-label bias** — order and class balance measurably
      change predictions. Shuffle and test.
    - More examples cost tokens and latency; find the smallest set that works.

## Reasoning techniques

**Chain-of-thought (CoT)** — ask the model to reason step by step before answering.
Substantially improves multi-step arithmetic, logic and planning. Zero-shot CoT is
literally the phrase *"Let's think step by step."* Few-shot CoT includes worked
reasoning in the examples.

**Self-consistency** — sample several CoT paths at non-zero temperature and take the
majority answer. More reliable, proportionally more expensive.

**ReAct (Reason + Act)** — interleave reasoning with tool calls:
`Thought → Action → Observation → Thought …`. The basis of most agent frameworks.

**Tree of Thoughts** — explore and evaluate multiple reasoning branches.

**Prompt chaining** — decompose one hard prompt into a pipeline of simple ones, each
consuming the previous output. Easier to debug and to evaluate stage by stage.

!!! note "When CoT is *not* the answer"
    For simple classification or extraction, chain-of-thought adds latency and cost for
    no gain — and reasoning-tuned models do it internally already. Scenario questions
    often hinge on matching technique to task complexity.

## Other named techniques

- **Output priming** — end the prompt with the beginning of the desired output
  (`{"cause":`) to force a shape.
- **Delimiters** — fence untrusted or long inputs in `<tags>` or triple backticks. Also
  a first-line defence against [prompt injection](../domain-5/guardrails-security.md).
- **Structured output** — request JSON and validate it against a schema; retry on parse
  failure. Constrained decoding / JSON mode enforces this at the sampling level.
- **Negative instructions are weak** — "do not use jargon" works less well than
  "use plain language a beginner understands". State what you *want*.
- **Instruction placement** — with long contexts, repeat the instruction *after* the
  context as well; models attend more to the beginning and end than the middle
  ("lost in the middle").

## Decoding parameters

These control **sampling**, not the model's knowledge, and they are frequently examined.

| Parameter | Effect | Guidance |
| --- | --- | --- |
| **Temperature** | Scales the logits before softmax. `0` → deterministic/greedy; higher → flatter distribution, more random | **Low (0–0.3)** for extraction, classification, RAG, code. **High (0.7–1.0)** for brainstorming and creative writing |
| **Top-k** | Sample only from the *k* most likely tokens | Hard cutoff |
| **Top-p (nucleus)** | Sample from the smallest set whose cumulative probability ≥ *p* | Adaptive; 0.9–0.95 typical. Usually tune **either** top-p or temperature, not both |
| **Max tokens** | Output length cap | Truncation ≠ completion — a cut-off answer is a common bug |
| **Frequency / presence penalty** | Discourage repetition | Fixes loops |
| **Stop sequences** | Halt generation on a string | Keeps structured output clean |
| **Seed** | Reproducibility (where supported) | Essential for [experiments](../domain-3/experiment-design.md) |

!!! warning "Temperature 0 is not fully deterministic"
    Greedy decoding removes *sampling* randomness, but batching, GPU floating-point
    non-determinism and model updates can still change outputs. Pin model versions when
    reproducibility matters.

## The iterative workflow

NVIDIA's course objective says it directly: *"apply **iterative** prompt engineering best
practices."* Prompt engineering is empirical, not a one-shot act of writing.

1. **Build an evaluation set first** — 20–50 representative inputs with expected outputs.
   Without it you are tuning on vibes.
2. **Write a baseline prompt** — simple and zero-shot.
3. **Measure** — accuracy, format-compliance rate, latency, cost, human ratings, or an
   [LLM-as-a-judge](../domain-3/evaluation-metrics.md).
4. **Change one thing at a time** — otherwise you cannot attribute the improvement.
5. **Re-measure** on the same set; watch for regressions on cases that previously passed.
6. **Version the prompt** like code, and A/B test the finalists in production.

## Prompt-related risks

- **Prompt injection** — instructions hidden in user input or retrieved documents
  override your intent ("ignore previous instructions"). Untrusted content must be
  fenced and treated as data. See [Guardrails & LLM Security](../domain-5/guardrails-security.md).
- **Jailbreaking** — prompts crafted to bypass safety alignment.
- **Prompt leaking** — the model reveals its system prompt. Never put secrets in a prompt.
- **Context-window overflow** — silent truncation of exactly the part you needed.
- **Sensitive data exposure** — prompts sent to a hosted API leave your perimeter.

## Key takeaways

- Prompting adapts behaviour with **no weight updates**; always the first thing to try.
- Zero-shot → few-shot → chain-of-thought, in increasing order of cost and capability.
- Few-shot = **in-context learning**; consistent formatting matters more than example count.
- **Low temperature for factual/structured tasks, high for creative ones.**
- Iterate against a fixed evaluation set, changing one variable at a time.
- Fence untrusted input; assume anything in the prompt can be leaked or injected.
