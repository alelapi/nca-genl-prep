# LLM Application Architecture

*Covers tasks 4.2 (build LLM use cases such as RAGs, chatbots and summarizers) and 4.7
(write software components or scripts).*

## Anatomy of an LLM application

```text
  client
    │
    ▼
 ┌───────────────────────────────────────────────────────────┐
 │  API layer        auth · rate limiting · request validation │
 ├───────────────────────────────────────────────────────────┤
 │  Orchestration    prompt assembly · routing · tool calling  │
 │                   memory · retries · fallback               │
 ├───────────────────────────────────────────────────────────┤
 │  Guardrails       input filtering · output validation · PII │
 ├───────────────────────────────────────────────────────────┤
 │  Retrieval        embed · vector search · rerank            │
 ├───────────────────────────────────────────────────────────┤
 │  Model serving    Triton / NIM / hosted API                 │
 ├───────────────────────────────────────────────────────────┤
 │  Observability    tracing · token & cost metrics · eval     │
 └───────────────────────────────────────────────────────────┘
```

Every one of these layers is a place where questions can be posed: *where does PII
redaction belong?* (guardrails, on the way in **and** out), *where do you enforce
per-user document permissions?* (retrieval, as metadata filters).

## Design decisions that get examined

**Hosted API vs. self-hosted model**

| | Hosted API | Self-hosted (Triton / NIM) |
| --- | --- | --- |
| Time to first prototype | Minutes | Days |
| Data residency / privacy | Data leaves your perimeter | Stays in your VPC/on-prem |
| Cost model | Per token | Per GPU-hour (cheap at high, sustained volume) |
| Customization | Limited | Full: fine-tune, quantize, swap models |
| Latency control | Vendor's | Yours |

**Model size selection.** Bigger is not automatically the answer. A 7–8B instruct model
with good RAG usually beats a 70B model with poor retrieval, at a fraction of the cost.
Route by difficulty: small model by default, escalate to a large one only when needed.

**Streaming.** Return tokens as they are produced (server-sent events). Time-to-first-token
dominates *perceived* latency far more than total generation time.

**Caching.** Three distinct kinds, worth distinguishing:

- **Exact-match response cache** — identical prompt → stored response. Trivial, effective
  for FAQs.
- **Semantic cache** — embed the query, return a cached answer if a previous query is
  close enough. Riskier; needs a tight threshold.
- **Prefix / KV cache reuse** — reuse computed key/value tensors for a shared prompt
  prefix (system prompt, few-shot examples) across requests. Pure latency win, no
  behaviour change.

## Conversational assistants

The hard part is not the model call, it is **state**.

| Memory strategy | Behaviour |
| --- | --- |
| Full history | Simple; grows until it overflows the context window and cost rises every turn |
| Sliding window (last *n* turns) | Bounded; forgets earlier context abruptly |
| Running summary | Bounded; compresses old turns, loses detail |
| Summary + recent window | The common production compromise |
| Retrieval over history | Scales to long relationships; retrieves only relevant past turns |

**Multi-turn RAG needs query rewriting.** "And what about the second one?" is
un-retrievable on its own. Rewrite it, using the conversation, into a standalone query
before embedding.

Other essentials: a **system prompt** carrying identity and policy, **guardrails** on
both directions, and **citations** whenever RAG is involved.

## Summarization services

- **Extractive vs. abstractive** — see [RAG](../domain-1/rag.md#other-classic-llm-use-cases-task-13).
- **Documents longer than the context window** — two standard strategies:
  - **Map-reduce**: summarise each chunk independently (parallelisable), then summarise
    the summaries. Fast; can lose cross-chunk connections.
  - **Refine**: carry a running summary forward chunk by chunk. Better continuity;
    strictly sequential, so slower.
- Control length, audience and format explicitly in the prompt; measure faithfulness,
  because abstractive summaries hallucinate.

## Structured output

For anything downstream code will parse:

1. Specify the schema **in the prompt**, with a filled example.
2. Use JSON mode / constrained decoding where the serving stack supports it.
3. **Validate** with Pydantic or JSON Schema.
4. **Retry** with the parse error fed back to the model.
5. Keep temperature low.

Never `eval()` model output, and never interpolate it directly into SQL or a shell
command — treat generated text exactly like untrusted user input.

## Agents and tool use

An **agent** is an LLM that chooses actions in a loop: reason → call a tool → observe →
repeat. Tools are functions with a described schema (search, calculator, SQL query, API
call).

**Function/tool calling** is the mechanism: the model emits a structured call, your code
executes it, and the result is fed back. The model never executes anything itself.

Risks to guard against: infinite loops (cap iterations), unbounded cost, and tools with
side effects (require confirmation before anything destructive or outward-facing).

## Testing an LLM application

Standard software testing assumes deterministic output. LLM applications need a layered
approach:

| Layer | What it tests | How |
| --- | --- | --- |
| Unit | Deterministic pieces: chunking, parsing, prompt templating, retrieval filters | Ordinary unit tests |
| Contract | Output is valid JSON / matches schema | Schema validation over a sample |
| Retrieval | Does the right chunk come back? | Recall@k, MRR against a labeled set |
| End-to-end quality | Is the answer good? | A fixed eval set + [metrics or LLM-as-judge](../domain-3/evaluation-metrics.md) |
| Regression | Did the prompt/model change break anything? | Re-run the eval set on every change; block on regressions |
| Safety | Jailbreaks, injection, PII leakage | Adversarial test suite |
| Load | Latency and throughput under concurrency | Load testing against the serving endpoint |

!!! tip "The practice that separates real systems from demos"
    **Version prompts, models and indexes together, and pin them.** An LLM app has three
    moving dependencies most codebases do not — a prompt, a model version and a document
    index. Any of them changing silently changes behaviour. Treat all three as versioned
    artefacts, and re-run the eval set whenever one moves.

## Key takeaways

- The layered architecture: API → orchestration → guardrails → retrieval → serving →
  observability.
- Streaming improves perceived latency; caching (exact, semantic, prefix/KV) cuts cost.
- Chatbots are a **memory** problem; multi-turn RAG requires **query rewriting**.
- Long-document summarization: **map-reduce** (parallel) vs. **refine** (sequential).
- Validate and retry structured output; never execute model output.
- Test in layers, and gate every prompt/model change on a fixed evaluation set.
