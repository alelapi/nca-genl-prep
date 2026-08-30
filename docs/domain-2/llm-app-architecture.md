# LLM Application Architecture

*Covers tasks 4.2 (build LLM use cases such as RAGs, chatbots and summarizers) and 4.7 (write
software components or scripts under supervision).*

The study guide's job description is explicit that this role "contributes to the development,
programming, and quality assurance of state-of-the-art generative AI LLM systems". This page is
about the *system* around the model.

---

## 1. Why an LLM application is not just an API call

A demo is one function call. A production system has to answer questions a demo never asks:

- What happens when the model returns malformed JSON? (It will.)
- What happens when the vector database is down?
- How do you stop one user's prompt from costing $40?
- How do you know the answer was right?
- How do you make sure user A cannot retrieve user B's documents?
- What do you change when quality drops, and how do you know it worked?

Each of those pushes a layer into the architecture.

```text
  client
    │
    ▼
 ┌───────────────────────────────────────────────────────────────┐
 │  API layer         auth · rate limiting · request validation  │
 ├───────────────────────────────────────────────────────────────┤
 │  Orchestration     prompt assembly · routing · memory         │
 │                    tool calling · retries · fallback          │
 ├───────────────────────────────────────────────────────────────┤
 │  Guardrails        input filtering · PII · output validation  │
 ├───────────────────────────────────────────────────────────────┤
 │  Retrieval         embed · vector search · rerank · ACL filter│
 ├───────────────────────────────────────────────────────────────┤
 │  Model serving     Triton / NIM / hosted API                  │
 ├───────────────────────────────────────────────────────────────┤
 │  Observability     tracing · token & cost metrics · eval      │
 └───────────────────────────────────────────────────────────────┘
```

Each layer is a place a question can be posed. *Where does PII redaction belong?* — guardrails,
on the way in **and** on the way out. *Where do you enforce per-user document permissions?* —
the retrieval layer, as metadata filters, because once a chunk reaches the prompt it must be
treated as disclosed.

---

## 2. The architectural decisions that get examined

### Hosted API vs. self-hosted

| | **Hosted API** | **Self-hosted (Triton / NIM)** |
| --- | --- | --- |
| Time to first prototype | Minutes | Days |
| Data residency | Data **leaves your perimeter** | Stays in your VPC or on-prem |
| Cost model | Per token — cheap at low or spiky volume | Per GPU-hour — cheap at high sustained volume |
| Customization | Limited | Full: fine-tune, quantize, swap models |
| Latency control | The vendor's | Yours |
| Version stability | Can change with (or without) notice | You pin it |

The crossover is real and worth reasoning about: at low volume, GPU-hours are mostly idle time
you are paying for; at high sustained volume, per-token pricing is far more expensive than
owning the hardware. And for regulated data, the residency row often decides it before cost
enters the discussion.

### Model size and routing

Bigger is not automatically better. A 7–8B instruct model with **good retrieval** routinely
beats a 70B model with **poor retrieval**, at a fraction of the cost — because in a RAG system
the answer is in the context, and reading is easier than recalling.

**Routing** is the practical pattern: use a small model by default and escalate to a large one
only when needed, based on a classifier, a complexity heuristic, or the small model's own
confidence. Most traffic in most applications is easy.

### Streaming

Return tokens as they are produced (server-sent events) rather than waiting for the full
response.

This changes nothing about total generation time and transforms the experience, because
**perceived** latency is dominated by time-to-first-token. A response that starts in 300 ms and
finishes in 8 seconds feels dramatically faster than one that appears complete at 6 seconds.

### Caching

Three distinct kinds, and the distinction matters:

| Kind | Mechanism | Risk |
| --- | --- | --- |
| **Exact-match response cache** | Identical prompt → stored response | None, beyond staleness. Very effective for FAQs |
| **Semantic cache** | Embed the query; return a cached answer if a previous query is close enough | **Real.** "How do I cancel?" and "How do I cancel my *enterprise* plan?" are semantically close and may have different answers. Needs a tight threshold and careful evaluation |
| **Prefix / KV cache reuse** | Reuse computed K/V tensors for a shared prompt prefix — system prompt, few-shot block | None — it is a pure computation saving with identical output |

---

## 3. Conversational assistants

The model call is trivial. **State** is the hard part, because an LLM is stateless: every turn
must carry the relevant history in the prompt, and history grows without bound.

| Memory strategy | Behaviour |
| --- | --- |
| **Full history** | Simple. Grows every turn until it overflows the context window; cost rises every turn |
| **Sliding window** (last *n* turns) | Bounded cost. Forgets earlier context abruptly and completely |
| **Running summary** | Bounded. Compresses old turns; loses specifics like names and numbers |
| **Summary + recent window** | The common production compromise |
| **Retrieval over history** | Scales to long relationships; retrieves only the relevant past turns |

And the requirement people discover the hard way:

!!! important "Multi-turn RAG needs query rewriting"
    ```text
    Turn 1: "Tell me about the H100."
    Turn 2: "What about its memory bandwidth?"
    ```

    Embedding *"What about its memory bandwidth?"* retrieves nothing useful — "its" carries all
    the meaning and the embedding has no idea what it refers to. You must rewrite the query into
    a standalone form before retrieval:

    → *"What is the memory bandwidth of the NVIDIA H100 GPU?"*

    This is not optional in a multi-turn RAG system, and it is the most common reason such
    systems work perfectly in single-turn testing and fall apart in real conversations.

The rest: a **system prompt** carrying identity and policy, **guardrails** in both directions,
and **citations** whenever retrieval is involved.

---

## 4. Summarization services

**Extractive vs. abstractive** — extractive selects existing sentences (faithful by
construction, sometimes choppy); abstractive generates new text (fluent, **can hallucinate**).

For documents longer than the context window, two standard strategies:

```text
MAP-REDUCE                                REFINE
──────────                                ──────
chunk 1 ─► summary 1 ─┐                   chunk 1 ─► summary
chunk 2 ─► summary 2 ─┼─► final summary       │
chunk 3 ─► summary 3 ─┘                   chunk 2 + summary ─► updated summary
chunk 4 ─► summary 4 ─┘                       │
                                          chunk 3 + summary ─► updated summary
✓ parallel — fast                             │
✗ can miss cross-chunk connections        chunk 4 + summary ─► final

                                          ✓ better continuity across the document
                                          ✗ strictly sequential — slow
```

Control length, audience and format explicitly in the prompt, and **measure faithfulness** — a
fluent summary that invents a number is worse than no summary. See
[RAG Evaluation](../domain-3/rag-evaluation.md).

---

## 5. Structured output

For anything downstream code will parse, a five-step discipline:

1. **Specify the schema in the prompt**, with a filled example.
2. **Use JSON mode or constrained decoding** where the serving stack supports it — this enforces
   validity at the sampling level rather than hoping.
3. **Validate** the parsed result against a schema (Pydantic, JSON Schema).
4. **Retry**, feeding the parse error back to the model.
5. **Keep temperature low.**

!!! danger "Model output is untrusted input"
    Never `eval()` model output. Never interpolate it directly into SQL, a shell command, a
    file path or HTML. Use parameterised queries and escape for the destination context, exactly
    as you would with data submitted by an anonymous user on the internet.

    In the [OWASP Top 10 for LLMs](../domain-5/guardrails-security.md) this is **insecure output
    handling**, and it is a favourite exam scenario. The model may have been instructed to
    produce that SQL by a payload hidden in a retrieved document.

---

## 6. Agents and tool use

An **agent** is an LLM that chooses actions in a loop:

```text
    ┌──────────────────────────────────────────┐
    │                                          │
    ▼                                          │
  REASON  ──►  choose a TOOL  ──►  OBSERVE ────┘
  "I need the current price"     the result
                                            (repeat until done)
```

**Function/tool calling** is the mechanism: you describe available tools with a schema, the model
emits a structured call, **your code executes it**, and the result is fed back. The model never
executes anything itself — it only requests.

The risks are specific and worth naming:

- **Infinite loops** — cap the iteration count.
- **Unbounded cost** — each iteration is a full model call; cap the budget per request.
- **Side effects** — a tool that sends email, writes to a database or spends money should require
  confirmation. In OWASP terms this is **excessive agency**.
- **Least privilege** — scope every tool's permissions to the minimum. A read-only database role
  for a query tool is not paranoia; it is the difference between a bad answer and a dropped
  table.

---

## 7. Testing an LLM application

Standard testing assumes deterministic output. LLM applications are probabilistic, so testing
becomes layered — and the trick is that **most of the system is still deterministic and testable
normally.**

| Layer | What it tests | How |
| --- | --- | --- |
| **Unit** | Chunking, parsing, prompt templating, retrieval filters, ACL logic | Ordinary unit tests — these are deterministic |
| **Contract** | Output is valid JSON / matches the schema | Schema validation over a sample; measure the **format compliance rate** |
| **Retrieval** | Does the right chunk come back? | **Recall@k**, MRR against a labeled set |
| **End-to-end quality** | Is the answer good? | Fixed eval set + metrics or [LLM-as-a-judge](../domain-3/evaluation-metrics.md) |
| **Regression** | Did this change break anything? | Re-run the eval set on **every** prompt/model/index change; block on regressions |
| **Safety** | Jailbreaks, injection, PII leakage | An adversarial suite, grown from red-teaming |
| **Load** | Latency and throughput under concurrency | Load testing against the serving endpoint |

Two observations that make this tractable:

**Test the deterministic parts deterministically.** Chunking, metadata filtering, permission
checks and prompt assembly are ordinary code with ordinary tests. Do not let "LLMs are
non-deterministic" become an excuse for testing nothing.

**Separate retrieval testing from generation testing.** If the right chunk was never retrieved,
no prompt change will fix the answer — and an end-to-end score cannot tell you which half broke.
See [RAG Evaluation](../domain-3/rag-evaluation.md).

!!! tip "The practice that separates real systems from demos"
    **Version prompts, models and indexes together, and pin them in production.**

    An LLM app has three moving dependencies most codebases do not. Any of them changing
    silently changes behaviour, and two of them (the prompt and the index) are routinely edited
    without review because they do not feel like code. Treat all three as versioned artefacts,
    and re-run the evaluation set whenever one moves.

---

## 8. Recap

- Layered architecture: **API → orchestration → guardrails → retrieval → serving →
  observability**. Each layer answers a specific production question.
- **Hosted vs. self-hosted** turns on data residency, sustained volume and customization needs.
- **Streaming** improves perceived latency more than almost anything else; **caching** (exact,
  semantic, prefix/KV) cuts cost — but semantic caching carries real correctness risk.
- Chatbots are a **memory** problem; **multi-turn RAG requires query rewriting**.
- Long-document summarization: **map-reduce** (parallel, may miss connections) vs. **refine**
  (sequential, better continuity).
- Validate and retry structured output. **Never execute or interpolate model output unescaped.**
- Agents need iteration caps, cost caps, least-privilege tools, and confirmation for side effects.
- Test in layers, keep the deterministic parts deterministic, and gate every prompt/model/index
  change on a fixed evaluation set.
