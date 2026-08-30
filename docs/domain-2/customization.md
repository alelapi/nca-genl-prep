# Model Customization & PEFT

*NVIDIA's suggested reading for this domain names "Mastering LLM Techniques: Customization" and
the **LoRA** paper directly. Expect several questions on choosing a customization technique.*

---

## 1. The customization ladder

You have a pretrained model and it does not quite do what you need. There is a spectrum of
responses, ordered by cost. **Climb it from the bottom, and only move up when the rung below
provably fails.**

```text
                                      cost ──────────────────────────►

7. Pretrain from scratch      ████████████████████████████████  $1M–$100M
6. Continued pretraining      ██████████████████                $10k–$1M
5. Full fine-tuning (SFT)     ████████                          $100–$10k
4. PEFT / LoRA                ███                               $1–$500
3. RAG                        ██                                infra + per-query
2. Few-shot prompting         ▌                                 tokens per call
1. Prompt engineering         ▏                                 ~free
```

| # | Technique | Weights changed | Data needed | Best for |
| --- | --- | --- | --- | --- |
| 1 | **Prompt engineering** | None | 0 | Behaviour you can describe in words |
| 2 | **Few-shot / in-context** | None | 2–10 examples | Format and pattern imitation |
| 3 | **RAG** | None | A document corpus | **Knowledge** the model lacks |
| 4 | **PEFT (LoRA, QLoRA, adapters)** | ~0.1–1% | 100s–10,000s examples | Style, domain, task behaviour |
| 5 | **Full fine-tuning (SFT)** | 100% | 10,000s+ examples | Deep specialisation |
| 6 | **Continued pretraining** | 100% | Billions of tokens | A new language or a very distant domain |
| 7 | **Pretraining from scratch** | 100% | Trillions of tokens | Almost never the right answer |

The ladder is not just about money. Each rung up adds an artefact you must version, evaluate,
store and serve; adds a way for things to silently regress; and adds latency between "we want
to change this" and "it is changed". A prompt edit ships in seconds. A fine-tune ships in days.

### The decision that gets examined

!!! tip "The question shape to recognise"
    *"A team wants the model to answer in their house style, use their internal terminology, and
    always return a specific JSON structure. They have 3,000 labeled examples and one GPU."*

    → **LoRA / PEFT.** This is a *behaviour* problem, not a knowledge problem; the data volume
    is modest; the compute is limited.

    Change one clause — *"…and answer questions about documents updated daily"* — and the answer
    becomes **RAG**, because that clause describes a knowledge problem.

    Change it again — *"…they have 40 examples"* — and the answer becomes **few-shot prompting**,
    because 40 examples will not fine-tune anything reliably.

The underlying rule, which resolves most of these:

- **Knowledge gap** — the model doesn't *know* something → **RAG**
- **Behaviour gap** — the model doesn't *act* how you want → **fine-tuning**
- **Both** → do both. Fine-tune for behaviour, retrieve for facts.

---

## 2. Supervised fine-tuning

Train on `(instruction, ideal response)` pairs so the model learns to perform the task rather
than merely continue text.

```text
{"instruction": "Summarise this incident report in two sentences.\n\n<report>...",
 "response":    "A memory leak in the ingest service caused OOM kills across three
                 pods. Mitigated by a rolling restart; permanent fix tracked in ENG-4471."}
```

### The rules that determine whether it works

**Data quality beats data quantity, by a wide margin.** A thousand carefully written,
consistently formatted examples routinely outperform a hundred thousand scraped ones. Every
inconsistency in your training data is a lesson the model dutifully learns.

**Use a low learning rate** — typically 1e-5 to 5e-5 for full fine-tuning, an order of magnitude
below what you would use training from scratch. The weights already encode enormous value;
large updates destroy it.

**Use few epochs** — 1 to 3. LLMs memorise fast. By epoch 5 on a small dataset the model is
reproducing your examples verbatim rather than generalising from them.

**Hold out a validation set and stop early.** Watch validation loss, not training loss.

**Match the model's chat template exactly.** Every instruct model was trained with specific
special tokens delimiting system, user and assistant turns. Fine-tuning with a different
template — or none — degrades results substantially, and nothing warns you. This is one of the
most common silent failures in practice.

---

## 3. Parameter-efficient fine-tuning

### The problem PEFT solves

Full fine-tuning a 7B model requires roughly **16 bytes per parameter** for optimizer state —
about **112 GB** — before activations. That does not fit on a single 80 GB GPU. The arithmetic
is in [Hardware & Memory Sizing](hardware-sizing.md), and it is worth internalising because it
is the entire motivation for what follows.

The core insight of PEFT: **freeze the pretrained weights and train a small number of new
parameters instead.**

Since the frozen weights need no gradients and no optimizer states, the memory cost collapses.
You still store the weights (to compute forward passes), but you no longer store a gradient, an
FP32 master copy, and two Adam moments for each of them.

### LoRA — Low-Rank Adaptation

The technique NVIDIA lists as required reading.

**The observation.** When you fine-tune a model, the *change* to each weight matrix — `ΔW` — has
low intrinsic rank. The model itself is high-rank and complex, but the *adaptation* to a
specific task turns out to be describable with far fewer numbers.

**The method.** Instead of learning a full `ΔW` of shape `d × k`, factor it into two thin
matrices:

$$ W' = W + \Delta W = W + BA, \qquad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k) $$

```text
        FULL FINE-TUNING                        LoRA
        ────────────────                        ────

         ┌─────────────┐                  ┌─────────────┐
         │             │                  │             │
    x ──►│      W      │──► out      x ──►│   W (FROZEN)│──►(+)──► out
         │  (trained)  │                  │             │    ▲
         └─────────────┘                  └─────────────┘    │
          d × k params                          │            │
          ALL updated                           └──►[A]──►[B]┘
                                                   r×k    d×r
                                                   ~0.1–1% of params
                                                   TRAINED
```

**Work through the numbers.** Take an attention projection with `d = k = 4096`:

```text
full ΔW:      4096 × 4096              = 16,777,216 parameters
LoRA, r=8:    (4096 × 8) + (8 × 4096)  =     65,536 parameters

reduction: 256×
```

**The hyperparameters:**

- **`r` (rank)** — typically 4 to 64. Higher `r` means more capacity to adapt and more
  parameters. Trainable parameters scale **linearly** with `r`.
- **`alpha`** — a scaling factor; the update is applied as `BA · (alpha / r)`. A common
  heuristic is `alpha = 2r`. Its purpose is to decouple the learning rate from the rank, so
  changing `r` does not require retuning everything else.
- **`target_modules`** — which matrices get adapters. Conventionally the attention projections
  (Q, K, V, O); adding the FFN layers increases capacity and cost.

**`A` is initialised randomly and `B` is initialised to zero.** That detail matters: it means
`BA = 0` at the start, so the adapted model is *exactly* the base model on step one, and
training begins from a known-good state rather than a perturbed one.

### Why LoRA is so widely used

Four benefits, and you should be able to state all four:

**Memory.** No optimizer states for 99%+ of parameters. A 7B model becomes fine-tunable on a
single 24 GB consumer card.

**Storage.** An adapter checkpoint is **megabytes**, not gigabytes. You can keep hundreds of
task-specific versions in a git repository.

**Multi-tenancy.** Serve **many adapters against one loaded base model**, swapping per request.
Twenty customers with twenty fine-tuned variants share one set of base weights in GPU memory
instead of twenty.

**No inference latency.** At serving time, `BA` can be **merged into `W`**:

```text
W_merged = W + BA          computed once, offline
```

The result is a model architecturally *identical* to the base — same layers, same shapes, same
speed. This is LoRA's structural advantage over adapter layers, which insert extra modules that
must be executed on every forward pass.

Plus, because the base weights are never touched, LoRA substantially reduces **catastrophic
forgetting**.

[Lab 6](../labs/06-peft-lora.md) has you run this and print the actual parameter counts, adapter
size on disk, and the merge.

### QLoRA

**QLoRA = LoRA on top of a 4-bit quantized frozen base model.**

The base weights are quantized to 4 bits (typically NF4, a data type designed for
normally-distributed weights). Gradients flow *through* those quantized weights during
backpropagation to reach the LoRA parameters, which stay in higher precision.

```text
7B model, full fine-tuning     ≈ 112 GB   ✗ needs multiple A100s
7B model, LoRA (FP16 base)     ≈  20 GB   ✓ fits a 24 GB card, tightly
7B model, QLoRA (4-bit base)   ≈   6 GB   ✓ fits comfortably on consumer hardware
```

This is what put LLM fine-tuning within reach of individuals rather than just labs.

### The other PEFT methods

| Method | Idea | Trade-off |
| --- | --- | --- |
| **Adapters** | Insert small bottleneck layers between transformer sub-layers | Works well; **adds inference latency** because the layers cannot be merged |
| **Prefix tuning** | Learn continuous vectors prepended to keys and values at every layer | No architecture change; consumes context |
| **P-tuning / prompt tuning** | Learn "soft prompt" embeddings prepended to the input; model entirely frozen | Fewest parameters; less expressive |
| **IA³** | Learn per-channel rescaling vectors | Even smaller than LoRA |

!!! note "Soft prompts vs. hard prompts"
    A **hard prompt** is text you write — real tokens, human-readable.

    A **soft prompt** is a set of learned embedding vectors that do not correspond to any real
    token. They are trained by gradient descent, live in continuous space, and are unreadable.
    P-tuning and prompt tuning learn soft prompts.

    This is a real conceptual distinction and appears in questions: prompt *tuning* trains
    parameters, prompt *engineering* does not.

---

## 4. Knowledge distillation

Train a small **student** model to reproduce the behaviour of a large **teacher**.

```text
                  input
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  ┌──────────┐            ┌──────────┐
  │ TEACHER  │            │ STUDENT  │
  │  (70B)   │            │   (7B)   │
  └────┬─────┘            └────┬─────┘
       │                       │
   soft outputs ──────────► match them
   [0.7, 0.2, 0.1]          (this is the loss)
```

**Why soft outputs matter — the key insight.** A hard label tells the student "the answer is
cat". The teacher's full probability distribution says "70% cat, 20% dog, 10% car". That second
form encodes something the label does not: it says dogs are more cat-like than cars are. These
relative probabilities across the *wrong* answers — often called **dark knowledge** — carry a
compressed picture of the teacher's learned structure, and training on them transfers far more
than training on labels alone.

**DistilBERT** is the canonical result: ~40% smaller, ~60% faster, retaining ~97% of BERT's
performance.

The modern LLM variant is slightly different in mechanics but identical in spirit: use a large
model to **generate synthetic instruction data**, filter it for quality, and fine-tune a small
model on it.

Related compression techniques worth naming: **pruning** (remove weights, attention heads or
entire layers judged unimportant) and **quantization** (reduce numeric precision — see
[Inference Optimization](inference-optimization.md)).

---

## 5. Quantization-aware training

Named directly in NVIDIA's reading list: *"Achieving FP32 accuracy for INT8 inference using
quantization-aware training with NVIDIA TensorRT."*

| | **Post-training quantization (PTQ)** | **Quantization-aware training (QAT)** |
| --- | --- | --- |
| When | After training is complete | During training, or as a fine-tune afterwards |
| Cost | Minutes; needs only a small calibration set | Requires a full training run |
| Accuracy | Good; can degrade on sensitive models | Better — usually recovers most of the loss |
| Mechanism | Measure activation ranges on calibration data, compute scale factors, cast | Insert "fake quantization" into the forward pass so the model *learns* to be robust |

**How QAT actually works.** During training, weights and activations are quantized and
immediately dequantized in the forward pass — simulating the precision loss the deployed model
will experience — while the backward pass uses full precision (via a straight-through
estimator, since quantization has zero gradient almost everywhere). The model therefore learns
weights that remain accurate *after* rounding, rather than weights that happen to round badly.

!!! tip "The exam rule"
    **Try PTQ first.** It is cheap and usually sufficient. **Move to QAT when accuracy loss is
    unacceptable**, particularly at INT8 and below.

    Question shape: *"After quantizing to INT8 the model's accuracy dropped unacceptably. What
    should the team try?"* → **quantization-aware training**.

---

## 6. Data for customization

The study guide lists *"define, curate, label, and annotate LLM datasets"* as a core job
responsibility, so dataset work is examinable in its own right.

**Deduplicate.** Duplicates cause memorisation, waste compute, and corrupt evaluation when the
same item lands in both training and test.

**Decontaminate.** Remove anything overlapping your evaluation or benchmark sets. Otherwise your
metrics measure memorisation, not capability. See
[Evaluation Metrics](../domain-3/evaluation-metrics.md#6-benchmarks).

**Balance.** The model over-produces whatever is over-represented. If 80% of your fine-tuning
examples are refusals, expect a model that refuses.

**Ensure annotation quality.** Write an explicit guideline, use multiple annotators, and measure
**inter-annotator agreement** (Cohen's or Fleiss' κ). Low agreement means the *task* is
underspecified — a model trained on those labels cannot exceed their noise ceiling. See
[Alignment & RLHF](../domain-3/rlhf-alignment.md#5-human-annotation-quality).

**Consider synthetic data.** Generate with a stronger model, then filter. Cheap and effective;
the risk is amplifying the teacher's biases and errors, and producing data that is
suspiciously uniform in style.

**NeMo Curator** is NVIDIA's GPU-accelerated tool for this pipeline at scale: extraction,
cleaning, quality filtering, exact and fuzzy deduplication, PII removal.

---

## 7. Recap

- Climb the ladder: **prompting → few-shot → RAG → PEFT → full FT → continued pretraining →
  from scratch**. Each rung costs more money, more time and more operational surface.
- **Knowledge gap → RAG. Behaviour gap → fine-tuning.**
- SFT rules: quality over quantity, **low learning rate**, **1–3 epochs**, match the chat
  template exactly.
- **LoRA** freezes `W` and learns `ΔW = BA` with rank `r`. Trainable parameters scale linearly
  with `r`; `B` starts at zero so training begins from the exact base model.
- LoRA's four benefits: **low memory, tiny checkpoints, many adapters per base model, and zero
  added inference latency once merged.**
- **QLoRA** = LoRA over a 4-bit quantized frozen base — single-GPU fine-tuning of a 7B model.
- **Distillation** transfers the teacher's **soft output distribution**, which carries more
  information than hard labels.
- **PTQ first, QAT when accuracy suffers.**
- Curate ruthlessly: dedupe, decontaminate, balance, and measure annotator agreement.
