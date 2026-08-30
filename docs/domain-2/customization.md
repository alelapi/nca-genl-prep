# Model Customization & PEFT

*NVIDIA's suggested reading names "Mastering LLM Techniques: Customization" and the
**LoRA** paper. Expect several questions on choosing a customization technique.*

## The customization ladder

Climb it from the bottom. Only move up when the rung below provably fails.

| # | Technique | Weights changed | Data needed | Cost | Best for |
| --- | --- | --- | --- | --- | --- |
| 1 | **Prompt engineering** | None | 0 | ~free | Behaviour that can be described |
| 2 | **Few-shot / in-context learning** | None | 2–10 examples | ~free | Format and pattern imitation |
| 3 | **RAG** | None | A document corpus | Low | **Knowledge** the model lacks |
| 4 | **PEFT (LoRA, QLoRA, adapters, P-tuning)** | ~0.1–1% | 100s–10,000s examples | Low–medium | Style, domain, task behaviour |
| 5 | **Full fine-tuning (SFT)** | 100% | 10,000s+ examples | High | Deep specialisation, distillation targets |
| 6 | **Continued pretraining** | 100% | Billions of tokens | Very high | A new language or a very distant domain |
| 7 | **Pretraining from scratch** | 100% | Trillions of tokens | $M+ | Almost never the right answer |

!!! tip "The exam's favourite question shape"
    *"A team wants the model to answer in their house style using their internal
    terminology, and to always return a specific JSON structure. They have 3,000 labeled
    examples and one GPU."* → **LoRA / PEFT**. Behaviour, not knowledge; modest data;
    limited compute.

    Change the stem to *"answer questions about documents updated daily"* and the answer
    becomes **RAG**.

## Supervised fine-tuning (SFT)

Train on `(instruction, ideal response)` pairs so the model learns to *do the task*
rather than merely continue text.

Practical rules:

- **Data quality beats data quantity.** A thousand carefully curated examples routinely
  beat a hundred thousand scraped ones.
- **Low learning rate** (typically 1e-5 to 5e-5 for full FT) and **few epochs** (1–3).
  Aggression causes [catastrophic forgetting](../domain-1/neural-networks.md#transfer-learning).
- **Hold out a validation set** and stop early.
- Format examples in the model's **expected chat template** — mismatched templates
  silently degrade results.

## Parameter-efficient fine-tuning (PEFT)

The core idea: **freeze the pretrained weights and train a small number of new
parameters.**

### LoRA — Low-Rank Adaptation

The technique NVIDIA explicitly lists as reading. For a pretrained weight matrix
`W ∈ ℝ^{d×k}`, LoRA learns a low-rank update:

$$ W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d\times r},\; A \in \mathbb{R}^{r\times k},\; r \ll \min(d,k) $$

- `W` is **frozen**; only `A` and `B` train.
- **Rank `r`** is typically 4–64. Parameters trained drop by ~100–1000×.
- `alpha` scales the update (`ΔW · alpha/r`); a common heuristic is `alpha = 2r`.
- Applied to the attention projection matrices (Q, K, V, O), sometimes the FFN too.
- At inference, `BA` can be **merged into `W`** — meaning **zero added latency**.

Why it works: the *adaptation* required to specialise a model is empirically low-rank,
even though the model itself is not.

**Benefits to be able to recite:**

- Trains on far less GPU memory (no optimizer states for the frozen weights).
- Checkpoints are megabytes, not gigabytes.
- **Many adapters, one base model** — serve dozens of task-specific LoRAs against a
  single loaded backbone and swap per request.
- Less catastrophic forgetting, because the base weights are untouched.

**QLoRA** = LoRA on top of a **4-bit quantized** frozen base model. Backpropagation flows
through the quantized weights into the LoRA parameters. This is what lets a 7B model
fine-tune on a single consumer GPU.

### Other PEFT methods

| Method | Idea |
| --- | --- |
| **Adapters** | Insert small bottleneck layers between transformer sub-layers; adds a little inference latency |
| **Prefix tuning** | Learn continuous vectors prepended to the keys/values of every layer |
| **P-tuning / prompt tuning** | Learn "soft prompt" embeddings prepended to the input; the model itself is entirely frozen |
| **IA³** | Learn per-channel rescaling vectors — even fewer parameters than LoRA |

!!! note "Soft prompts vs. hard prompts"
    A **hard prompt** is the text you write. A **soft prompt** is a set of learned
    embedding vectors that do not correspond to any real tokens — trained by gradient
    descent, invisible to a human reader. P-tuning and prompt tuning learn soft prompts.

## Alignment

After SFT, alignment tunes the model toward human preferences — helpful, harmless,
honest. **RLHF**, **DPO** and related methods are covered in
[Alignment & RLHF](../domain-3/rlhf-alignment.md), since the blueprint places RLHF under
Experimentation.

## Knowledge distillation

Train a small **student** model to reproduce the behaviour of a large **teacher**.

- The student learns from the teacher's **soft outputs** (the full probability
  distribution), which carry far more information than a hard label — the relative
  probabilities of the wrong answers encode the teacher's "understanding".
- Result: much smaller, faster, cheaper model retaining much of the quality.
- **DistilBERT** is the canonical example (~40% smaller, ~60% faster, ~97% of BERT's
  performance).
- Modern variant: generate synthetic instruction data with a large model and fine-tune a
  small one on it.

Related compression techniques: **pruning** (remove weights, heads or whole layers) and
**quantization** (reduce numeric precision — see
[Inference Optimization](inference-optimization.md)).

## Quantization-aware training (QAT)

Named directly in NVIDIA's reading list: *"Achieving FP32 accuracy for INT8 inference
using quantization-aware training with NVIDIA TensorRT."*

| | **Post-training quantization (PTQ)** | **Quantization-aware training (QAT)** |
| --- | --- | --- |
| When | After training is finished | During (or as a fine-tune after) training |
| Cost | Minutes; needs only a small calibration set | Requires a training run |
| Accuracy | Good; can degrade on sensitive models | Better — recovers most or all of the lost accuracy |
| How | Calibrate ranges, then cast | Simulate quantization ("fake quant") in the forward pass so the model *learns* to be robust to it |

**Rule:** try PTQ first. If accuracy drops unacceptably — especially at INT8 or lower —
move to QAT.

## Data for customization

The job description lists *"define, curate, label, and annotate LLM datasets"* as a core
responsibility.

- **Deduplicate** — duplicates cause memorisation and skew evaluation.
- **Decontaminate** — remove anything overlapping your evaluation/benchmark sets, or
  your metrics are fiction.
- **Balance** — the model will over-produce whatever is over-represented.
- **Annotation quality** — write a clear guideline, use multiple annotators, measure
  **inter-annotator agreement** (Cohen's/Fleiss' kappa). Low agreement means the task is
  underspecified, not that the annotators are bad.
- **Synthetic data** — generate with a stronger model, then filter. Cheap and effective;
  risks amplifying the teacher's biases and errors.
- **NeMo Curator** is NVIDIA's tool for this at scale (extraction, cleaning, exact and
  fuzzy deduplication, quality filtering, PII removal).

## Key takeaways

- Climb the ladder: prompting → few-shot → RAG → PEFT → full FT → pretraining.
- **Knowledge gap → RAG. Behaviour gap → fine-tuning.**
- **LoRA** freezes `W` and learns a low-rank `BA`; ~0.1–1% of parameters; adapters are
  tiny and **merge into the base with no inference latency**.
- **QLoRA** = LoRA over a 4-bit base — single-GPU fine-tuning of a 7B model.
- **Distillation** transfers a teacher's soft outputs to a smaller student.
- **PTQ first, QAT when accuracy suffers.**
- Curate ruthlessly: dedupe, decontaminate, balance, measure annotator agreement.
