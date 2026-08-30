# LLMs & Foundation Models

*Covers task 1.7 (read research papers to identify emerging LLM trends and technologies) and
supplies the vocabulary the rest of the exam assumes.*

---

## 1. What a foundation model is, and why the term exists

A **foundation model** is a large model trained with self-supervision on broad data at scale,
designed to be **adapted** to many downstream tasks rather than built for one.

The term was coined in 2021 to name something that had already happened. Before it, the
standard practice was one model per task: a sentiment classifier, a separate named-entity
recogniser, a separate translation model, each trained from scratch on its own labeled dataset.
Every new task meant a new data-collection effort and a new training run.

Foundation models inverted this. You pretrain **once**, at enormous cost, on unlabeled data;
then you adapt cheaply, many times, to many tasks:

```text
OLD PARADIGM                          FOUNDATION MODEL PARADIGM
────────────                          ─────────────────────────
labeled data A ─► model A             ┌──────────────────────────┐
labeled data B ─► model B             │   ONE pretrained model   │
labeled data C ─► model C             │  (unlabeled data, once)  │
                                      └────────────┬─────────────┘
each from scratch                          ┌───────┼───────┐
each needs labels                          ▼       ▼       ▼
                                        task A  task B  task C
                                        (prompt / RAG / PEFT — cheap)
```

The defining properties:

- **Scale** — billions of parameters, trillions of training tokens.
- **Self-supervised pretraining** — labels come from the data itself, so training data is
  effectively unlimited. See [ML Fundamentals](ml-fundamentals.md).
- **Generality** — one model, many tasks.
- **Adaptability** — via prompting, RAG, PEFT, or full fine-tuning.
- **Emergent abilities** — capabilities that appear only past a certain scale.

An **LLM** is a foundation model for text. Foundation models also exist for vision (CLIP, SAM),
speech (Whisper), multimodal understanding, and protein structure.

!!! note "Three terms that overlap but are not synonyms"
    - **Generative AI** — any model that produces new content: text, images, audio, code, video.
    - **Foundation model** — a large, broadly pretrained, adaptable base model.
    - **LLM** — a foundation model for language.

    Not all generative AI is an LLM: Stable Diffusion generates images and is a diffusion model,
    not a language model. Not all foundation models generate: embedding models produce vectors,
    not text. And not all LLMs are generative in the usual sense: BERT is a language model that
    produces representations.

---

## 2. The training lifecycle

Understanding these four stages — and their wildly different costs — explains most of the
practical decisions on this exam.

```text
┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌──────────────┐
│  PRETRAINING    │──►│  SUPERVISED      │──►│  ALIGNMENT   │──►│  ADAPTATION  │
│                 │   │  FINE-TUNING     │   │              │   │              │
│ trillions of    │   │ 10k–1M           │   │ human        │   │ prompting,   │
│ tokens,         │   │ (instruction,    │   │ preference   │   │ RAG,         │
│ self-supervised │   │  response) pairs │   │ data, RLHF   │   │ PEFT/LoRA    │
│                 │   │                  │   │ or DPO       │   │              │
│ months, $1M–$100M│  │ hours–days       │   │ days         │   │ minutes–hours│
│                 │   │                  │   │              │   │              │
│ → BASE model    │   │ → INSTRUCT model │   │ → ALIGNED    │   │ → YOUR app   │
└─────────────────┘   └──────────────────┘   └──────────────┘   └──────────────┘
     done by            done by model           done by            done by YOU
   model builders         builders            model builders
```

**Pretraining** produces a **base model**. This distinction matters and is often misunderstood:
a base model is excellent at *continuing text* and poor at *following instructions*. Ask a base
model "What is the capital of France?" and a plausible continuation is another question — because
documents full of questions exist in the corpus. It is not being unhelpful; it is doing exactly
what it was trained to do.

**Supervised fine-tuning (SFT) / instruction tuning** is what converts that into something
usable. Train on (instruction, ideal response) pairs and the model learns that the correct
continuation of an instruction is a *response*.

**Alignment** then shapes *how* it responds — helpful, harmless, honest — using human preference
data via RLHF or DPO. See [Alignment & RLHF](../domain-3/rlhf-alignment.md).

**Adaptation** is your layer: prompting, RAG, PEFT. See
[Customization](../domain-2/customization.md).

!!! tip "Why this diagram is worth memorising"
    The cost column explains the entire practical calculus of the field. Pretraining is the
    domain of a handful of organisations. Everything an "associate developer" does — the role
    NVIDIA's study guide describes — happens in the rightmost box, occasionally the second.
    Exam scenarios that end with "so we should pretrain a model from scratch" are essentially
    always wrong.

---

## 3. Autoregressive language modeling

A decoder-only LLM factorises the probability of a sequence into a product of conditionals:

$$ P(x_1, x_2, \dots, x_n) = \prod_{i=1}^{n} P(x_i \mid x_1, \dots, x_{i-1}) $$

In words: the probability of the whole text is the probability of the first token, times the
probability of the second given the first, times the probability of the third given the first
two, and so on. Generation is the same formula run forwards — at each step, produce a
distribution over the vocabulary and sample a token.

Two consequences explain a remarkable share of LLM behaviour, and therefore of exam questions.

**Output is probabilistic, not deterministic.** The same prompt can produce different answers
across runs unless you force greedy decoding. This is not a bug; sampling is how the model
produces varied, natural text. See
[Prompt Engineering](prompt-engineering.md#6-decoding-parameters).

**The model optimises plausibility, not truth.** Nothing in the objective rewards being correct
— only being likely. A fabricated citation with a real-looking author, journal and year is *very
likely text*. This is the mechanistic origin of [hallucination](../domain-3/hallucinations.md),
and understanding it turns a whole family of questions from memorisation into reasoning.

### The alternative objective

**Masked language modeling** (BERT) hides ~15% of the tokens and asks the model to predict them
using **both** left and right context. This produces better representations — bidirectional
context is strictly more information — but the model cannot generate fluently left to right,
because it was never trained to.

This is the objective difference underlying the encoder/decoder split in
[Transformers](transformers.md).

---

## 4. Scaling laws

NVIDIA's reading list includes *Deep Learning Scaling Is Predictable, Empirically*, and the
topic matters more than it might appear — it is the reason anyone was willing to spend
$100 million on a training run.

**The empirical finding:** model loss falls as a **power law** in parameters, dataset size and
compute — smoothly and predictably across many orders of magnitude. That predictability is the
point. You can train a series of small models, fit the curve, and *forecast* what a model
10,000× larger will achieve before you build it. Training runs became capital investments with
projectable returns rather than gambles.

**Kaplan et al. (2020)** established the power-law relationships and were read as emphasising
model size.

**Chinchilla (Hoffmann et al., 2022)** corrected the interpretation, and this is the exam-worthy
part. For a **fixed compute budget**, most models of that era were **undertrained**: too large,
and fed too little data. The compute-optimal ratio is roughly **20 training tokens per
parameter**.

```text
Same compute budget, two ways to spend it:

  A 175B model trained on 300B tokens        ← 1.7 tokens/param — badly undertrained
  A  70B model trained on 1.4T tokens        ← 20 tokens/param  — compute-optimal

The 70B model performs BETTER and is ~2.5× cheaper to serve, forever.
```

That second clause is the practical punchline. Inference cost scales with model size and is
paid on every single request for the life of the model, while training cost is paid once.
Modern practice pushes past even Chinchilla-optimal — training 8B models on 15T tokens
(1,875 tokens per parameter) — because *serving* economics dominate once a model is deployed.

---

## 5. Emergent abilities and in-context learning

Some capabilities do not improve smoothly with scale — they are essentially absent below a
threshold and present above it. These are **emergent abilities**, and they were not designed;
they were discovered by testing models that had already been built.

The most important is **in-context learning**: the model performs a task from examples supplied
in the prompt, with **no weight updates**. This is what makes few-shot prompting work, and it is
why "just show it three examples" is a viable engineering strategy at all. See
[Prompt Engineering](prompt-engineering.md).

Others: chain-of-thought reasoning, instruction following, multi-step arithmetic, code
generation, tool use.

!!! note "A caveat worth knowing"
    There is genuine debate about whether emergence is a real phase transition or an artefact of
    using discontinuous metrics — measuring exact-match accuracy makes a smoothly improving
    capability *look* like it appears suddenly, because partially-correct answers score zero
    until they become fully correct. Both readings agree on the practical fact: below a certain
    scale these capabilities are not usable, above it they are.

---

## 6. The model landscape

| Family | Type | Notable for |
| --- | --- | --- |
| **BERT**, RoBERTa, DeBERTa | Encoder | Bidirectional MLM pretraining; classification, NER, **embeddings** |
| **GPT** series | Decoder | Autoregressive generation; demonstrated few-shot prompting at scale |
| **T5**, BART | Encoder–decoder | Reframed every NLP task as text-to-text |
| **Llama, Mistral, Qwen, Gemma** | Decoder | **Open-weight** — the practical default for self-hosting, fine-tuning and research |
| **NVIDIA Nemotron** | Decoder | NVIDIA's open model family, optimised for its own stack |
| **Sentence-BERT, E5, BGE, NV-Embed** | Encoder | Purpose-built **embedding** models |
| **CLIP** | Multimodal | Joint image–text embedding space; enabled text-to-image search and generation |
| **Whisper** | Encoder–decoder | Speech recognition |
| **Stable Diffusion, DALL·E** | Diffusion | Image generation — generative AI, **not** LLMs |

### Open-weight vs. proprietary

A distinction that drives real architectural decisions and appears in scenario questions:

| | **Open-weight** (Llama, Mistral, Nemotron) | **Proprietary API** |
| --- | --- | --- |
| Where it runs | Your infrastructure | The vendor's |
| Data residency | Stays in your perimeter | Leaves it |
| Fine-tuning | Full control (PEFT, full FT, quantization) | Limited or unavailable |
| Cost model | GPU-hours — cheap at high sustained volume | Per token — cheap at low volume |
| Operations | You run it | They run it |
| Version stability | You pin it; it never changes underneath you | Can change with notice, or without |

For regulated industries, the data-residency row alone often decides it — which is exactly the
market NVIDIA's **NIM** addresses.

---

## 7. The NVIDIA stack — names and roles

Product-identification questions are close to free marks if you have the map. Full detail in
[Deployment & Serving](../domain-2/deployment.md); here is the orientation.

```text
   DATA                BUILD / CUSTOMIZE           OPTIMIZE              SERVE
 ──────────           ──────────────────         ────────────        ─────────────
 NeMo Curator   ────► NeMo Framework      ─────► TensorRT-LLM  ────► Triton Server
 curate, dedupe,      pretrain, SFT, PEFT,       TensorRT             NIM microservices
 strip PII            align, evaluate, export    ONNX, quantization    OpenAI-style API
                            │                                              │
 RAPIDS ───────────────────►│                                              │
 cuDF/cuML/cuGraph          │                                       NeMo Guardrails
 GPU data science      Megatron-LM                                  safety rails
                       large-scale training
                                                                    NeMo Retriever
                                                                    embed + rerank
```

| Name | One-line identity |
| --- | --- |
| **NeMo Framework** | End-to-end platform to build, customize, align, evaluate and export generative AI models |
| **NeMo Curator** | Large-scale data curation: extraction, filtering, deduplication, PII removal |
| **NeMo Guardrails** | Programmable safety and topic rails around an LLM application |
| **NeMo Retriever** | Embedding and reranking microservices for RAG |
| **NeMo Evaluator** | Model and RAG evaluation |
| **Megatron-LM** | Large-scale distributed **training** (tensor and pipeline parallelism) |
| **TensorRT / TensorRT-LLM** | Inference **optimization**: fusion, kernel tuning, quantization |
| **Triton Inference Server** | Production **serving**: multi-framework, dynamic batching, versioning, metrics |
| **NIM** | Prepackaged, containerised, optimized model endpoints with a standard API |
| **RAPIDS** | GPU-accelerated data science: cuDF, cuML, cuGraph |
| **NCCL** | Multi-GPU collective communication library |

!!! tip "The confusion to avoid"
    **TensorRT optimizes a model. Triton serves models.** They are complementary, not
    alternatives — you typically compile with TensorRT-LLM and deploy the resulting engine on
    Triton. Questions naming "optimize inference performance" want TensorRT; those naming
    "serve multiple models with dynamic batching over HTTP/gRPC" want Triton.

---

## 8. Other generative architectures

The blueprint's reading list includes NVIDIA's article on diffusion models, so recognise the
non-LLM generative families.

**Diffusion models.** Training progressively adds Gaussian noise to real data across many
steps, and the model learns to **reverse** each step — to denoise. Generation starts from pure
noise and iteratively denoises into an image.

```text
TRAINING:    clean image ──add noise──► noisier ──► noisier ──► pure noise
                                    (model learns to reverse each step)

GENERATION:  pure noise ──denoise──► ... ──denoise──► clean image
```

State of the art for images and video. Much more stable to train than GANs; slower to sample,
because generation requires many sequential denoising steps.

**GANs.** A **generator** creates fakes while a **discriminator** tries to detect them, trained
adversarially until the fakes are convincing. Fast single-pass sampling; notoriously unstable
training, and prone to **mode collapse** where the generator produces only a few outputs it
knows fool the discriminator.

**VAEs.** Encode to a probabilistic latent space and decode back. Blurrier outputs than GANs or
diffusion, but the latent space is smooth and useful — and VAEs appear *inside* latent diffusion
models like Stable Diffusion.

---

## 9. Reading research papers (task 1.7)

The blueprint literally asks you to "read research papers to identify emerging LLM trends and
technologies". Here is a workable method.

**The order to read in** — not front to back:

1. **Title and abstract** — what problem, what claim.
2. **Figures and tables** — the results, before the prose that frames them.
3. **Introduction and conclusion** — context, and what the authors admit they did not solve.
4. **Method** — only once you have decided you care.
5. **Related work** — to place it in the field, or to find better prior papers.

**The questions to ask of any result:**

- What is the **baseline**, and is it a fair one? A large gain over a weak baseline is not a gain.
- How big is the **delta**, and is it inside the noise? Are error bars or multiple seeds reported?
- What did it **cost**? A 2% quality gain for 10× the compute is not always progress.
- Does it **reproduce**? Is there code, are there weights?
- What is the **evaluation**, and could it be [contaminated](../domain-3/evaluation-metrics.md)?

**Where the field publishes:** arXiv (cs.CL and cs.LG), NeurIPS, ICML, ICLR, ACL, EMNLP; Hugging
Face Papers for what practitioners are actually reading; and the
[NVIDIA Technical Blog](https://developer.nvidia.com/blog/) for the engineering side.

The papers NVIDIA names in its study guide — **Attention Is All You Need**, **BERT**, **LoRA** —
are the ones worth having genuinely read rather than merely heard of.

---

## 10. Recap

- **Foundation model** = large + self-supervised + broadly adaptable. An **LLM** is the text
  case. Generative AI is the broader category and includes diffusion models, which are not LLMs.
- Lifecycle: **pretraining** (base model, months, $M) → **SFT/instruction tuning** (usable
  assistant) → **alignment** (helpful, harmless, honest) → **adaptation** (your work).
- A **base model continues text**; an **instruct model follows instructions**. That gap is what
  SFT closes.
- **Autoregressive** = next-token prediction. Output is probabilistic, and the objective rewards
  **plausibility rather than truth** — the root of hallucination.
- **Chinchilla**: ~20 tokens per parameter is compute-optimal; smaller-and-better-trained beats
  larger-and-starved, and is far cheaper to serve.
- **In-context learning** is emergent and requires **no weight updates**.
- **Encoder = understand/embed. Decoder = generate. Encoder–decoder = transduce.**
- NVIDIA map: **NeMo builds → TensorRT-LLM optimizes → Triton/NIM serves**, with RAPIDS for data
  and Guardrails for safety.
