# LLMs & Foundation Models

*Covers task 1.7 (read research papers to identify emerging LLM trends) and the
conceptual backbone of the whole exam.*

## What is a foundation model?

A **foundation model** is a large model trained with self-supervision on broad data at
scale, designed to be **adapted** to many downstream tasks rather than built for one.

Defining properties:

- **Scale** — billions of parameters, trillions of training tokens.
- **Self-supervised pretraining** — no human labels needed.
- **Generality** — one model, many tasks.
- **Adaptability** — prompting, RAG, fine-tuning, PEFT.
- **Emergent abilities** — capabilities (in-context learning, chain-of-thought
  reasoning) that appear only past a certain scale.

An **LLM** is a foundation model for text. Foundation models also exist for vision
(CLIP, SAM), speech (Whisper), multimodal (vision-language models) and biology.

!!! note "Generative AI vs. LLM vs. foundation model"
    - **Generative AI** — any model that produces new content (text, images, audio, code, video).
    - **Foundation model** — a large, broadly pretrained, adaptable base model.
    - **LLM** — a foundation model for language.

    Not all generative AI is an LLM (diffusion image models are not), and not all
    foundation models generate text (embedding models do not).

## The training lifecycle

```text
[ Pretraining ]  →  [ Supervised fine-tuning ]  →  [ Alignment ]  →  [ Adaptation ]
  trillions of        instruction / chat            RLHF, DPO         prompting,
  tokens, self-       demonstrations                                  RAG, PEFT
  supervised
  ─────────────       ─────────────────            ──────────        ──────────
  months, $M          hours–days                    days              minutes–hours
  → base model        → instruct model              → aligned         → your app
```

1. **Pretraining** — next-token (or masked-token) prediction over a huge corpus. Produces
   a **base model**: excellent at continuation, poor at following instructions.
2. **Supervised fine-tuning (SFT) / instruction tuning** — train on
   (instruction, ideal response) pairs. This is what makes a model *usable* as an assistant.
3. **Alignment** — RLHF, DPO or similar, using human preference data to make outputs
   helpful, harmless and honest. See [RLHF & Alignment](../domain-3/rlhf-alignment.md).
4. **Adaptation** — what *you* do: prompt engineering, RAG, PEFT/LoRA.

## Autoregressive language modeling

A decoder-only LLM models the probability of a sequence as a product of conditionals:

$$ P(x_1,\dots,x_n) = \prod_{i=1}^{n} P(x_i \mid x_1,\dots,x_{i-1}) $$

At each step it produces a probability distribution over the entire vocabulary and a
token is **sampled** from it. Two consequences that explain almost every LLM behaviour
question on the exam:

- **Output is probabilistic, not deterministic** — the same prompt can produce different
  answers (unless temperature is 0 / greedy decoding).
- **The model optimises plausibility, not truth** — which is the mechanistic origin of
  [hallucination](../domain-3/hallucinations.md).

**Masked language modeling** (BERT) is the alternative objective: mask ~15% of tokens
and predict them using **both** left and right context. Great representations, but the
model cannot generate fluently left-to-right.

## Scaling laws

NVIDIA's reading list includes *Deep Learning Scaling Is Predictable, Empirically*.

- Model loss falls as a **power law** in parameters, dataset size and compute — smoothly
  and predictably over many orders of magnitude.
- **Kaplan et al. (2020)** emphasised model size.
- **Chinchilla (Hoffmann et al., 2022)** corrected this: for a fixed compute budget most
  models were **undertrained**. The compute-optimal ratio is roughly **20 training tokens
  per parameter**. A smaller model trained on more data beats a larger, data-starved one.
- Practical consequence: modern 7–8B models trained on trillions of tokens outperform
  much larger 2020-era models, and are far cheaper to serve.

## Emergent abilities and in-context learning

**In-context learning** — the model performs a task from examples given *in the prompt*,
with **no weight updates**. This is what "few-shot" means; it is a property that emerges
with scale, not an explicitly trained feature. See [Prompt Engineering](prompt-engineering.md).

Other capabilities that appear with scale: chain-of-thought reasoning, instruction
following, basic arithmetic, code generation, tool use.

## Model families you should recognise

| Family | Type | Notable for |
| --- | --- | --- |
| **BERT** / RoBERTa / DeBERTa | Encoder | Bidirectional MLM pretraining; classification, NER, embeddings |
| **GPT** series | Decoder | Autoregressive generation; popularised few-shot prompting |
| **T5 / BART** | Encoder–decoder | Text-to-text framing of every NLP task |
| **Llama, Mistral, Qwen, Gemma** | Decoder | Open-weight models; the practical default for self-hosting |
| **NVIDIA Nemotron** | Decoder | NVIDIA's open model family, optimised for its stack |
| **Megatron-LM** | Decoder | NVIDIA's large-scale **training** framework (tensor/pipeline parallelism) |
| **Sentence-BERT / E5 / BGE / NV-Embed** | Encoder | Purpose-built **embedding** models |
| **CLIP** | Multimodal | Joint image–text embedding space |
| **Stable Diffusion / DALL·E** | Diffusion | Image generation — generative AI but **not** an LLM |

!!! tip "NVIDIA-specific names to keep straight"
    - **NeMo Framework** — end-to-end platform to build, customize and deploy generative
      AI models (data curation, training, fine-tuning, alignment, evaluation, export).
    - **NeMo Guardrails** — programmable safety/topic rails around an LLM application.
    - **NeMo Retriever** — embedding and reranking microservices for RAG.
    - **NeMo Curator** — large-scale data curation and deduplication.
    - **NIM (NVIDIA Inference Microservices)** — containerised, API-standard model
      inference you can deploy anywhere.
    - **TensorRT / TensorRT-LLM** — compilers/runtimes that optimize inference.
    - **Triton Inference Server** — the serving layer for models in production.
    - **RAPIDS** — GPU-accelerated data science (cuDF, cuML, cuGraph).
    - **Megatron-LM** — large-scale distributed *training*.

## Other generative architectures

**Diffusion models** (in NVIDIA's reading list) generate by learning to **reverse a
gradual noising process**: training adds Gaussian noise to data over many steps and the
model learns to denoise; generation starts from pure noise and denoises iteratively.
State of the art for images/video. Slower than GANs to sample, far more stable to train.

**GANs** pit a generator against a discriminator. Fast sampling, notoriously unstable
training, mode collapse.

**VAEs** encode to a probabilistic latent space and decode; blurrier outputs, useful
latents.

## Reading research papers (task 1.7)

The blueprint literally asks you to keep up with the literature. A practical method:

1. **Title + abstract** — what problem, what claim.
2. **Figures and tables** — the results, before the prose.
3. **Introduction and conclusion** — context and limitations.
4. **Method** — only once you care.
5. Ask: *what is the baseline, what is the delta, what did it cost, does it reproduce?*

Where the field is published: **arXiv** (cs.CL, cs.LG), **NeurIPS / ICML / ICLR / ACL /
EMNLP**, Hugging Face papers, the [NVIDIA Technical Blog](https://developer.nvidia.com/blog/).

The papers NVIDIA names — *Attention Is All You Need*, *BERT*, *LoRA* — are the ones to
have actually read.

## Key takeaways

- Foundation model = large + self-supervised + broadly adaptable; an LLM is the text case.
- Lifecycle: pretraining → SFT/instruction tuning → alignment → adaptation.
- Autoregressive = next-token prediction; probabilistic output is the root of hallucination.
- Chinchilla: ~20 tokens per parameter; smaller-but-better-trained beats bigger-and-starved.
- In-context learning needs **no weight updates**.
- Encoder = understand, decoder = generate, encoder–decoder = transduce.
- Know the NVIDIA product map: NeMo (build) → TensorRT-LLM (optimize) → Triton/NIM (serve).
