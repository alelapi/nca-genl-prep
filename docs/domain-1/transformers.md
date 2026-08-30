# Transformers & Attention

*The single most heavily examined topic on NCA-GENL. NVIDIA's suggested reading list
opens with* **Attention Is All You Need** *(Vaswani et al., 2017).*

## Why transformers replaced RNNs

| Problem with RNN/LSTM | Transformer's answer |
| --- | --- |
| Tokens processed **sequentially** → cannot parallelise across the sequence | Every position is processed **simultaneously** |
| Long-range dependencies decay through many recurrent steps | **Direct** connection between any two positions (path length 1) |
| Vanishing gradients over long sequences | Residual connections + attention |
| Training time scales with sequence length | Training parallelises across the whole sequence on GPUs |

The cost: self-attention is **O(n²)** in sequence length — every token attends to every
other token. That quadratic term is exactly why context windows are expensive and why
so much research targets efficient attention.

!!! quote "The one-sentence summary"
    Transformers replace recurrence with **attention**, letting every token look directly
    at every other token in a single, fully parallel operation.

## Self-attention, step by step

Each input token embedding is projected into three vectors by learned weight matrices:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I contain?"
- **Value (V)** — "what do I contribute if selected?"

**Scaled dot-product attention:**

$$ \text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V $$

Read it as five steps:

1. **Score** — `Q · Kᵀ` gives every token's affinity with every other token.
2. **Scale** — divide by `√d_k`. Without it, dot products grow with dimension, the
   softmax saturates, and gradients vanish. *This is a favourite exam detail.*
3. **Mask** (decoders only) — set future positions to `−∞` so they cannot be attended to.
4. **Softmax** — turn scores into attention weights that sum to 1.
5. **Weighted sum** — combine the `V` vectors using those weights.

## Multi-head attention

Instead of one attention function over `d_model` dimensions, run `h` attention heads in
parallel over `d_model / h` dimensions each, then concatenate and project.

Why: different heads learn different relationships — one may track syntactic
dependencies, another coreference, another positional proximity. It is an ensemble of
attention patterns computed at the same cost.

**Three flavours** you must distinguish:

| Type | Where | What attends to what |
| --- | --- | --- |
| **Self-attention (bidirectional)** | Encoder | Every token ↔ every token, both directions |
| **Masked (causal) self-attention** | Decoder | Token *i* attends only to tokens ≤ *i* |
| **Cross-attention** | Encoder–decoder | Decoder queries attend to **encoder** keys/values |

!!! warning "Why the causal mask exists"
    Without it, a decoder predicting token *i* could simply look at token *i* in the
    input — it would learn nothing. The mask makes next-token prediction a genuine
    prediction task.

## Positional encoding

Attention is **permutation-invariant**: shuffle the input and the attention math is
unchanged. Word order therefore has to be injected explicitly.

- **Sinusoidal** (original paper) — fixed sine/cosine functions of varying frequency,
  added to the token embeddings. No parameters; extrapolates to unseen lengths.
- **Learned absolute** — a trainable embedding per position (BERT, GPT-2).
- **RoPE (rotary)** — rotates Q and K by a position-dependent angle; encodes *relative*
  position and generalises better to long contexts. Used in Llama and most modern LLMs.
- **ALiBi** — adds a linear distance penalty to attention scores.

## The transformer block

Each layer is two sub-layers, each wrapped in a residual connection and layer
normalization:

```text
x ──►[ LayerNorm ]──►[ Multi-Head Attention ]──►(+)──┐
│                                                    │
└────────────────── residual ────────────────────────┘
                       │
                       ▼
x ──►[ LayerNorm ]──►[ Feed-Forward Network ]──►(+)──► output
│                                                    │
└────────────────── residual ────────────────────────┘
```

- **Feed-forward network (FFN)** — two linear layers with a non-linearity between them,
  applied **position-wise** and identically to every token. Its inner dimension is
  typically **4× `d_model`**. Most of a transformer's parameters live here.
- **Residual connections** — allow gradients to flow directly through deep stacks.
- **Layer normalization** — placed *before* the sub-layer in modern models
  (**pre-LN**, more stable to train) rather than after (**post-LN**, as in the original paper).

The original architecture: 6 encoder layers + 6 decoder layers, `d_model` = 512,
8 heads, FFN dim 2048.

## Three architecture families

| Family | Attention | Pretraining objective | Best at | Examples |
| --- | --- | --- | --- | --- |
| **Encoder-only** | Bidirectional | **Masked language modeling** (predict masked tokens) | *Understanding*: classification, NER, embeddings, extractive QA | BERT, RoBERTa, DeBERTa, embedding models |
| **Decoder-only** | Causal/masked | **Next-token prediction** (autoregressive) | *Generation*: chat, completion, summarization, code | GPT family, Llama, Mistral, Nemotron |
| **Encoder–decoder (seq2seq)** | Both + cross-attention | Span corruption / denoising | Transduction: translation, summarization | T5, BART, original transformer |

!!! tip "Highest-yield mapping on the whole exam"
    - Need to **classify, tag, or embed** text → **encoder** model.
    - Need to **generate** text → **decoder** model.
    - Need to map one sequence to another (translation) → **encoder–decoder**.

    NVIDIA's own course objective phrases it as: *"use encoder models for semantic
    analysis, embedding, question-answering and zero-shot classification"* and *"work
    with conditioned decoder-style models to generate data formats, styles and modalities."*

## From tokens to output

1. **Tokenization** — text → subword tokens. Algorithms: **BPE** (GPT), **WordPiece**
   (BERT), **SentencePiece/Unigram** (multilingual, whitespace-agnostic). Subwords give a
   fixed vocabulary that can still spell any word, and handle out-of-vocabulary terms by
   decomposition.
2. **Embedding lookup** — each token id → a dense vector of size `d_model`.
3. **+ positional encoding**.
4. **N transformer blocks**.
5. **Output head** — a linear projection to vocabulary size → **logits**.
6. **Softmax + sampling** → next token. Append and repeat (**autoregressive decoding**).

!!! note "Rule of thumb"
    ~1 token ≈ 0.75 English words ≈ 4 characters. Useful for cost and context-window
    estimates.

## Scaling and efficiency vocabulary

- **KV cache** — during generation, cache the K and V tensors of previous tokens so each
  new token costs O(n) instead of re-running O(n²). Trades memory for speed; the cache
  is often the dominant memory consumer at inference.
- **Multi-query (MQA) / grouped-query attention (GQA)** — heads share K/V projections,
  shrinking the KV cache dramatically. Standard in modern LLMs.
- **FlashAttention** — an IO-aware exact attention kernel that avoids materialising the
  n×n matrix in HBM. Same math, far less memory traffic.
- **Mixture of Experts (MoE)** — replace the FFN with many experts and route each token
  to a few. Large total parameter count, small *active* parameter count per token.
- **Sparse / linear attention** — approximations that reduce the O(n²) term.

## Key takeaways

- Attention lets every token attend to every other token in parallel; that is the whole
  reason transformers beat RNNs.
- `softmax(QKᵀ/√d_k)V` — know each symbol, and know that `√d_k` prevents softmax
  saturation.
- Multi-head attention = several attention patterns learned in parallel.
- Positional encodings are required because attention alone is order-blind.
- Encoder = understanding/embeddings, decoder = generation, encoder–decoder = transduction.
- Self-attention is **O(n²)** in sequence length — the root cause of context-length cost.
