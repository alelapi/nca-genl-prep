# Transformers & Attention

*The single most heavily examined topic on NCA-GENL. NVIDIA's suggested reading list opens
with* **Attention Is All You Need** *(Vaswani et al., 2017).*

This is a long page, and deliberately so. Almost everything else in the course — embeddings,
RAG, fine-tuning, KV caches, quantization, inference cost — is a consequence of how this
architecture works. If you understand the transformer properly, a large number of exam
questions stop being facts to memorise and become things you can reason out.

---

## 1. The problem the transformer was invented to solve

Start with what a language model actually does. Given a sequence of words, it must produce a
probability distribution over what comes next:

> "The cat sat on the ___"

To fill that blank well, a model needs two things. It needs to know **which earlier words
matter** — "cat" and "sat" are highly relevant, "The" barely at all — and it needs to know
**where those words were**, because "the cat chased the dog" and "the dog chased the cat"
contain identical words and mean opposite things.

Those two requirements — selective relevance and positional awareness — are the entire
design brief. Every component of the transformer exists to serve one of them.

### How the previous generation did it

Before 2017, sequence modeling meant **recurrent neural networks**. An RNN processes text the
way you read it: one token at a time, left to right, carrying a running summary called the
**hidden state**.

```text
        h₀ ──► h₁ ──► h₂ ──► h₃ ──► h₄ ──► h₅
               ▲      ▲      ▲      ▲      ▲
              "The"  "cat"  "sat"  "on"   "the"
```

At each step the network takes the previous hidden state `hₜ₋₁` and the current token, and
computes a new hidden state `hₜ`. The hidden state is a fixed-size vector — say 512 numbers —
that is supposed to encode everything relevant from all preceding text.

That design has two fatal problems, and understanding both is what makes the transformer's
solution feel inevitable.

**Problem one: the information bottleneck and vanishing gradients.**

Everything the model knows about token 1 must survive being repeatedly transformed on its way
to token 100. Each step multiplies the signal by a weight matrix and squashes it through a
non-linearity. If the effective multiplier at each step is slightly less than 1 — say 0.9 —
then after 100 steps the contribution of the first token has been scaled by 0.9¹⁰⁰ ≈ 0.000027.
It has effectively vanished.

The same thing happens in reverse during training. Gradients flowing backward from step 100
to step 1 get multiplied by the same small factors, so the model receives almost no learning
signal about long-range dependencies. This is the **vanishing gradient problem**, and it is
why plain RNNs cannot learn that "the **keys** that I left on the table in the kitchen this
morning **are** missing" requires "are" and not "is".

LSTMs and GRUs mitigated this with gating mechanisms — learned valves that decide what to
keep, forget and output — which pushed the usable range from tens of tokens to a few hundred.
They did not eliminate the problem; they postponed it.

**Problem two: sequence processing cannot be parallelised.**

This one killed RNNs commercially, and it is the reason a hardware company like NVIDIA cares.

To compute `h₅` you need `h₄`. To compute `h₄` you need `h₃`. The dependency chain is strictly
sequential. A GPU with 10,000 cores can only work on one timestep at a time, because timestep
*t+1* is not computable until timestep *t* has finished. You have bought a machine designed to
do thousands of things simultaneously and handed it a problem that must be done one thing at a
time.

Training time therefore scales linearly with sequence length and cannot be bought down with
more hardware. In an era where the winning strategy turned out to be "train on far more data",
that is a hard ceiling.

!!! quote "The insight behind the paper's title"
    The 2017 paper's contribution was not inventing attention — attention had been used *as an
    add-on to RNNs* since 2014. The contribution was noticing that if you keep the attention
    and **throw the recurrence away entirely**, you get a model that is both better at long
    range *and* fully parallelisable. Hence: *Attention Is All You Need*.

---

## 2. Attention, built up from scratch

Forget the formula for a moment. Let us derive the mechanism from the requirement.

We want: for each token, a way to gather information from every other token, weighted by how
relevant each one is. And we want the relevance weights to be **learned**, not hand-coded, and
computed for all tokens **simultaneously**.

### The dictionary analogy

Think about how a Python dictionary works:

```python
d = {"cat": "a small furry animal", "dog": "a loyal furry animal"}
result = d["cat"]     # exact match on the key, return the value
```

You supply a **query** (`"cat"`), it is compared against the **keys**, and on an exact match
you get back the corresponding **value**. It is all-or-nothing: the key either matches or it
does not.

Attention is a **soft** version of this lookup. Instead of returning one value on an exact
match, it returns a **weighted blend of all the values**, where the weights come from how
similar the query is to each key.

```text
query: "something furry"

    key "cat"  → similarity 0.5 ─┐
    key "dog"  → similarity 0.4 ─┼─► output = 0.5·value(cat) + 0.4·value(dog) + 0.1·value(car)
    key "car"  → similarity 0.1 ─┘
```

That is the whole idea. Everything that follows is mechanics.

### Where Q, K and V come from

Each token starts as an embedding vector — a list of numbers representing that token, say 512
of them. From that single vector, the model produces three different vectors by multiplying it
with three **learned weight matrices**:

$$ q_i = x_i W^Q \qquad k_i = x_i W^K \qquad v_i = x_i W^V $$

The three roles, in plain language:

- **Query (`q`)** — *"here is what I, this token, am looking for."* A verb might be looking
  for its subject. A pronoun might be looking for its antecedent.
- **Key (`k`)** — *"here is what I, this token, am. Here is what I advertise about myself."*
- **Value (`v`)** — *"here is the information I contribute if you decide to attend to me."*

Why three separate projections rather than just comparing the embeddings directly? Because
**what a token advertises and what it contributes are different things**, and both differ from
what it is looking for. The word "bank" might advertise "I am a noun, possibly financial,
possibly geographic" (its key) while contributing rich disambiguating content (its value) and
simultaneously asking "which nearby words tell me which sense I am?" (its query). Three
matrices give the model three independent degrees of freedom. `W^Q`, `W^K` and `W^V` are
learned by gradient descent like any other weights.

---

## 3. Scaled dot-product attention, worked through with real numbers

Here is the formula. We are going to take it apart and then compute an example by hand.

$$ \text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V $$

Five steps hide in there.

**Step 1 — Score.** Compute `Q · Kᵀ`. Every query is dotted with every key, producing an
`n × n` matrix where entry `(i, j)` is "how relevant is token *j* to token *i*".

The dot product is used because it is a cheap, differentiable similarity measure: it is large
and positive when two vectors point in the same direction, near zero when they are orthogonal,
and negative when they oppose. And critically, computing all n² dot products at once is a
single matrix multiplication — exactly the operation GPUs are built for.

**Step 2 — Scale.** Divide every score by `√d_k`. (Section 4 explains why in detail; it is a
favourite exam detail.)

**Step 3 — Mask.** In a decoder, set the scores for future positions to −∞ so they cannot be
attended to. (Section 5.)

**Step 4 — Softmax.** Convert each row of scores into a probability distribution — all weights
positive, each row summing to 1.

**Step 5 — Weighted sum.** Multiply the attention weights by `V`. Each token's output is a
blend of all the value vectors, weighted by relevance.

### A worked example

Let us make it concrete. Three tokens, and for readability we will use `d_k = 2`, so each
query, key and value is just two numbers. Assume the projections have already been applied and
produced:

| Token | query `q` | key `k` | value `v` |
| --- | --- | --- | --- |
| 1 | `[1, 0]` | `[1, 0]` | `[1, 0]` |
| 2 | — | `[0, 1]` | `[0, 1]` |
| 3 | — | `[1, 1]` | `[1, 1]` |

We will compute the output for **token 1** only.

**Score.** Dot `q₁` with each key:

```text
q₁ · k₁ = (1)(1) + (0)(0) = 1
q₁ · k₂ = (1)(0) + (0)(1) = 0
q₁ · k₃ = (1)(1) + (0)(1) = 1
```

Raw scores: `[1, 0, 1]`. Token 1 finds tokens 1 and 3 equally relevant, token 2 irrelevant.

**Scale.** Divide by `√d_k = √2 ≈ 1.414`:

```text
[1/1.414, 0/1.414, 1/1.414] = [0.707, 0, 0.707]
```

**Softmax.** Exponentiate and normalise:

```text
e^0.707 = 2.028      e^0 = 1.000      e^0.707 = 2.028
sum = 2.028 + 1.000 + 2.028 = 5.056

weights = [2.028/5.056, 1.000/5.056, 2.028/5.056]
        = [0.401, 0.198, 0.401]          (sums to 1.000 ✓)
```

**Weighted sum of values:**

```text
output = 0.401·[1, 0]  +  0.198·[0, 1]  +  0.401·[1, 1]
       = [0.401, 0]    +  [0, 0.198]    +  [0.401, 0.401]
       = [0.802, 0.599]
```

Token 1's output vector is `[0.802, 0.599]` — a blend dominated by tokens 1 and 3, with a
small contribution from token 2. **That is one attention operation, in full.**

The real thing does this for every token at once (so `q₂` and `q₃` get their own rows) with
`d_k = 64` or `128` instead of 2, but the arithmetic is identical. Nothing else is happening.

!!! note "The shapes, so you can follow any diagram"
    For a sequence of `n` tokens with model dimension `d_model` and head dimension `d_k`:

    - `X` is `n × d_model` — the input embeddings
    - `W^Q`, `W^K`, `W^V` are each `d_model × d_k` — the learned projections
    - `Q`, `K`, `V` are each `n × d_k`
    - `QKᵀ` is **`n × n`** — the attention score matrix, one row per token
    - The output is `n × d_v`, the same length as the input sequence

    That `n × n` matrix is the thing that makes attention powerful and expensive. Keep it in
    mind for section 9.

---

## 4. Why divide by √d_k — the argument in full

This is asked constantly, and most people memorise "it prevents saturation" without knowing
why. The reasoning is short and worth actually understanding.

Suppose the components of `q` and `k` are roughly independent, with mean 0 and variance 1 —
which is approximately what standard initialisation and layer normalization give you. The dot
product is a sum of `d_k` such products:

$$ q \cdot k = \sum_{i=1}^{d_k} q_i k_i $$

Each term `qᵢkᵢ` has mean 0 and variance 1. Variances of independent terms add, so:

$$ \operatorname{Var}(q \cdot k) = d_k \qquad\Longrightarrow\qquad \text{standard deviation} = \sqrt{d_k} $$

**The magnitude of the scores grows with the square root of the dimension.** With `d_k = 64`,
typical scores are around ±8. With `d_k = 512`, they are around ±23.

Now recall what softmax does with large inputs. Softmax is exponential, so it amplifies
differences ruthlessly. Two scores differing by 20 produce a probability ratio of
`e²⁰ ≈ 485,000,000` — one weight becomes essentially 1.0 and all the others essentially 0.
The distribution collapses to a **one-hot vector**.

Why is that bad? Because of the gradient. The derivative of softmax with respect to an input
whose output probability `p` is near 0 or near 1 involves the factor `p(1 − p)`, which
approaches **zero** in both cases. A saturated softmax passes almost no gradient backwards, so
the attention weights stop learning. The model gets stuck with whatever attention pattern it
happened to initialise with.

Dividing by `√d_k` exactly cancels the dimension-dependent growth, restoring the score
variance to 1 regardless of head size. The softmax stays in its responsive region and
gradients keep flowing.

!!! tip "The one-sentence exam answer"
    Without `√d_k`, dot products grow with dimension, the softmax saturates into a near-one-hot
    distribution, and the gradients vanish — so attention stops learning. The scaling keeps
    score variance constant so softmax stays in its sensitive range.

    Note what it is **not**: it is not normalising the output to unit length, and it does not
    change the O(n²) complexity. Both appear as distractors.

---

## 5. The causal mask — and why the model would learn nothing without it

Attention as described lets every token see every other token, including tokens that come
*after* it. For an encoder that is exactly what you want. For a model being trained to predict
the next token, it is catastrophic.

Think about what training looks like. You show the model "The cat sat on the mat" and ask it,
at every position simultaneously, to predict the following token. Position 3 should learn to
predict "on" from "The cat sat". But if position 3 can attend to position 4, it can simply
**read the answer** and copy it. The loss goes to zero, the model learns nothing, and at
inference time — when future tokens genuinely do not exist — it collapses.

The fix is the **causal mask** (also called the look-ahead mask). Before the softmax, every
score at position `(i, j)` where `j > i` is set to −∞:

```text
        to:  t₁    t₂    t₃    t₄
from t₁:   [ 0.4  -inf  -inf  -inf ]
from t₂:   [ 0.2   0.6  -inf  -inf ]
from t₃:   [ 0.1   0.3   0.5  -inf ]
from t₄:   [ 0.2   0.1   0.4   0.7 ]
```

Why −∞ specifically? Because `e^(−∞) = 0`. After softmax, those positions receive a weight of
exactly zero and contribute nothing to the weighted sum — while the remaining weights still
normalise correctly to sum to 1. Using −∞ rather than simply deleting the entries lets the
whole thing stay a single dense matrix operation, which is what keeps it fast.

This lower-triangular pattern is why decoder-only models are also called **causal** or
**autoregressive**: information flows only from past to future, never backwards.

!!! important "This mask is the entire difference between BERT and GPT"
    Same blocks, same attention, same feed-forward layers. BERT leaves the matrix full and
    trains by masking out ~15% of the *input tokens* and predicting them from both directions.
    GPT applies the causal mask and trains by predicting every next token.

    That single architectural choice is why encoders are good at understanding and embedding
    while decoders are good at generating. Section 10 develops this.

---

## 6. Multi-head attention — why one attention pattern is not enough

Consider the sentence:

> "The animal didn't cross the street because **it** was too tired."

To resolve "it", the model needs to link it to "animal". But other relationships matter at the
same time: "cross" needs its subject, "street" needs its determiner, "tired" needs whatever it
describes. A single attention distribution has to be *one* set of weights per token. It cannot
simultaneously look hard at "animal" for coreference and hard at "cross" for syntax — softmax
forces the weights to compete for a fixed budget of 1.0.

**Multi-head attention** solves this by running several attention operations in parallel, each
with its own `W^Q`, `W^K`, `W^V`, and concatenating the results:

$$ \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\,W^O $$

$$ \text{where}\quad \text{head}_i = \text{Attention}(XW_i^Q,\; XW_i^K,\; XW_i^V) $$

The dimensions are chosen so this costs **nothing extra**. With `d_model = 512` and `h = 8`
heads, each head works in `d_k = d_model / h = 64` dimensions:

```text
8 heads × 64 dims = 512 dims concatenated  →  W^O (512×512)  →  512 dims out
```

Parameter count for one big 512-dimensional head: `512 × 512 = 262,144` per projection.
For 8 heads of 64 dimensions: `8 × (512 × 64) = 262,144`. **Identical.** You get eight
different relationship patterns for the price of one.

The final `W^O` matters more than it looks: concatenation just stacks the heads side by side,
and `W^O` is the learned layer that mixes their outputs back into a single coherent
representation.

In practice, probing trained models shows heads specialising — some track syntactic
dependencies, some resolve coreference, some attend to the previous token, some attend almost
entirely to punctuation or `[SEP]` (so-called "null" heads that effectively opt out). Nobody
designed this; it emerges from training.

### The three flavours of attention

You must be able to distinguish these, because questions routinely test them:

| Type | Where it appears | What attends to what |
| --- | --- | --- |
| **Self-attention (bidirectional)** | Encoder | Every token ↔ every token, both directions. `Q`, `K`, `V` all come from the same sequence |
| **Masked (causal) self-attention** | Decoder | Token *i* attends only to tokens ≤ *i*. `Q`, `K`, `V` all from the same sequence, plus the mask |
| **Cross-attention** | Encoder–decoder bridge | `Q` comes from the **decoder**; `K` and `V` come from the **encoder** output |

Cross-attention is the mechanism by which a translation model looks back at the source
sentence while generating the target. The decoder asks the question; the encoder's
representation supplies the answer.

---

## 7. Positional encoding — because attention is blind to order

Here is a property of attention that surprises people: **it does not know what order the
tokens are in.**

Look back at the worked example. The output for token 1 was `0.401·v₁ + 0.198·v₂ + 0.401·v₃`.
Now shuffle tokens 2 and 3 in the input. The scores get computed against the same set of keys,
just in a different order; the softmax produces the same weights attached to the same tokens;
the weighted sum is the same numbers added in a different order — and addition is commutative.
**The output is identical.**

Formally, attention is *permutation-equivariant*: permute the input rows and you permute the
output rows, but nothing about the content changes. To attention, a sentence is a **bag of
tokens**, not a sequence. "Dog bites man" and "man bites dog" are the same input.

Since we deleted the recurrence — which was where order information used to live — we must put
order back in explicitly. The solution is to **add a position-dependent vector to each token
embedding** before the first layer:

```text
input to layer 1  =  token embedding  +  positional encoding
```

### Sinusoidal encoding (the original paper)

$$ PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right) \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

That looks arbitrary. It is not. Read it as a **binary counter made of sine waves**.

Each pair of dimensions oscillates at a different frequency. The lowest dimensions have a
wavelength of about 2π — they cycle every few positions, encoding fine-grained local position,
like the low-order bit of a binary number. The highest dimensions have a wavelength of about
`10000 · 2π` — they barely change across an entire document, encoding coarse global position,
like the high-order bit.

Read together, the full vector of sines and cosines specifies a position uniquely, the same
way the bits `0110` uniquely specify 6.

Two properties made this the original choice:

- **No parameters.** Nothing to learn, and it extrapolates to sequence lengths never seen
  during training — position 5,000 has a well-defined encoding even if training only went to
  512.
- **Relative position is linearly accessible.** Because of the sine and cosine addition
  formulas, `PE(pos + k)` can be written as a fixed linear transformation (a rotation) of
  `PE(pos)`. The model can therefore learn to attend by *relative offset* — "three tokens
  back" — using a linear operation, which is exactly the kind of thing a weight matrix can
  represent.

### The alternatives you should recognise

| Scheme | How it works | Where you meet it |
| --- | --- | --- |
| **Learned absolute** | A trainable embedding per position, added like a token embedding | BERT, GPT-2. Simple; cannot extrapolate past the trained maximum length |
| **RoPE (rotary)** | **Rotates** `q` and `k` by an angle proportional to position, instead of adding anything | Llama, Mistral, most modern LLMs |
| **ALiBi** | Adds a linear penalty to attention scores proportional to distance | Some long-context models |

**RoPE deserves a moment**, because it is what almost every current model uses and it appears
in questions. Rather than adding a position vector to the input, it rotates the query and key
vectors in 2-D subspaces by an angle `θ · position`. The mathematical payoff is that the dot
product between a rotated query at position *m* and a rotated key at position *n* depends
**only on their difference `m − n`**, not on their absolute values. Relative position falls out
of the geometry for free, and the encoding degrades gracefully at lengths beyond training —
which is why RoPE models can be context-extended and absolute-embedding models largely cannot.

---

## 8. The transformer block, layer by layer

A transformer is a stack of identical blocks. Each block has exactly two sub-layers, and each
sub-layer is wrapped in the same two things: a residual connection and a layer normalization.

```text
        x ─────────────────────────────┐
        │                              │ residual
        ▼                              │
   [ LayerNorm ]                       │
        │                              │
        ▼                              │
[ Multi-Head Attention ]               │
        │                              │
        ▼                              │
       (+) ◄───────────────────────────┘
        │
        ├─────────────────────────────┐
        ▼                             │ residual
   [ LayerNorm ]                      │
        │                             │
        ▼                             │
[ Feed-Forward Network ]              │
        │                             │
        ▼                             │
       (+) ◄──────────────────────────┘
        │
        ▼
     output
```

### The feed-forward network — where most of the parameters live

$$ \text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2 $$

Two linear layers with a non-linearity between them, applied **independently and identically
to every token position**. The inner dimension is conventionally **4× `d_model`** — so for
`d_model = 512`, the FFN expands to 2048 and projects back to 512.

Two questions worth answering properly.

**What is it actually for?** Attention *moves information between tokens*. It is fundamentally
a weighted average — it cannot compute new features, only mix existing ones. The FFN
*transforms information within a token*, and it is the only place in the block with substantial
non-linear processing capacity. A useful framing from the interpretability literature is that
the FFN behaves like a **key–value memory**: the rows of `W₁` act as pattern detectors that
fire on particular input features, and the corresponding columns of `W₂` are the content
written back when they fire. Much of a model's factual knowledge appears to live here.

**How big is it?** Per layer, attention uses four `d × d` matrices (`W^Q`, `W^K`, `W^V`, `W^O`)
for `4d²` parameters, and the FFN uses `d × 4d` and `4d × d` for `8d²`. Total ≈ **12d² per
layer**, of which **two thirds are in the feed-forward network**. When you read that a model's
weights are dominated by its FFNs, this is the arithmetic behind it.

### Residual connections — what makes depth possible

The `(+)` in the diagram is `output = SubLayer(x) + x`. The input is added back to the output.

This looks trivial and is one of the most important ideas in deep learning. Without it,
gradients flowing backward through 96 layers get multiplied by 96 Jacobians and vanish. With
it, there is a direct additive path from the loss all the way back to layer 1 — the gradient
can flow through the `+ x` branch essentially unattenuated.

There is a second, more conceptual benefit. Because each block *adds* to its input rather than
replacing it, the model maintains what is often called a **residual stream**: a running
representation that each layer reads from and writes small updates into. A block only has to
learn a *refinement*, not a full transformation, which is a much easier learning problem.

### Layer normalization — and why not batch normalization

Layer norm normalises each token's vector to zero mean and unit variance **across its own
features**, then applies a learned scale and shift.

Why not batch normalization, which is standard in vision? Batch norm computes statistics
**across the batch dimension** — the mean of feature 3 over all examples in the batch. That
creates two problems for text. First, sequences have **variable length**, so the number of
real values contributing to each position's statistics differs and padding contaminates them.
Second, it makes each example's output **depend on the other examples in its batch**, and it
behaves differently at inference (where you use stored running statistics, and batch size may
be 1). Layer norm computes statistics within a single token vector, so it is completely
independent of batch size and sequence length, and behaves identically in training and
inference.

**Pre-LN vs. post-LN** is worth knowing. The original paper put the normalization *after* the
sub-layer and residual addition (`LayerNorm(x + SubLayer(x))`). Modern models put it *before*
(`x + SubLayer(LayerNorm(x))`), as drawn above. Pre-LN keeps the residual path completely
clean — no normalization sits on it — which makes very deep stacks trainable without a
carefully tuned learning-rate warmup. Almost every model since around 2020 uses pre-LN.

### The original configuration

For reference, the 2017 paper's base model: 6 encoder layers, 6 decoder layers,
`d_model = 512`, 8 attention heads (so `d_k = 64`), FFN inner dimension 2048, ~65M parameters.
Modern LLMs keep the identical block and scale the numbers: a 70B model might use 80 layers,
`d_model = 8192`, 64 heads.

---

## 9. From text to prediction — the full pipeline

Putting it together, here is everything that happens when a decoder-only LLM produces one
token.

**1. Tokenization.** Text is split into subword units by a learned algorithm.

| Algorithm | Used by | Idea |
| --- | --- | --- |
| **BPE** (Byte-Pair Encoding) | GPT family | Start from characters, repeatedly merge the most frequent adjacent pair |
| **WordPiece** | BERT | Similar, but merges are chosen to maximise training-data likelihood |
| **SentencePiece / Unigram** | Multilingual models, Llama | Operates on raw bytes with no whitespace assumption — essential for languages without spaces |

Subwords are the compromise between two bad extremes. Word-level vocabularies are enormous and
break completely on unseen words. Character-level vocabularies are tiny but produce very long
sequences, and attention is O(n²). Subwords keep the vocabulary fixed (typically 32k–256k) while
still being able to spell **any** input by decomposing it — so there is no true
out-of-vocabulary problem.

!!! note "The ratio worth remembering"
    Roughly **1 token ≈ 0.75 English words ≈ 4 characters**. This is what you use for cost and
    context-window estimates. Non-English text and code tokenize less efficiently, sometimes
    much less.

**2. Embedding lookup.** Each token id indexes into an embedding matrix of shape
`vocab_size × d_model`, producing a dense vector.

**3. Add positional information.** Either added to the embedding (sinusoidal/learned) or
applied inside attention at every layer (RoPE).

**4. N transformer blocks.** The representation is refined layer by layer.

**5. Output head.** A final linear projection from `d_model` to `vocab_size` produces
**logits** — one raw score per vocabulary entry. This matrix is often *tied* to the input
embedding matrix (the same weights, transposed) to save parameters.

**6. Softmax and sample.** Logits become a probability distribution; a token is chosen from it
according to the decoding parameters — temperature, top-k, top-p. See
[Prompt Engineering](prompt-engineering.md#6-decoding-parameters).

**7. Append and repeat.** The chosen token is appended to the input and the whole thing runs
again. This is **autoregressive decoding**, and it is why generating 500 tokens costs roughly
500 forward passes.

---

## 10. Three architecture families, and why each is good at what it is good at

This mapping produces more exam questions than almost anything else, so it is worth
understanding causally rather than memorising.

| Family | Attention | Pretraining objective | Good at | Examples |
| --- | --- | --- | --- | --- |
| **Encoder-only** | Bidirectional | **Masked language modeling** | Understanding: classification, NER, **embeddings**, extractive QA | BERT, RoBERTa, DeBERTa, most embedding models |
| **Decoder-only** | Causal | **Next-token prediction** | Generation: chat, completion, summarization, code | GPT, Llama, Mistral, Nemotron |
| **Encoder–decoder** | Both, plus cross-attention | Span corruption / denoising | Transduction: translation, seq2seq summarization | T5, BART, the original transformer |

**Why bidirectional attention makes better representations.** To build a vector that captures
what a sentence *means*, you want every token informed by full context in both directions. In
"I went to the **bank** to deposit a cheque", the disambiguating evidence ("deposit", "cheque")
comes *after* the ambiguous word. An encoder sees it; a causal decoder at that position cannot.
This is why embedding models are overwhelmingly encoders.

**Why causal attention is required for generation.** As section 5 established, a model that can
see the future cannot be trained to predict it. Generation demands the mask, and the mask
costs you bidirectional context. That is the trade.

**Why encoder–decoder for translation.** The source sentence should be encoded bidirectionally
— you want full understanding of the input before producing anything — while the output must be
generated causally. Encoder–decoder gives you both, joined by cross-attention.

!!! tip "The mapping to internalise"
    - Classify, tag, or **embed** text → **encoder**
    - **Generate** text → **decoder**
    - Map one sequence to another (translation) → **encoder–decoder**

    NVIDIA phrases this in its own course objectives as *"use encoder models for semantic
    analysis, embedding, question-answering and zero-shot classification"* and *"work with
    conditioned decoder-style models to take in and generate interesting data formats, styles
    and modalities."* Recognising that language is worth a mark or two.

---

## 11. The cost of attention, and everything built to reduce it

Return to that `n × n` score matrix. For a sequence of `n` tokens, attention computes and
stores n² scores. **Self-attention is O(n²) in sequence length**, in both compute and memory.

Concretely: doubling the context from 4k to 8k tokens does not double attention cost — it
**quadruples** it. This single fact drives an enormous amount of engineering, and it is why
long context is expensive rather than merely inconvenient.

### The KV cache

During generation there is a huge amount of redundant work. To produce token 101, the model
needs the keys and values of tokens 1–100 — which it already computed when producing token 100.
Recomputing them every step makes generating an `n`-token sequence O(n³) overall.

The fix is to **cache** the K and V tensors for all previous positions. Each new token then
only computes its own `q`, `k`, `v`, appends to the cache, and attends over it. Generation
drops to O(n) per step.

This is not optional in any real serving stack, and the memory cost is substantial enough that
you should be able to estimate it:

$$ \text{KV cache bytes} = 2 \times \text{layers} \times \text{seq len} \times \text{kv heads} \times \text{head dim} \times \text{batch} \times \text{bytes per value} $$

The leading 2 is for K and V. Work through a real example — Llama-2-7B in FP16, which has 32
layers, 32 attention heads and head dimension 128:

```text
per token, per layer:  2 × 32 heads × 128 dims × 2 bytes  =  16,384 bytes  =  16 KB
per token, all layers: 16 KB × 32 layers                  =  512 KB
for a 4,096-token context:  512 KB × 4,096               ≈  2 GB
```

**2 GB of KV cache for a single 4k-token conversation** — against 14 GB for the model weights
themselves. Now serve 16 concurrent users and the cache needs 32 GB, more than double the
weights. This is why the KV cache, not the model, is usually what limits how many users a GPU
can serve.

### The optimizations you should recognise by name

- **Multi-query attention (MQA) / grouped-query attention (GQA)** — all heads (MQA) or groups
  of heads (GQA) *share* one set of K/V projections. Queries stay independent, so most of the
  expressiveness is retained, but the KV cache shrinks by the sharing factor. Rerun the
  calculation above with 8 KV heads instead of 32 and the cache drops from 2 GB to **512 MB**.
  Standard in every modern LLM.
- **FlashAttention** — computes exactly the same result, but reorders the computation into
  tiles that fit in the GPU's fast on-chip SRAM, so the `n × n` matrix is never written to
  slow HBM at all. Same math, dramatically less memory traffic, and memory that scales linearly
  rather than quadratically.
- **PagedAttention** — manages the KV cache in fixed-size blocks like operating-system virtual
  memory, eliminating fragmentation and allowing far larger batches. The core idea in vLLM.
- **Sparse / linear attention** — approximations (Longformer, BigBird, Performer) that attend
  to a subset of positions, trading exactness for sub-quadratic scaling.
- **Mixture of Experts (MoE)** — replaces the FFN with many expert FFNs and routes each token
  to only a couple of them. Total parameters are huge; *active* parameters per token stay
  small, so quality scales with total size while cost scales with active size.

---

## 12. Recap

The chain of reasoning, end to end:

1. RNNs failed for two reasons: information degraded over long distances (vanishing gradients),
   and sequential processing could not use GPU parallelism.
2. **Attention** replaces recurrence with a direct, learned, weighted lookup: every token can
   reach every other token in one step. `softmax(QKᵀ/√d_k)V`.
3. **Q, K, V** are three learned projections of the same input: what I seek, what I advertise,
   what I contribute.
4. **`√d_k`** keeps score variance constant so softmax does not saturate and gradients keep
   flowing.
5. The **causal mask** stops a decoder from reading the answer it is being trained to predict.
   It is the single difference that separates BERT-style from GPT-style models.
6. **Multi-head** attention learns several relationship patterns in parallel, at no extra
   parameter cost.
7. **Positional encodings** are mandatory because attention is order-blind. Sinusoidal,
   learned, or — in modern models — **RoPE**, which encodes relative position by rotation.
8. Each block is **attention + FFN**, each wrapped in a **residual connection** (makes depth
   trainable) and **layer normalization** (batch-size independent, unlike batch norm). The FFN
   holds about two thirds of the parameters.
9. **Encoder = understanding and embeddings. Decoder = generation. Encoder–decoder =
   transduction.** This follows directly from bidirectional vs. causal attention.
10. Attention is **O(n²)**, which makes long context expensive; the **KV cache** makes
    generation tractable but often dominates serving memory, which is why GQA, FlashAttention
    and PagedAttention exist.
