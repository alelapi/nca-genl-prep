# Neural Networks & Deep Learning

*Foundation for the whole exam. NVIDIA's suggested readings name activation functions,
"Training Hidden Units With Back Propagation", and transfer learning explicitly.*

A transformer is a neural network. Everything in this page — how a neuron computes, what an
activation function is for, how gradients flow, why dropout works — is machinery the
transformer inherits. If any of it is shaky, [Transformers](transformers.md) will feel like
memorisation instead of understanding.

---

## 1. The artificial neuron

Everything starts with one very small computation. A neuron takes several inputs, weights each
one, sums them, adds a bias, and pushes the result through a non-linear function:

$$ y = \sigma\!\left(\sum_{i=1}^{n} w_i x_i + b\right) $$

Concretely, with three inputs:

```text
x₁ = 0.5 ──(w₁ = 2.0)──┐
                       │
x₂ = 1.0 ──(w₂ = −1.0)─┼──►  sum = (0.5)(2.0) + (1.0)(−1.0) + (0.2)(3.0) + 0.1
                       │          = 1.0 − 1.0 + 0.6 + 0.1
x₃ = 0.2 ──(w₃ = 3.0)──┘          = 0.7
                                                 ──► ReLU(0.7) = 0.7  ──► output
              bias b = 0.1
```

- The **weights** are what the network learns. A large positive weight means "this input
  strongly pushes me to fire"; a large negative weight means the opposite; near zero means
  "I ignore this input".
- The **bias** is a learned offset. Without it, the neuron's decision boundary would always
  pass through the origin — the bias lets it sit anywhere.
- The **activation** `σ` introduces non-linearity.

### Why non-linearity is not optional

This is worth proving to yourself, because it explains why activation functions exist at all.

Suppose you remove the activation and stack two layers. Layer 1 computes `h = W₁x + b₁`, layer
2 computes `y = W₂h + b₂`. Substitute:

$$ y = W_2(W_1x + b_1) + b_2 = \underbrace{(W_2W_1)}_{\text{just another matrix}}x + \underbrace{(W_2b_1 + b_2)}_{\text{just another vector}} $$

The composition of two linear functions is a **single linear function**. Stack a hundred layers
and you still have one matrix multiplication — a model no more expressive than linear
regression, but a hundred times slower.

Activation functions are what break this collapse. With a non-linearity between layers, each
layer genuinely adds representational power, and a sufficiently wide network can approximate
any continuous function (the **universal approximation theorem**).

---

## 2. Network anatomy

```text
   INPUT          HIDDEN 1        HIDDEN 2         OUTPUT
   layer           layer           layer           layer

    x₁ ────────►    ●  ────────►    ●  ────────►    ŷ₁
                 ╱  │  ╲         ╱  │  ╲         ╱
    x₂ ────────►    ●  ────────►    ●  ────────►    ŷ₂
                 ╲  │  ╱         ╲  │  ╱         ╲
    x₃ ────────►    ●  ────────►    ●  ────────►    ŷ₃

  one unit per   representation learning happens    shape set by
  feature        here — this is what "deep" means   the task
```

- **Input layer** — one unit per feature. Not really a layer; it is just the data.
- **Hidden layers** — where the network builds intermediate representations. "Deep" means more
  than one. Early layers learn simple patterns; later layers compose them into complex ones.
- **Output layer** — its shape and activation are determined by the task:

| Task | Output units | Output activation |
| --- | --- | --- |
| Binary classification | 1 | Sigmoid |
| Multi-class classification | *k* | Softmax |
| Regression | 1 | None (linear) |
| **Language modeling** | **vocabulary size** | **Softmax** |

That last row is the transformer's output head: a projection to 32,000-or-more logits followed
by a softmax over the whole vocabulary.

### The architecture families

| Family | Built for | Key idea |
| --- | --- | --- |
| **MLP / feedforward** | Tabular, generic | Fully connected layers |
| **CNN** | Images, local spatial patterns | Convolutions with shared weights → translation invariance and far fewer parameters |
| **RNN / LSTM / GRU** | Sequences | A hidden state carried forward; sequential, hard to parallelise, degrades over long range |
| **Transformer** | Sequences, and now almost everything | Attention; fully parallel. See [Transformers](transformers.md) |
| **Autoencoder** | Compression, denoising, anomaly detection | Encoder → narrow bottleneck → decoder; forced to keep only what matters |
| **GAN** | Generation | Generator vs. discriminator, trained adversarially |
| **Diffusion** | Image/video/audio generation | Learn to reverse a gradual noising process |

---

## 3. Activation functions

Each one is a different answer to "what shape should the non-linearity be?", and each has a
characteristic failure mode.

| Function | Formula | Range | Where used | Problem |
| --- | --- | --- | --- | --- |
| **Sigmoid** | `1/(1+e⁻ˣ)` | (0, 1) | Binary output; LSTM gates | Saturates → **vanishing gradient**; not zero-centred |
| **Tanh** | `tanh(x)` | (−1, 1) | Older RNNs | Zero-centred, but still saturates |
| **ReLU** | `max(0, x)` | [0, ∞) | The general deep-learning default | **Dying ReLU** — units stuck at zero |
| **Leaky ReLU** | `max(αx, x)`, α≈0.01 | (−∞, ∞) | Fixing dying ReLU | An extra hyperparameter |
| **GELU** | `x·Φ(x)` | ≈(−0.17, ∞) | **Transformers**: BERT, GPT | Slightly more expensive |
| **SwiGLU** | Gated variant | — | Modern LLMs: Llama, PaLM | More parameters per layer |
| **Softmax** | `eᶻⁱ/Σeᶻʲ` | (0,1), sums to 1 | **Output** layer; attention weights | Not a hidden activation |

### Understanding the two failure modes

**Saturation (sigmoid, tanh).** Look at a sigmoid's shape. For inputs beyond roughly ±5 the
curve is essentially flat, so its **derivative is essentially zero**. During backpropagation,
gradients are multiplied by these derivatives layer after layer. The maximum derivative of a
sigmoid is only 0.25 — so even in the best case, ten stacked sigmoid layers multiply the
gradient by at most `0.25¹⁰ ≈ 0.000001`. Early layers receive nothing and stop learning. This
is why deep networks were considered untrainable before ReLU.

**Dying ReLU.** ReLU's derivative is exactly 1 for positive inputs — no saturation, which is
why it fixed the problem above. But for negative inputs the output is 0 and the derivative is
**also 0**. If a neuron's weights drift so that it outputs negative for every training example,
it receives zero gradient forever and can never recover. It is dead. Leaky ReLU fixes this by
giving negatives a small non-zero slope, so there is always some gradient.

**Why transformers use GELU rather than ReLU.** GELU is smooth everywhere — no sharp kink at
zero — and it weights an input by the probability that a standard normal variable is below it,
a kind of probabilistic gating. In practice it trains slightly better in transformers, which
is empirical rather than principled, but it is the convention.

!!! tip "The two facts most likely to be tested"
    **ReLU** is the classic default hidden activation in deep learning. **GELU/SwiGLU** are
    what transformers actually use.

    **Softmax** converts a vector of raw scores (logits) into a probability distribution. It
    appears at the LLM output head **and inside attention** — it is not a hidden-layer
    activation, and options offering it as one are wrong.

---

## 4. Loss functions

The loss is the definition of "wrong". Everything the network learns is downstream of this
choice.

| Task | Loss | Notes |
| --- | --- | --- |
| Binary classification | **Binary cross-entropy** (log loss) | Pairs with sigmoid |
| Multi-class classification | **Categorical cross-entropy** | Pairs with softmax |
| **Language modeling** | **Cross-entropy over the vocabulary** | Per token. Exponentiating the mean gives **perplexity** |
| Regression | **MSE / L2** | Squares errors — punishes large ones hard, sensitive to outliers |
| Regression, robust | **MAE / L1**, **Huber** | Huber is quadratic near zero and linear in the tails |
| Embeddings / retrieval | **Contrastive**, **triplet**, **InfoNCE** | Pull positives together, push negatives apart |

### Cross-entropy, and why it is everywhere

For a single example with true class `c`, cross-entropy is:

$$ L = -\log p_c $$

where `p_c` is the probability the model assigned to the correct class. Work through what that
means:

```text
model assigns the correct class p = 0.99  →  L = −log(0.99) = 0.01     tiny loss
model assigns the correct class p = 0.50  →  L = −log(0.50) = 0.69     moderate
model assigns the correct class p = 0.10  →  L = −log(0.10) = 2.30     large
model assigns the correct class p = 0.01  →  L = −log(0.01) = 4.61     huge
```

The loss grows without bound as the assigned probability approaches zero. **Being confidently
wrong is punished enormously**, which is precisely the behaviour you want: a model that says
"99% certain" and is wrong should be penalised far more than one that says "60% certain" and
is wrong.

Because language modeling is classification over the vocabulary, this is the loss that trains
every LLM. Averaging it over tokens and exponentiating gives perplexity — see
[Evaluation Metrics](../domain-3/evaluation-metrics.md).

---

## 5. How training actually works

### The loop

```text
   ┌────────────────────────────────────────────────────┐
   │                                                    │
   ▼                                                    │
1. FORWARD PASS      inputs → predictions               │
   │                                                    │
   ▼                                                    │
2. COMPUTE LOSS      how wrong were we?                 │
   │                                                    │
   ▼                                                    │
3. BACKWARD PASS     ∂Loss/∂w for every parameter       │
   │                 (backpropagation)                  │
   ▼                                                    │
4. OPTIMIZER STEP    nudge every weight to reduce loss  │
   │                                                    │
   └────────────────────────────────────────────────────┘
              repeat over batches; one full pass = one EPOCH
```

### Backpropagation

**Backpropagation is not a learning algorithm.** It is an efficient algorithm for computing
gradients — specifically, the partial derivative of the loss with respect to every single
weight in the network. Gradient descent is what does the learning; backprop just supplies the
numbers it needs.

The mechanism is the **chain rule**, applied backwards through the computation graph. If the
loss depends on the output, which depends on layer 3, which depends on layer 2, which depends
on a weight `w`, then:

$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial \text{out}} \cdot \frac{\partial \text{out}}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial w} $$

The insight that makes it efficient is that these intermediate terms are shared between many
weights. Computing them once, from the output backwards, and reusing them costs roughly the
same as one forward pass — rather than the astronomically expensive alternative of perturbing
each weight individually to measure its effect.

Notice the **product** in that formula. It is the reason both vanishing and exploding gradients
exist, and section 7 returns to it.

### Gradient descent and its variants

The update rule is simply "step in the direction that reduces the loss":

$$ w \leftarrow w - \eta \frac{\partial L}{\partial w} $$

where `η` is the **learning rate**. The variants differ in how much data you use per step:

| Variant | Data per update | Character |
| --- | --- | --- |
| **Batch GD** | The entire dataset | Smooth, accurate gradient; hopelessly slow and memory-hungry at scale |
| **Stochastic GD (SGD)** | One sample | Very noisy; the noise can help escape shallow local minima; poor hardware utilisation |
| **Mini-batch GD** | 8–1024+ samples | **The practical default.** Good gradient estimate, excellent GPU utilisation |

### Optimizers

Plain gradient descent has known weaknesses: it oscillates across steep ravines, crawls along
shallow ones, and uses one learning rate for every parameter regardless of how different their
gradients are. Optimizers fix these.

| Optimizer | The idea it adds |
| --- | --- |
| **SGD** | The baseline update rule |
| **SGD + momentum** | Accumulate a velocity term. Like a ball rolling downhill — it damps oscillation across ravines and accelerates along consistent directions |
| **RMSProp** | Divide each parameter's step by a running average of its recent gradient magnitudes, giving every parameter its own effective learning rate |
| **Adam** | Momentum **and** RMSProp together. The general-purpose default |
| **AdamW** | Adam with *decoupled* weight decay — regularization applied directly to the weights rather than folded into the gradient. **The standard for training transformers** |

The AdamW distinction is small but real. In Adam, L2 regularization gets divided by the same
adaptive scaling as the gradient, which weakens it inconsistently across parameters. AdamW
applies the decay separately, so regularization behaves as intended. Every major LLM is trained
with AdamW.

### The learning rate — the hyperparameter that matters most

```text
too small                  about right                 too large
    │                          │                          │
    │  ╲                       │  ╲                       │    ╱╲    ╱╲
    │   ╲___                   │   ╲                      │   ╱  ╲  ╱  ╲
    │       ╲___               │    ╲___                  │  ╱    ╲╱    ╲
    └────────────► steps       └──────────► steps         └───────────────► steps
    creeps; may stall in       converges                  oscillates or
    a poor region                                         diverges entirely
```

The standard LLM schedule is **linear warmup followed by cosine decay**. Warmup — ramping the
learning rate up from near zero over the first few thousand steps — exists because at
initialisation the model's gradient estimates are unreliable and Adam's adaptive statistics
have not stabilised. A large step taken early can push the model into a bad region it never
recovers from. Cosine decay then anneals the rate down so training settles into a good minimum
rather than bouncing around it.

---

## 6. Regularization

Every technique here answers one question: **how do we stop the model memorising the training
set?**

### Weight decay (L2) and L1

Add a penalty for large weights to the loss:

$$ L_{\text{total}} = L_{\text{data}} + \lambda \sum_i w_i^2 \quad \text{(L2)} \qquad\qquad L_{\text{total}} = L_{\text{data}} + \lambda \sum_i |w_i| \quad \text{(L1)} $$

Why does this reduce overfitting? Large weights mean the model's output changes sharply in
response to small input changes — that is exactly the wiggly, high-variance behaviour that
fits noise. Penalising magnitude pushes the model toward smoother functions.

**L2** shrinks all weights smoothly toward zero without ever quite reaching it. **L1** has a
constant-magnitude gradient, which drives some weights to **exactly zero** — producing sparse
models and doing feature selection as a side effect.

`λ` controls the strength: too small and it does nothing, too large and you underfit.

### Dropout

During training, randomly set a fraction `p` of activations in a layer to zero on each forward
pass. A typical `p` is 0.1–0.5.

```text
training step 1        training step 2        at inference
   ●  ●  ✕  ●             ●  ✕  ●  ✕            ●  ●  ●  ●
   ✕  ●  ●  ●             ●  ●  ✕  ●            ●  ●  ●  ●
   (different units dropped each step)          (nothing dropped)
```

Why it works: a neuron cannot rely on any particular other neuron being present, because that
neuron might be dropped. This prevents **co-adaptation** — brittle chains where several units
only work in combination — and forces redundant, robust representations. It is also
interpretable as cheaply training an ensemble of exponentially many sub-networks that share
weights.

**Dropout is disabled at inference.** This is a real bug source: forgetting `model.eval()` in
PyTorch leaves dropout active and makes predictions randomly worse.

### Early stopping

Monitor validation loss each epoch. When it stops improving, stop training and keep the best
checkpoint.

```text
loss
  │╲
  │ ╲___________________ validation ── best checkpoint here ──┐
  │  ╲              ____/                                     ▼
  │   ╲________ ___/                                    stop training
  │    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ training                     (keep this one)
  └────────────────────────────────► epochs
```

The point where validation loss turns upward while training loss keeps falling is the moment
the model switches from learning the pattern to memorising the data. Simple, free, and
effective — which is why it is in every training loop.

### Data augmentation

Named in NVIDIA's course objectives: *"enhance datasets through data augmentation to improve
model accuracy."* Synthesise new training examples by transforming existing ones in ways that
preserve the label.

- **Images** — crop, flip, rotate, colour jitter, mixup, cutout.
- **Text** — synonym replacement, random insertion/swap/deletion, **back-translation**
  (translate to another language and back, producing a natural paraphrase), LLM paraphrasing.

The constraint is that the transformation must preserve the label. Flipping a photo of a cat
still shows a cat; flipping a photo of the digit "2" does not still show a "2". In text,
randomly deleting words can delete a negation and invert the sentiment. See
[Data Preprocessing](../domain-4/data-preprocessing.md).

### Normalization layers

Normalising activations to zero mean and unit variance stabilises and accelerates training by
keeping each layer's inputs in a well-conditioned range as the layers below it change.

**Transformers use layer normalization, not batch normalization**, and the reason is
structural. Batch norm computes statistics **across the batch** — the mean of feature 3 over
every example in the batch. For text that breaks twice: sequences have variable length, so
padding contaminates the statistics, and each example's output becomes dependent on which other
examples share its batch (including at inference, where the batch may be a single request).
Layer norm computes statistics **within one token's own feature vector**, so it is completely
independent of batch size and sequence length, and behaves identically in training and
inference.

---

## 7. Vanishing and exploding gradients

Return to the chain rule product from section 5:

$$ \frac{\partial L}{\partial w_{\text{early}}} = \frac{\partial L}{\partial h_n} \cdot \frac{\partial h_n}{\partial h_{n-1}} \cdots \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial w} $$

That is a product of many terms. Products of many numbers behave badly:

```text
if each term ≈ 0.8 :   0.8⁵⁰  ≈  0.0000143      →  VANISHING  (early layers get nothing)
if each term ≈ 1.2 :   1.2⁵⁰  ≈  9,100          →  EXPLODING  (updates blow up, loss → NaN)
```

**Vanishing gradients** mean early layers stop learning — they are effectively frozen at their
random initialisation while later layers do all the work.

**Exploding gradients** produce enormous weight updates, and training diverges. The symptom is
a loss that suddenly becomes `NaN`.

The mitigations, and what each one addresses:

| Technique | How it helps |
| --- | --- |
| **ReLU-family activations** | Derivative of exactly 1 for positive inputs — no saturating factor to multiply |
| **Residual (skip) connections** | `out = F(x) + x` creates an additive path with derivative 1, so gradients bypass the multiplicative chain entirely |
| **Normalization layers** | Keep activations, and therefore local derivatives, in a well-behaved range |
| **Careful initialization** | **Xavier/Glorot** for tanh, **He** for ReLU — scaled so signal variance is preserved layer to layer |
| **Gradient clipping** | Caps the gradient norm at a threshold. Directly prevents explosion; standard in LLM training |

!!! important "Residual connections are why deep transformers exist"
    `output = F(x) + x`. Because the derivative of the `+ x` branch is exactly 1, there is a
    path from the loss back to every layer along which the gradient is not attenuated at all.
    This is what makes 96-layer networks trainable, and it is why every transformer block is
    wrapped in one.

---

## 8. Transfer learning

Named repeatedly across NVIDIA's course objectives: *"leverage transfer learning between models
to achieve efficient results with less data and computation."*

### The idea

Training from scratch requires enormous data and compute. But a model trained on a large
general corpus has already learned representations that are useful far beyond its original
task — early layers of a vision model learn edges and textures; early layers of a language
model learn syntax and word meaning. Those are not task-specific. **Reuse them.**

```text
        Pretrained on a huge general corpus
        ┌──────────────────────────────────────┐
        │  layer 1   layer 2  ...  layer N     │  ──►  original head
        └──────────────────────────────────────┘
                        │
                        │  keep the body, replace the head
                        ▼
        ┌──────────────────────────────────────┐
        │  layer 1   layer 2  ...  layer N     │  ──►  YOUR new head
        └──────────────────────────────────────┘
             train on your (much smaller) dataset
```

### The three levels

**1. Feature extraction.** Freeze the entire pretrained body, replace and train only a new
output head. Fast, needs very little data, and there is almost no overfitting risk because you
are training very few parameters. Weakest adaptation.

**2. Fine-tuning.** Unfreeze some or all layers and continue training on your data at a **low
learning rate** — typically 10–100× lower than you would use from scratch. The low rate matters:
the weights already encode valuable knowledge, and large updates destroy it. A common pattern
is to unfreeze gradually, from the top down.

**3. Parameter-efficient fine-tuning (PEFT / LoRA).** Freeze the pretrained weights entirely
and train a small number of new parameters injected alongside them. This is the LLM-era answer,
and it is covered in [Customization & PEFT](../domain-2/customization.md).

### Catastrophic forgetting

The characteristic failure of aggressive fine-tuning. Train hard on a narrow dataset and the
model's weights shift to serve that dataset — destroying the general capability it arrived
with. A model fine-tuned intensively on legal contracts may become noticeably worse at ordinary
conversation.

Mitigations: low learning rates, few epochs (1–3 is typical), mixing general data into the
fine-tuning set, early stopping, or PEFT — which sidesteps the problem structurally by leaving
the base weights untouched.

!!! tip "The framing that ties the exam together"
    **The entire foundation-model paradigm is transfer learning at industrial scale.** Pretrain
    once at enormous cost; adapt cheaply, many times, for many tasks. Prompting, RAG and PEFT
    are all just increasingly lightweight points on the adaptation spectrum.

---

## 9. Recap

- A neuron is a weighted sum plus a bias, passed through a non-linearity. **Without the
  non-linearity, any stack of layers collapses to a single linear function.**
- **ReLU** is the classic default; **GELU/SwiGLU** are what transformers use; **softmax** is an
  output/attention function, not a hidden activation.
- Sigmoid and tanh **saturate** (vanishing gradients); ReLU can **die** (stuck at zero).
- **Cross-entropy** is the loss for classification and language modeling, and it punishes
  confident wrongness without bound. **MSE** for regression, **MAE/Huber** when outliers matter.
- **Backpropagation computes gradients** via the chain rule; the **optimizer** (AdamW for
  transformers) applies them. Mini-batch is the practical default.
- The **learning rate** is the most important hyperparameter; warmup then cosine decay is the
  LLM standard.
- Regularization: **L2/weight decay** (smooth shrinkage), **L1** (sparsity), **dropout**
  (prevents co-adaptation, off at inference), **early stopping**, **data augmentation**.
- **Layer norm, not batch norm**, in transformers — batch-size and sequence-length independent.
- Gradients vanish or explode because backprop **multiplies** many terms. **Residual
  connections** create an unattenuated additive path, which is what makes depth trainable.
- **Transfer learning** delivers strong results with far less data and compute than training
  from scratch; **catastrophic forgetting** is its characteristic failure, and PEFT avoids it
  structurally.
