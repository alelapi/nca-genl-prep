# Neural Networks & Deep Learning

*Foundation for the whole exam. NVIDIA's suggested readings name activation functions,
backpropagation and transfer learning explicitly.*

## The artificial neuron

A neuron computes a weighted sum of its inputs plus a bias, then applies a non-linear
activation:

$$ y = \sigma\!\left(\sum_i w_i x_i + b\right) $$

- **Weights** `w` — learned parameters, the strength of each connection.
- **Bias** `b` — a learned offset that lets the activation shift left/right.
- **Activation** `σ` — the non-linearity.

!!! important "Why non-linearity is not optional"
    Stack a hundred purely linear layers and you still have a linear function — the
    composition collapses to a single matrix. Activation functions are what let deep
    networks approximate arbitrary functions (the **universal approximation theorem**).

## Network anatomy

- **Input layer** — one unit per feature.
- **Hidden layers** — where representation learning happens. "Deep" simply means more
  than one.
- **Output layer** — shaped by the task: 1 unit + sigmoid (binary), *k* units + softmax
  (multi-class), 1 linear unit (regression), vocabulary-sized + softmax (language modeling).

**Architecture families**

| Family | Built for | Note |
| --- | --- | --- |
| **MLP / feedforward** | Tabular, generic | Fully connected layers |
| **CNN** | Images, local patterns | Convolutions, weight sharing, translation invariance |
| **RNN / LSTM / GRU** | Sequences | Sequential, hard to parallelise, vanishing gradients on long context |
| **Transformer** | Sequences, now everything | Attention; parallelisable; see [Transformers](transformers.md) |
| **Autoencoder** | Compression, anomaly detection | Encoder → bottleneck → decoder |
| **GAN** | Generation | Generator vs. discriminator, adversarial |
| **Diffusion** | Image/video/audio generation | Learn to reverse a noising process |

## Activation functions

| Function | Formula | Range | Where used | Watch out |
| --- | --- | --- | --- | --- |
| **Sigmoid** | $1/(1+e^{-x})$ | (0, 1) | Binary output, gates | Saturates → **vanishing gradient**; not zero-centred |
| **Tanh** | $\tanh(x)$ | (−1, 1) | Older RNNs | Zero-centred but still saturates |
| **ReLU** | $\max(0, x)$ | [0, ∞) | Default hidden activation | Cheap, no saturation for x>0; **dying ReLU** for x<0 |
| **Leaky ReLU** | $\max(\alpha x, x)$ | (−∞, ∞) | Fixes dying ReLU | Small slope α ≈ 0.01 |
| **GELU** | $x\,\Phi(x)$ | ≈(−0.17, ∞) | **Transformers** (BERT, GPT) | Smooth, probabilistic gating |
| **SwiGLU** | gated variant | — | Modern LLMs (Llama, PaLM) | Better quality per parameter |
| **Softmax** | $e^{z_i}/\sum_j e^{z_j}$ | (0,1), sums to 1 | Multi-class **output**, attention weights | Not a hidden activation |

!!! tip "Two facts worth memorising"
    **ReLU** is the default hidden-layer activation in deep learning; **GELU/SwiGLU** are
    what modern transformers actually use. **Softmax** turns logits into a probability
    distribution and appears both at the LLM output head and inside attention.

## Loss functions

The loss defines *what the network is being asked to minimise*.

| Task | Loss | Notes |
| --- | --- | --- |
| Binary classification | **Binary cross-entropy** (log loss) | Pairs with sigmoid |
| Multi-class classification | **Categorical cross-entropy** | Pairs with softmax |
| **Language modeling** | **Cross-entropy over the vocabulary** | Per-token; exponentiating the mean gives [perplexity](../domain-3/evaluation-metrics.md) |
| Regression | **MSE / L2** | Penalises large errors quadratically; sensitive to outliers |
| Regression, robust | **MAE / L1**, **Huber** | Huber = quadratic near zero, linear in the tails |
| Embeddings / retrieval | **Contrastive**, **triplet**, **InfoNCE** | Pull positives together, push negatives apart |

## Training: the loop

1. **Forward pass** — compute predictions.
2. **Compute loss** — compare to targets.
3. **Backward pass (backpropagation)** — apply the chain rule backwards through the
   graph to get $\partial L/\partial w$ for every parameter.
4. **Optimizer step** — update weights in the direction that lowers the loss.
5. Repeat over batches; one pass over the whole dataset is an **epoch**.

**Backpropagation** is *not* a learning algorithm — it is an efficient algorithm for
computing gradients. Gradient descent is what actually learns.

**Gradient descent variants**

- **Batch** — gradient over the entire dataset. Stable, memory-hungry, slow.
- **Stochastic (SGD)** — one sample at a time. Noisy, can escape shallow minima.
- **Mini-batch** — the practical default (batch sizes 8–1024+). Best hardware
  utilisation.

**Optimizers**

| Optimizer | Idea |
| --- | --- |
| **SGD** | `w ← w − η ∇L` |
| **SGD + momentum** | Accumulate a velocity term to damp oscillation |
| **RMSProp** | Per-parameter adaptive learning rate |
| **Adam** | Momentum + RMSProp; the general-purpose default |
| **AdamW** | Adam with *decoupled* weight decay — **the standard for training transformers** |

**Learning rate** is the most important hyperparameter. Too high → divergence or
oscillation; too low → glacial training and stalling in poor minima. Standard practice:
**warmup** (ramp up over the first few thousand steps) followed by **cosine decay**.

## Regularization — fighting overfitting

- **L2 / weight decay** — penalise large weights; shrinks them smoothly.
- **L1** — penalise absolute weights; drives some to exactly zero (feature selection).
- **Dropout** — randomly zero a fraction *p* of activations during training only;
  forces redundant representations. Disabled at inference.
- **Early stopping** — halt when validation loss stops improving; keep the best checkpoint.
- **Data augmentation** — synthesise new training examples (images: crop/flip/rotate;
  text: back-translation, synonym replacement, paraphrasing). Named in NVIDIA's course
  objectives.
- **Batch / layer normalization** — normalise activations to stabilise and speed up
  training. **Transformers use layer normalization**, not batch norm, because it is
  independent of batch size and works with variable-length sequences.

## Vanishing and exploding gradients

In deep or recurrent networks, repeated multiplication of small (<1) gradients shrinks
them toward zero — earlier layers stop learning. Repeated multiplication of large (>1)
gradients blows them up.

Mitigations: **ReLU-family activations**, **residual/skip connections**, **normalization
layers**, careful **initialization** (Xavier/Glorot for tanh, He for ReLU), and
**gradient clipping** for the exploding case.

Residual connections (`output = F(x) + x`) are what make 100-layer networks trainable —
and they are a core component of every transformer block.

## Transfer learning

Named repeatedly in NVIDIA's course objectives: *"leverage transfer learning between
models to achieve efficient results with less data and computation."*

Take a model pretrained on a large general corpus and adapt it to your task:

1. **Feature extraction** — freeze the backbone, train only a new head. Fast, needs
   little data, lowest risk of overfitting.
2. **Fine-tuning** — unfreeze some or all layers and continue training at a **low
   learning rate**.
3. **Parameter-efficient fine-tuning (PEFT/LoRA)** — the LLM-era answer; see
   [Customization](../domain-2/customization.md).

**Catastrophic forgetting** is the risk: aggressive fine-tuning on a narrow dataset
destroys the general capability the model came with. Low learning rates, fewer epochs,
mixing in general data, or PEFT all mitigate it.

The entire foundation-model paradigm *is* transfer learning at scale: pretrain once,
adapt many times.

## Key takeaways

- Activations provide non-linearity; ReLU is the classic default, **GELU/SwiGLU** power
  transformers, softmax produces probability distributions.
- Backpropagation computes gradients; the optimizer (**AdamW** for transformers) applies
  them.
- Cross-entropy is the loss for classification and for language modeling; MSE for
  regression.
- Dropout, weight decay, early stopping and data augmentation combat overfitting.
- Residual connections + layer normalization are what make deep transformers trainable.
- Transfer learning gets strong results with far less data and compute than training
  from scratch.
