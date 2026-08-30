# Glossary

Every term you should be able to define on sight. Grouped by theme; skim the whole page the
day before the exam.

## Core concepts

**Artificial intelligence** — systems performing tasks that normally require human
intelligence.

**Machine learning** — algorithms that learn patterns from data rather than being
explicitly programmed.

**Deep learning** — machine learning with multi-layer neural networks.

**Generative AI** — models that produce new content: text, images, audio, code, video.

**Foundation model** — a large model trained with self-supervision on broad data, designed
to be adapted to many downstream tasks.

**LLM (large language model)** — a foundation model for text.

**Supervised / unsupervised / reinforcement learning** — learning from labeled data / from
unlabeled data / from environment rewards.

**Self-supervised learning** — labels derived from the data itself; how LLMs are pretrained.

**Transfer learning** — reusing a pretrained model as the starting point for a new task.

**Emergent ability** — a capability (in-context learning, chain-of-thought) that appears
only past a certain model scale.

## Transformers

**Attention** — mechanism letting every token weight every other token: `softmax(QKᵀ/√d_k)V`.

**Query / Key / Value** — what a token seeks / advertises / contributes.

**Multi-head attention** — several attention functions computed in parallel over different
projections.

**Causal (masked) attention** — a decoder mask preventing a token from attending to future
positions.

**Cross-attention** — decoder queries attending to encoder keys and values.

**Positional encoding** — order information added to embeddings; sinusoidal, learned, RoPE,
ALiBi.

**Layer normalization** — normalisation across features within a token; used in transformers
(pre-LN in modern models).

**Residual (skip) connection** — `F(x) + x`; lets gradients flow through deep stacks.

**Feed-forward network (FFN)** — position-wise two-layer MLP with inner dimension ≈ 4× `d_model`.

**Encoder-only / decoder-only / encoder–decoder** — understanding & embeddings /
generation / transduction.

**Tokenization** — splitting text into subword units. **BPE** (GPT), **WordPiece** (BERT),
**SentencePiece/Unigram** (multilingual).

**KV cache** — cached keys/values of past tokens; makes decoding O(n) instead of O(n²).

**GQA / MQA** — grouped-/multi-query attention; heads share K/V, shrinking the KV cache.

**FlashAttention** — IO-aware exact attention kernel.

**Mixture of Experts (MoE)** — many expert FFNs with per-token routing; large total, small
active parameter count.

## Embeddings & retrieval

**Embedding** — a dense vector representing content, where geometric closeness means
semantic similarity.

**Static vs. contextual embedding** — Word2Vec/GloVe (one vector per word) vs. BERT (vector
depends on context).

**Cosine similarity** — angle-based similarity; the default for text. Equals dot product on
normalized vectors.

**Vector database** — stores embeddings and answers nearest-neighbour queries. FAISS,
Milvus, Qdrant, Weaviate, Chroma, pgvector.

**ANN (approximate nearest neighbour)** — trades exactness for speed. **HNSW**, **IVF**,
**PQ**, **Flat**.

**Chunking** — splitting documents into retrievable units; size trades context against
precision, overlap prevents cutting answers in half.

**Bi-encoder** — embeds query and document separately; fast, precomputable.

**Cross-encoder** — scores query and passage jointly; accurate, slow; used for **reranking**.

**Hybrid search** — dense + sparse (BM25) retrieval, fused (Reciprocal Rank Fusion).

**RAG (retrieval-augmented generation)** — grounding generation in retrieved documents.

**HyDE** — embed an LLM-drafted hypothetical answer as the search query.

**GraphRAG** — retrieval over a knowledge graph rather than flat chunks.

**Grounding** — constraining answers to supplied evidence.

## Prompting

**Prompt engineering** — designing inputs to elicit desired outputs, without weight updates.

**Zero-/one-/few-shot** — 0 / 1 / a handful of examples in the prompt.

**In-context learning** — performing a task from prompt examples with no gradient updates.

**Chain-of-thought (CoT)** — prompting for step-by-step reasoning.

**Self-consistency** — sample several reasoning paths, take the majority answer.

**ReAct** — interleaved reasoning and tool use.

**System prompt** — persistent instructions and policy for a conversation.

**Soft prompt** — learned embedding vectors (P-tuning/prompt tuning) with no textual form.

**Temperature** — scales logits before softmax; 0 = deterministic, higher = more random.

**Top-k / top-p (nucleus)** — restrict sampling to the *k* most likely tokens / the smallest
set with cumulative probability ≥ *p*.

**Prompt injection** — hidden instructions in input or retrieved content hijacking the model.
**Indirect** injection arrives via retrieved documents.

**Jailbreak** — a prompt that bypasses safety alignment.

## Training & customization

**Pretraining** — self-supervised training on a large corpus; produces a base model.

**SFT (supervised fine-tuning) / instruction tuning** — training on (instruction, response)
pairs.

**Alignment** — shaping behaviour toward human preferences (helpful, harmless, honest).

**RLHF** — SFT → reward model trained on human **rankings** → PPO with a **KL penalty**.

**DPO** — direct preference optimization; no reward model, no RL loop.

**RLAIF / Constitutional AI** — AI-generated preference labels / self-critique against
written principles.

**PEFT** — parameter-efficient fine-tuning: LoRA, QLoRA, adapters, prefix/prompt tuning, IA³.

**LoRA** — freeze `W`, learn a low-rank update `BA` of rank `r`; ~0.1–1% of parameters;
merges into `W` with **no added inference latency**.

**QLoRA** — LoRA on top of a 4-bit quantized frozen base.

**Catastrophic forgetting** — loss of general capability from aggressive narrow fine-tuning.

**Knowledge distillation** — training a small student on a large teacher's soft outputs.

**Backpropagation** — the chain-rule algorithm for computing gradients.

**Optimizer** — SGD, momentum, RMSProp, **Adam**, **AdamW** (the transformer standard).

**Learning rate warmup / cosine decay** — the standard LLM schedule.

**Regularization** — L1/L2 (weight decay), dropout, early stopping, data augmentation.

**Overfitting / underfitting** — low training error with high validation error / high error
on both.

**Bias–variance trade-off** — capacity lowers bias and raises variance.

**Scaling laws** — loss falls as a power law in parameters, data and compute.
**Chinchilla**: ~20 training tokens per parameter is compute-optimal.

## Inference & deployment

**Prefill / decode** — parallel processing of the prompt (compute-bound) / autoregressive
token generation (memory-bandwidth-bound).

**TTFT / TPOT** — time to first token / time per output token.

**Latency vs. throughput** — per-request speed vs. aggregate tokens per second.

**Quantization** — reducing numeric precision: FP32, TF32, FP16, BF16, FP8, INT8, INT4.

**PTQ / QAT** — post-training quantization (calibration only) / quantization-aware training
(simulated during training, recovers accuracy).

**Mixed precision** — FP16/BF16 compute with an FP32 master copy and loss scaling.

**Static / dynamic / continuous (in-flight) batching** — fixed batches / time-windowed
grouping / token-level scheduling that inserts new requests as sequences finish.

**PagedAttention** — block-based KV cache management (vLLM).

**Speculative decoding** — a draft model proposes tokens the large model verifies in one
pass; same output distribution, lower latency.

**Pruning** — removing weights, heads or layers.

**ONNX** — open interchange format for models across frameworks and runtimes.

**Model drift** — **data drift** (input distribution shifts) and **concept drift** (the
input→output relationship changes). Both are silent failures.

**Canary / blue-green / shadow deployment** — small traffic slice / atomic switch with
instant rollback / mirrored traffic with discarded responses.

## Distributed training

**Data parallelism** — replicate the model, split the data, **AllReduce** the gradients.

**Tensor parallelism** — split individual weight matrices across GPUs; high-frequency
communication; keep inside a node.

**Pipeline parallelism** — assign different layers to different GPUs; low communication;
suffers **pipeline bubbles**, mitigated with micro-batches.

**3D parallelism** — data × tensor × pipeline combined.

**AllReduce** — collective that combines values from all ranks and returns the result to all.

**Ring-AllReduce** — ReduceScatter + AllGather around a ring; per-GPU communication volume
independent of GPU count.

**NCCL** — NVIDIA Collective Communications Library.

**ZeRO / FSDP** — shard optimizer states (1), gradients (2) and parameters (3) across
data-parallel ranks.

**Activation checkpointing** — recompute activations in the backward pass to save memory.

**Gradient accumulation** — several micro-batches per optimizer step.

## Evaluation

**Accuracy / precision / recall / F1** — fraction correct / of flagged, how many right / of
existing, how many found / their harmonic mean.

**ROC-AUC / PR-AUC** — threshold-independent ranking quality; PR-AUC is the honest one on
imbalanced data.

**Perplexity** — exponentiated mean cross-entropy; lower is better; measures fluency, not
truth.

**BLEU** — n-gram **precision**, machine **translation**, with a brevity penalty.

**ROUGE** — n-gram **recall**, **summarization**; ROUGE-N, ROUGE-L, ROUGE-S.

**METEOR** — matching with stems/synonyms plus word-order penalty.

**BERTScore** — embedding-based semantic similarity; credits paraphrase.

**pass@k** — code generation metric: does any of *k* samples pass the tests.

**Recall@k / precision@k / MRR / nDCG** — retrieval metrics: found it / of the top *k*, how
many relevant / reciprocal rank of the first hit / position-discounted graded relevance.

**RAG triad** — context relevance, **faithfulness/groundedness**, answer relevance.

**Negative rejection** — correctly saying "I don't know" when the context lacks the answer.

**LLM-as-a-judge** — using a strong model to grade outputs; subject to **position**,
**verbosity**, **self-enhancement** and **formatting** biases.

**Benchmarks** — GLUE, SuperGLUE, SQuAD, MMLU, HellaSwag, GSM8K, HumanEval, TruthfulQA,
BIG-bench, HELM, MTEB, Chatbot Arena.

**Contamination** — benchmark data present in the training corpus, inflating scores.

**Cross-validation** — k-fold, **stratified** (classification), **group** (correlated rows),
**time-series** (never shuffle), nested.

**Data leakage** — information from validation/test reaching training.

**A/B test** — randomised controlled online experiment.

**p-value** — probability of a result at least this extreme **if the null were true**.

**Type I / Type II error** — false positive / false negative.

**Statistical power** — probability of detecting a real effect; target 80%.

**Guardrail metric** — a secondary metric (latency, cost, error rate) that must not regress.

**Interleaving** — blending two ranking systems' results in one list; very traffic-efficient.

**Inter-annotator agreement** — Cohen's κ (2 raters), Fleiss' κ (3+), Krippendorff's α.

**Hallucination** — fluent, confident, wrong output. **Factual** (wrong about the world) vs.
**faithfulness** (unsupported by the provided context).

## Data & statistics

**Feature engineering** — one-hot / ordinal / target encoding, scaling, standardization,
binning, log transforms.

**TF-IDF** — term frequency × inverse document frequency; distinctive terms score highest.

**Bag of words** — raw token counts; order-blind.

**Stemming** — rule-based affix chopping; fast; **may not produce a real word**.

**Lemmatization** — dictionary and POS-aware; **always produces a real word**.

**NER** — named entity recognition.

**PCA** — linear dimensionality reduction preserving variance; usable as model input.

**t-SNE / UMAP** — non-linear projections **for visualization only**.

**Pearson / Spearman / Kendall** — linear correlation / monotonic (rank) correlation / rank
concordance.

**Simpson's paradox** — a trend that reverses when subgroups are pooled.

**Multicollinearity** — highly correlated predictors making coefficients unstable.

**R²** — proportion of explained variance; **adjusted R²** penalises extra predictors.

**MSE / RMSE / MAE / Huber** — squared error (outlier-sensitive) / same units / absolute
error (robust) / the compromise.

**SMOTE** — synthetic minority oversampling; apply to the **training set only**.

**Back-translation** — text augmentation by translating out and back.

**IQR** — interquartile range; robust spread; basis of the box plot.

**Central limit theorem** — the sample mean tends to normality as *n* grows.

## NVIDIA stack

**CUDA** — NVIDIA's parallel computing platform.

**Tensor Cores** — matrix-multiply units delivering FP16/BF16/FP8/INT8 acceleration.

**NVLink / NVSwitch** — high-bandwidth GPU-to-GPU interconnect.

**MIG (Multi-Instance GPU)** — partitioning one GPU into isolated instances.

**TensorRT / TensorRT-LLM** — inference **optimization**: layer fusion, kernel auto-tuning,
precision calibration; TensorRT-LLM adds in-flight batching, paged KV cache, tensor
parallelism and LLM quantization.

**Triton Inference Server** — production **serving**: multi-framework, dynamic batching,
versioning, HTTP/gRPC, Prometheus metrics.

**NIM (NVIDIA Inference Microservices)** — containerised optimized model endpoints with a
standard API, deployable in your own environment.

**NeMo Framework** — build, customize, align, evaluate and export generative AI models.

**NeMo Curator** — large-scale data curation: extraction, filtering, deduplication, PII
removal.

**NeMo Guardrails** — programmable **input, dialog, retrieval, execution and output rails**,
defined in Colang, outside the model.

**NeMo Retriever** — embedding and reranking microservices for RAG.

**NeMo Evaluator** — model and RAG evaluation.

**Megatron-LM** — large-scale distributed training (tensor and pipeline parallelism).

**RAPIDS** — GPU data science: **cuDF** (pandas), **cuML** (scikit-learn), **cuGraph**
(NetworkX), **CuPy** (NumPy), **cuVS/RAFT** (vector search), **cuxfilter** (dashboards).

**DCGM** — Data Center GPU Manager; the metrics exporter behind GPU monitoring.

**NGC** — NVIDIA GPU Cloud: containers, models and Helm charts.

## Trustworthy AI

**Trustworthy AI pillars (NVIDIA)** — privacy, safety and security, transparency,
non-discrimination.

**Transparency vs. explainability** — disclosure about the **system** vs. explaining a
**specific decision**.

**Model card** — intended use, out-of-scope uses, training data summary, evaluation results
**disaggregated by subgroup**, limitations.

**Datasheet for a dataset** — provenance, collection method, consent basis, composition,
known biases.

**Historical bias** — accurate data encoding past injustice.

**Representation bias** — under-represented groups suffering worse performance.

**Fairness definitions** — demographic parity, equal opportunity, equalized odds, predictive
parity, individual and counterfactual fairness. **Mutually incompatible** when base rates
differ.

**Disaggregated evaluation** — reporting metrics per subgroup; the highest-value fairness
practice.

**PII** — personally identifiable information.

**Data minimisation / purpose limitation / storage limitation** — collect only what is
needed, use only for the stated purpose, keep only as long as needed.

**Right to erasure** — deletion right; nearly impossible to honour in trained weights, hence
a strong argument for RAG over fine-tuning with personal data.

**Differential privacy** — a formal guarantee that any one individual's data does not
meaningfully change the output.

**Federated learning** — training across sites where only **model updates** are shared.

**Memorisation / extraction** — verbatim reproduction of training data; driven mainly by
**duplication**, mitigated by deduplication.

**OWASP LLM Top 10** — prompt injection, insecure output handling, training data poisoning,
model DoS, supply chain, sensitive information disclosure, insecure plugin design, excessive
agency, overreliance, model theft.

**Red-teaming** — systematic adversarial testing; findings become permanent regression tests.

**Human-in-the-loop / on-the-loop / in-command** — approval per output / monitoring with
intervention / deciding whether the system runs at all.
