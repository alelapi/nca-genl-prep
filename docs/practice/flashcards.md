# Rapid-Fire Flashcards

The night-before review. Cover the right column, work down the page, and mark anything you
hesitate on.

## The decision reflexes

| Situation | Answer |
| --- | --- |
| Model lacks knowledge (private, recent, factual) | **RAG** |
| Model lacks correct behaviour (style, format, domain tone) | **Fine-tuning — PEFT/LoRA first** |
| Both | Fine-tune for behaviour **+** RAG for facts |
| No data, quick result needed | **Prompt engineering** (zero → few-shot → CoT) |
| Model won't fit in GPU memory | **Quantize**, then tensor/pipeline parallelism |
| Need higher server throughput | **Continuous (in-flight) batching**, TensorRT-LLM, Triton |
| Need lower perceived latency | **Stream tokens**; optimise TTFT |
| Need to compare two variants in production | **A/B test** with a pre-declared primary metric |
| Need verifiable, auditable answers | **RAG with citations** |
| Need to constrain LLM app behaviour (NVIDIA) | **NeMo Guardrails** |
| Accuracy lost to INT8 quantization | **Quantization-aware training (QAT)** |
| Retrieval misses exact product codes | **Hybrid search** (dense + BM25) |
| Right doc retrieved but ranked low | **Cross-encoder reranker** |

## NVIDIA product map

| Need | Product |
| --- | --- |
| Curate/dedupe/PII-strip a training corpus | **NeMo Curator** |
| Pretrain, fine-tune, align, evaluate a model | **NeMo Framework** |
| Large-scale distributed training | **Megatron-LM** |
| Optimize a model for inference | **TensorRT** / **TensorRT-LLM** |
| Serve models in production | **Triton Inference Server** |
| Prepackaged optimized endpoint, standard API | **NIM** |
| Safety/topic rails around an LLM app | **NeMo Guardrails** |
| Embeddings + reranking for RAG | **NeMo Retriever** |
| GPU dataframes / ML / graphs | **RAPIDS**: cuDF / cuML / cuGraph |
| Multi-GPU collective communication | **NCCL** |

## Transformers

| Prompt | Answer |
| --- | --- |
| Attention formula | `softmax(QKᵀ/√d_k)V` |
| Why `√d_k` | Prevents softmax saturation → vanishing gradients |
| Q, K, V | Query = what I seek; Key = what I contain; Value = what I contribute |
| Why positional encoding | Attention is permutation-invariant |
| Attention complexity | **O(n²)** in sequence length |
| Encoder-only | Bidirectional, MLM → classification, NER, **embeddings** (BERT) |
| Decoder-only | Causal, next-token → **generation** (GPT, Llama) |
| Encoder–decoder | Both + cross-attention → translation, seq2seq (T5) |
| Normalization used | **Layer norm** (pre-LN in modern models) |
| FFN inner dimension | **4 × d_model** |
| Transformer activation | **GELU / SwiGLU** |
| KV cache | Caches past K/V; O(n²) → O(n) decoding; often huge |
| GQA/MQA | Share K/V across heads → smaller KV cache |
| Token ≈ | 0.75 words ≈ 4 characters |

## Embeddings & RAG

| Prompt | Answer |
| --- | --- |
| Default text similarity metric | **Cosine** (== dot product on normalized vectors) |
| Word2Vec vs. BERT | Static one-vector-per-word vs. **contextual** |
| Cardinal RAG rule | **Same embedding model for index and query** |
| Chunk too small / too large | Missing context / diluted embedding + wasted context |
| RAG pipeline | chunk → embed → index → retrieve → rerank → augment → generate → cite |
| Bi-encoder vs. cross-encoder | Separate & precomputable (fast) vs. joint (accurate, slow) |
| ANN indexes | HNSW, IVF, PQ, Flat |
| Hybrid search fusion | **Reciprocal Rank Fusion** |
| HyDE | Embed a hypothetical *answer* to search with |

## Customization

| Prompt | Answer |
| --- | --- |
| Ladder | prompting → few-shot → RAG → PEFT → full FT → pretraining |
| LoRA | Freeze `W`, learn `ΔW = BA`, rank `r` = 4–64, ~0.1–1% params |
| LoRA at inference | `BA` **merges into `W`** → zero added latency |
| QLoRA | LoRA over a **4-bit quantized** frozen base |
| LoRA learning rate | ~1e-4 to 3e-4 (≈10× full FT) |
| Distillation | Student learns teacher's **soft output distributions** |
| PTQ vs. QAT | Calibrate after training vs. simulate quantization during training |
| Catastrophic forgetting | Aggressive fine-tuning destroys general ability |

## Memory arithmetic

| Prompt | Answer |
| --- | --- |
| FP16 inference weights | **~2 GB per billion parameters** |
| INT8 / INT4 | ~1 GB / ~0.5 GB per billion |
| Full FT state | **~16 bytes per parameter** (params + grads + FP32 master + Adam) |
| 70B in FP16 | ~140 GB → needs 2× 80 GB with tensor parallelism |
| Decoding is bound by | **Memory bandwidth**, not compute |
| Prefill is bound by | Compute |

## Distributed training

| Prompt | Answer |
| --- | --- |
| Data parallel | Replicate model, split data, **AllReduce** gradients |
| Tensor parallel | Split **within** a layer; chatty; keep inside a node (NVLink) |
| Pipeline parallel | Split **across** layers; cheap comms; watch the **bubble** |
| Ring-AllReduce | ReduceScatter + AllGather; comms volume independent of GPU count |
| ZeRO/FSDP stages | 1 optimizer states · 2 + gradients · 3 + parameters |
| Activation checkpointing | Recompute activations to save memory (~30% extra compute) |
| Mixed precision | FP16/BF16 compute + FP32 master + loss scaling |
| BF16 vs FP16 | BF16 = FP32 exponent range, fewer mantissa bits, more robust |

## Metrics

| Prompt | Answer |
| --- | --- |
| Precision | TP/(TP+FP) — of what I flagged, how much was right |
| Recall | TP/(TP+FN) — of what existed, how much I found |
| F1 | Harmonic mean of precision and recall |
| Accuracy fails when | Classes are **imbalanced** |
| Perplexity | exp(mean cross-entropy); **lower better**; fluency ≠ truth |
| BLEU | **Precision**, machine **translation** |
| ROUGE | **Recall**, **summarization** |
| BERTScore | Semantic similarity via embeddings — credits paraphrase |
| R² | **Proportion of explained variance**; can be negative |
| MSE vs. MAE | Punishes large errors vs. robust to outliers |
| Retrieval metrics | Recall@k (found it?), MRR/nDCG (ranked it well?) |
| RAG triad | Context relevance · **Faithfulness** · Answer relevance |
| Judge biases | **Position**, verbosity, self-enhancement, formatting |
| Benchmarks | GLUE/SuperGLUE (NLU) · MMLU (knowledge) · HELM (holistic) · MTEB (embeddings) · HumanEval (code, pass@k) |

## Experimentation

| Prompt | Answer |
| --- | --- |
| A/B test essentials | Randomise · consistent bucketing · pre-declared primary metric · guardrails · fixed sample size · **no peeking** |
| p-value | P(result this extreme \| null true) — **not** P(null true) |
| Type I / II | False positive / false negative |
| Power | P(detecting a real effect); target 80% |
| Cross-validation variants | Stratified (classification) · Group (correlated rows) · Time-series (never shuffle) |
| RLHF | SFT → reward model from **rankings** → PPO with **KL penalty** |
| DPO | Preference optimization **without** reward model or RL |
| IAA | Cohen's κ (2 raters) · Fleiss' κ (3+); low κ = bad guideline |
| Hallucination root cause | Objective optimises **plausibility, not truth** |
| Hallucination fix | RAG + grounding instruction + citations + low temperature |
| Not a hallucination fix | **Fine-tuning on more facts** |

## Data analysis

| Prompt | Answer |
| --- | --- |
| Stemming | Rule-based, fast, **may not be a real word** ("studies"→"studi") |
| Lemmatization | Dictionary + POS, **always a real word** ("studies"→"study") |
| TF-IDF | tf × log(N/df) — distinctive terms score highest |
| Histogram / scatter / box plot | One variable's distribution / relationship / spread + outliers |
| t-SNE & UMAP | **Visualization only**; cluster distances meaningless |
| PCA | Linear, preserves variance; usable as model input |
| Mean ≫ median | Right-skewed |
| Pearson vs. Spearman | Linear vs. monotonic/rank |
| r = 0 means | No **linear** relationship — not independence |
| Simpson's paradox | Trend reverses when subgroups are pooled |
| cuDF / cuML / cuGraph / CuPy | pandas / scikit-learn / NetworkX / NumPy on GPU |
| `cudf.pandas` | Zero-code-change GPU pandas, CPU fallback |
| XGBoost vs. random forest | **Boosting** (sequential, bias↓) vs. **bagging** (parallel, variance↓) |
| Back-translation | Text augmentation producing natural paraphrases |

## Trustworthy AI

| Prompt | Answer |
| --- | --- |
| NVIDIA pillars | Privacy · Safety & security · Transparency · Non-discrimination |
| Transparency vs. explainability | The system vs. a specific decision |
| Historical bias | Accurate data encoding past injustice |
| Removing the protected attribute | **Does not work** — proxies reconstruct it |
| Best fairness practice | **Disaggregated evaluation** (metrics per subgroup) |
| Fairness definitions | Demographic parity · equal opportunity · equalized odds · predictive parity — **mutually incompatible** |
| Consent must be | Informed · specific · freely given · **revocable** · documented |
| Right to erasure problem | Cannot remove influence from trained weights → prefer RAG |
| Memorisation driver | **Duplication** in training data → deduplicate |
| Differential privacy | Noise so no individual measurably changes the output |
| Federated learning | Data stays local; only **model updates** are shared |
| Prompt injection | Direct (user) vs. **indirect** (retrieved content) |
| Injection defence | Defence in depth — never a prompt alone |
| Never do | Pass model output unvalidated into shell / SQL / `eval()` / browser |

## Ten things people get wrong under time pressure

1. **BLEU = translation/precision, ROUGE = summarization/recall.** Not the reverse.
2. **Fine-tuning does not fix missing knowledge.** RAG does.
3. **Few-shot involves no training.**
4. **Accuracy is worthless on imbalanced data.**
5. **t-SNE is not a feature engineering step.**
6. **TensorRT optimizes; Triton serves.**
7. **Tensor parallelism splits within a layer; pipeline splits across layers.**
8. **Dropping the protected attribute does not remove bias.**
9. **A p-value is not the probability the hypothesis is true.**
10. **Absolutes ("always", "guarantees", "completely eliminates") are almost always wrong
    answers.**
