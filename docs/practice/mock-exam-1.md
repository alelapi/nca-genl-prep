# Mock Exam 1

**60 questions · 60 minutes · no notes.** Write your answers down before expanding any.

---

## Domain 1 — Core ML & AI (questions 1–18)

**1.** Which component of the transformer architecture makes it possible to process all
tokens in a sequence in parallel?

- A. Recurrent connections
- B. Self-attention
- C. Convolutional filters
- D. Positional encoding

??? success "Answer"
    **B.** Attention computes relationships between all positions simultaneously, unlike an
    RNN's sequential recurrence. Positional encoding (D) is needed *because* of this
    parallelism, not the cause of it.

**2.** A pretrained LLM must answer questions about documents that change daily. The team
has no training budget. The correct approach is:

- A. Continued pretraining
- B. Full fine-tuning nightly
- C. RAG over an index refreshed daily
- D. Few-shot prompting with 3 examples

??? success "Answer"
    **C.** Fresh, private, factual knowledge → retrieval. Index updates are a database
    write, not a training run.

**3.** In `softmax(QKᵀ/√d_k)V`, the `V` matrix contains:

- A. The query vectors
- B. The values that get weighted and summed to form the output
- C. The vocabulary embeddings
- D. The positional encodings

??? success "Answer"
    **B.** Q asks, K advertises, and V is the content actually aggregated using the
    attention weights.

**4.** Which model type is best suited to producing embeddings for semantic search?

- A. Decoder-only with causal attention
- B. Encoder-only with bidirectional attention
- C. A diffusion model
- D. A GAN

??? success "Answer"
    **B.** Bidirectional context yields better whole-text representations.

**5.** Which is TRUE of Word2Vec?

- A. It produces a different vector for a word depending on its sentence
- B. It produces one static vector per word
- C. It uses multi-head attention
- D. It requires labeled training data

??? success "Answer"
    **B.** Static embeddings. Contextual vectors came with ELMo/BERT.

**6.** A team observes 97% training accuracy and 64% validation accuracy. The first fix to
try is:

- A. Increase model depth
- B. Increase the learning rate
- C. Add regularization and/or more training data
- D. Train for more epochs

??? success "Answer"
    **C.** Classic overfitting. Options A and D make it worse.

**7.** What does the `√d_k` scaling factor prevent?

- A. Gradient explosion in the FFN
- B. Softmax saturation and vanishing gradients as dimension grows
- C. Positional information from being lost
- D. Overfitting to the training corpus

??? success "Answer"
    **B.**

**8.** Which best describes in-context learning?

- A. Fine-tuning at inference time
- B. Performing a task from examples in the prompt with no weight updates
- C. Continual pretraining on user data
- D. Caching key/value tensors

??? success "Answer"
    **B.**

**9.** For a RAG system whose users search by exact SKU codes but get poor results, the
best fix is:

- A. Larger chunks
- B. Hybrid search combining dense embeddings with BM25
- C. Higher temperature
- D. A larger generator model

??? success "Answer"
    **B.** Dense retrieval is weak on rare exact tokens.

**10.** Which activation function is standard inside modern transformer feed-forward
layers?

- A. Sigmoid
- B. Tanh
- C. GELU or SwiGLU
- D. Softmax

??? success "Answer"
    **C.** ReLU is the classic default in deep learning generally; transformers use
    GELU/SwiGLU. Softmax is an output/attention function, not a hidden activation.

**11.** Which normalization do transformer blocks use, and why?

- A. Batch normalization, for speed
- B. Layer normalization, because it is batch-size independent and handles variable-length sequences
- C. No normalization
- D. Group normalization, for images

??? success "Answer"
    **B.**

**12.** In embedding-based retrieval, the standard similarity metric for text is:

- A. Euclidean distance
- B. Cosine similarity
- C. Jaccard index
- D. Hamming distance

??? success "Answer"
    **B.** Document length inflates magnitude but not direction. On L2-normalized vectors,
    cosine, dot and Euclidean rank identically.

**13.** Chunks in a RAG index are set to 1,500 tokens each. The likely consequence is:

- A. Retrieved chunks lack context
- B. Embeddings become diluted averages of several topics, hurting precision
- C. The index cannot be built
- D. Recall becomes perfect

??? success "Answer"
    **B.** The opposite failure from chunks that are too small.

**14.** Which of these is NOT a decoder-only model?

- A. GPT-2
- B. Llama
- C. BERT
- D. Mistral

??? success "Answer"
    **C.** BERT is encoder-only, trained with masked language modeling.

**15.** Chain-of-thought prompting most improves performance on:

- A. Simple binary sentiment classification
- B. Multi-step arithmetic and logical reasoning
- C. Embedding generation
- D. Tokenization

??? success "Answer"
    **B.** On simple classification it adds cost and latency for no gain.

**16.** Which parameter should be set low for factual, structured extraction tasks?

- A. Max tokens
- B. Temperature
- C. Top-k
- D. Frequency penalty

??? success "Answer"
    **B.** Low temperature → near-deterministic, high-probability output.

**17.** Which Python library provides tokenization, POS tagging, dependency parsing,
lemmatization and NER in one production pipeline?

- A. NumPy
- B. spaCy
- C. Gensim
- D. XGBoost

??? success "Answer"
    **B.**

**18.** The Chinchilla scaling result showed that, for a fixed compute budget:

- A. Models should be as large as possible
- B. Models were undertrained; ~20 tokens per parameter is compute-optimal
- C. Data quality does not matter
- D. Inference cost dominates training cost

??? success "Answer"
    **B.**

---

## Domain 2 — Software Development (questions 19–32)

**19.** Which NVIDIA component serves models in production with dynamic batching, model
versioning and Prometheus metrics?

- A. TensorRT
- B. Triton Inference Server
- C. NeMo Curator
- D. cuML

??? success "Answer"
    **B.** TensorRT *optimizes*; Triton *serves*.

**20.** A 13B model must run in INT8. Approximate weight memory:

- A. ~6.5 GB
- B. ~13 GB
- C. ~26 GB
- D. ~52 GB

??? success "Answer"
    **B.** INT8 = 1 byte/param → ~13 GB, plus KV cache and overhead.

**21.** LoRA's chief advantage at inference time is that:

- A. It reduces the number of layers
- B. `BA` can be merged into `W`, adding no latency
- C. It quantizes the base model automatically
- D. It removes the need for a tokenizer

??? success "Answer"
    **B.** Adapter *layers* add latency; LoRA does not, once merged. (C describes QLoRA.)

**22.** Which parallelism splits a model across GPUs *by layer*?

- A. Data parallelism
- B. Tensor parallelism
- C. Pipeline parallelism
- D. Expert parallelism

??? success "Answer"
    **C.** Tensor parallelism splits *within* layers.

**23.** NCCL is:

- A. A quantization format
- B. NVIDIA's library for multi-GPU collective communication
- C. A vector database
- D. A tokenizer

??? success "Answer"
    **B.**

**24.** Continuous (in-flight) batching improves throughput by:

- A. Compressing weights
- B. Adding new requests to the running batch as sequences finish
- C. Skipping the attention layers
- D. Caching identical prompts

??? success "Answer"
    **B.**

**25.** A team needs a 70B model on two 80 GB GPUs with acceptable quality. The most
appropriate approach is:

- A. Prune 50% of the layers
- B. Serve in FP16 across both GPUs with tensor parallelism
- C. Train a new smaller model
- D. Run on CPU

??? success "Answer"
    **B.** 70B FP16 ≈ 140 GB — fits across 2×80 GB with tensor parallelism, leaving room
    for the KV cache.

**26.** ONNX exists primarily to:

- A. Quantize models
- B. Provide an open interchange format between frameworks and runtimes
- C. Serve models over gRPC
- D. Orchestrate training jobs

??? success "Answer"
    **B.**

**27.** Which technique reduces latency without changing the output distribution?

- A. Reducing max tokens
- B. Speculative decoding
- C. Raising temperature
- D. Pruning

??? success "Answer"
    **B.**

**28.** Activation checkpointing saves memory by:

- A. Storing activations on disk
- B. Recomputing activations during the backward pass instead of storing them
- C. Quantizing activations to INT4
- D. Skipping the backward pass

??? success "Answer"
    **B.** Costs ~30% extra compute.

**29.** The KV cache primarily:

- A. Stores model weights
- B. Avoids recomputing keys and values for previous tokens during decoding
- C. Caches HTTP responses
- D. Holds the tokenizer vocabulary

??? success "Answer"
    **B.**

**30.** Which is the correct order of the NVIDIA workflow?

- A. Triton → TensorRT-LLM → NeMo
- B. NeMo (build/customize) → TensorRT-LLM (optimize) → Triton (serve)
- C. TensorRT-LLM → NeMo → cuDF
- D. NIM → NeMo Curator → Megatron

??? success "Answer"
    **B.**

**31.** A model must be validated against real production traffic with zero user risk. Use:

- A. Canary release
- B. Blue/green deployment
- C. Shadow deployment (traffic mirroring)
- D. Feature flag

??? success "Answer"
    **C.**

**32.** The main reason LLM inference is memory-bandwidth bound is:

- A. Tokenization is slow
- B. Each generated token requires reading the full weight set from HBM
- C. The softmax is expensive
- D. Networks are the bottleneck

??? success "Answer"
    **B.**

---

## Domain 3 — Experimentation (questions 33–45)

**33.** ROUGE is primarily used for:

- A. Machine translation
- B. Summarization
- C. Embedding quality
- D. Image generation

??? success "Answer"
    **B.** Recall-oriented. BLEU is precision-oriented and used for translation.

**34.** Perplexity measures:

- A. Factual accuracy
- B. How well a language model predicts a sample; lower is better
- C. Retrieval recall
- D. Latency per token

??? success "Answer"
    **B.**

**35.** In RLHF, the reward model is trained on:

- A. Absolute quality scores
- B. Human rankings/comparisons of candidate responses
- C. Ground-truth answers
- D. Token probabilities

??? success "Answer"
    **B.**

**36.** DPO differs from RLHF in that it:

- A. Requires no preference data
- B. Removes the separate reward model and the PPO loop
- C. Replaces supervised fine-tuning
- D. Works only on encoder models

??? success "Answer"
    **B.**

**37.** Which metric best indicates whether a RAG system is hallucinating?

- A. Recall@k
- B. Faithfulness/groundedness
- C. Perplexity
- D. BLEU

??? success "Answer"
    **B.**

**38.** Which is a valid criticism of stopping an A/B test the moment p < 0.05?

- A. p-values are invalid for A/B tests
- B. Repeated peeking inflates the false-positive rate well above α
- C. It requires too much traffic
- D. It only applies to regression problems

??? success "Answer"
    **B.**

**39.** A model achieves 99% accuracy detecting a condition present in 1% of cases. This
most likely indicates:

- A. An excellent model
- B. The model may be predicting the majority class; check recall and PR-AUC
- C. Data leakage is impossible
- D. The threshold is too low

??? success "Answer"
    **B.**

**40.** Which cross-validation strategy suits data with repeated measurements from the same
patients?

- A. Standard k-fold
- B. Stratified k-fold
- C. Group k-fold
- D. Leave-one-out

??? success "Answer"
    **C.** Keeping each patient entirely within one fold prevents leakage across folds.

**41.** GLUE is:

- A. A GPU library
- B. A benchmark suite of NLU tasks
- C. A quantization method
- D. A vector index

??? success "Answer"
    **B.**

**42.** Which is a known bias of LLM-as-a-judge?

- A. It always underrates long answers
- B. Position bias — it favours whichever response is presented first
- C. It cannot process JSON
- D. It requires reference answers

??? success "Answer"
    **B.** (It also *over*rates long answers — verbosity bias, the opposite of A.)

**43.** Few-shot evaluation results vary noticeably between runs. The best response is:

- A. Report the highest score achieved
- B. Vary example order and selection, and report mean ± standard deviation
- C. Increase temperature
- D. Use fewer examples

??? success "Answer"
    **B.**

**44.** "Negative rejection" in RAG evaluation tests whether the system:

- A. Rejects toxic inputs
- B. Says it does not know when the context lacks the answer
- C. Filters negative sentiment
- D. Rejects duplicate documents

??? success "Answer"
    **B.**

**45.** Inter-annotator agreement of κ = 0.18 most likely means:

- A. Annotators are careless
- B. The annotation guideline is ambiguous and the task is underspecified
- C. The dataset is too large
- D. The model will still learn fine

??? success "Answer"
    **B.**

---

## Domain 4 — Data Analysis (questions 46–54)

**46.** cuGraph is the GPU-accelerated equivalent of:

- A. pandas
- B. scikit-learn
- C. NetworkX
- D. Matplotlib

??? success "Answer"
    **C.**

**47.** "studies" → "studi" is the result of:

- A. Lemmatization
- B. Stemming
- C. Tokenization
- D. Normalization

??? success "Answer"
    **B.** A lemmatizer would return "study".

**48.** R² of −0.2 on held-out data means:

- A. The calculation is wrong
- B. The model performs worse than simply predicting the mean
- C. The model explains 20% of variance
- D. There is perfect negative correlation

??? success "Answer"
    **B.**

**49.** Which chart shows the distribution of a single continuous variable?

- A. Scatter plot
- B. Histogram
- C. Line chart
- D. Heatmap

??? success "Answer"
    **B.**

**50.** A dataset's mean is far above its median. This indicates:

- A. Left skew
- B. Right skew
- C. Normal distribution
- D. Bimodality

??? success "Answer"
    **B.**

**51.** Which loss is most robust to outliers?

- A. MSE
- B. MAE
- C. Cross-entropy
- D. Hinge

??? success "Answer"
    **B.**

**52.** t-SNE output should be used for:

- A. Model input features
- B. Visualization only
- C. Measuring cluster distances precisely
- D. Dimensionality reduction before regression

??? success "Answer"
    **B.** Use PCA if you need reduction for modeling.

**53.** `cudf.pandas` provides:

- A. A new DataFrame API to learn
- B. Zero-code-change GPU acceleration for existing pandas scripts, with CPU fallback
- C. Distributed training
- D. Model quantization

??? success "Answer"
    **B.**

**54.** XGBoost is best characterised as:

- A. Bagging — parallel independent trees
- B. Boosting — sequential trees, each correcting prior errors
- C. A transformer variant
- D. A clustering algorithm

??? success "Answer"
    **B.**

---

## Domain 5 — Trustworthy AI (questions 55–60)

**55.** NVIDIA's Trustworthy AI pillars include:

- A. Speed, cost, scale, accuracy
- B. Privacy, safety and security, transparency, non-discrimination
- C. Precision, recall, F1, AUC
- D. Build, optimize, serve, monitor

??? success "Answer"
    **B.**

**56.** Removing the protected attribute from training data:

- A. Guarantees a fair model
- B. Does not remove bias, because proxy variables reconstruct it
- C. Is required by GDPR
- D. Improves accuracy

??? success "Answer"
    **B.**

**57.** A malicious instruction embedded in a retrieved document is:

- A. A jailbreak
- B. Indirect prompt injection
- C. Data poisoning
- D. Model theft

??? success "Answer"
    **B.**

**58.** Which NVIDIA tool adds input, dialog, retrieval, execution and output rails to an
LLM application?

- A. NeMo Curator
- B. NeMo Guardrails
- C. TensorRT-LLM
- D. Triton

??? success "Answer"
    **B.**

**59.** Federated learning protects privacy by:

- A. Encrypting the model weights
- B. Keeping raw data local and sharing only model updates
- C. Adding noise to the outputs
- D. Deleting data after training

??? success "Answer"
    **B.** (C describes differential privacy.)

**60.** The main technical driver of an LLM memorising and regurgitating training data is:

- A. High temperature
- B. Duplication of that content in the training corpus
- C. Small model size
- D. Missing layer normalization

??? success "Answer"
    **B.** Which is why deduplication is the primary defence.

---

## Grade yourself

| Domain | Questions | Score | Target |
| --- | --- | --- | --- |
| 1. Core ML & AI | 1–18 | ___ / 18 | 15+ |
| 2. Software Development | 19–32 | ___ / 14 | 11+ |
| 3. Experimentation | 33–45 | ___ / 13 | 10+ |
| 4. Data Analysis | 46–54 | ___ / 9 | 7+ |
| 5. Trustworthy AI | 55–60 | ___ / 6 | 5+ |
| **Total** | | **___ / 60** | **52+** |

Now do the review: for every miss, name the concept, re-read its page, and write one line
on why each distractor was wrong. Then take **[Mock Exam 2](mock-exam-2.md)**.
