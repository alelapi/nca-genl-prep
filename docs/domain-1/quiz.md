# Domain 1 Quiz — Core ML & AI

20 exam-style questions. Answer before expanding. Target: **17/20**.

!!! note
    Original questions written to mirror NVIDIA's style. They are not real exam items.

---

**1.** In scaled dot-product attention, why is the dot product divided by `√d_k`?

- A. To normalise the output to unit length
- B. To prevent large dot products from saturating the softmax and vanishing the gradients
- C. To reduce the computational complexity from O(n²) to O(n log n)
- D. To ensure attention weights sum to 1

??? success "Answer"
    **B.** As `d_k` grows, dot products grow in magnitude, pushing softmax into a
    saturated region where gradients approach zero. The `√d_k` scaling keeps the variance
    of the scores roughly constant. Softmax — not the scaling — is what makes the weights
    sum to 1 (D), and the complexity is unchanged (C).

---

**2.** A team must build an assistant that answers questions about internal policy
documents updated weekly. Which approach is most appropriate?

- A. Pretrain a new LLM on the documents
- B. Fully fine-tune an open model each week
- C. Retrieval-augmented generation over an index of the documents
- D. Increase the temperature so the model explores more answers

??? success "Answer"
    **C.** Weekly-changing, private, factual knowledge is the textbook RAG case: update
    the index, not the weights. Fine-tuning (B) is expensive, cannot be cited, and goes
    stale immediately. Pretraining (A) is absurd at this scale. Temperature (D) is unrelated.

---

**3.** Which statement best distinguishes Word2Vec from BERT embeddings?

- A. Word2Vec is contextual; BERT is static
- B. Word2Vec produces one fixed vector per word; BERT's vectors depend on the surrounding sentence
- C. Word2Vec uses attention; BERT uses convolution
- D. They are identical apart from dimensionality

??? success "Answer"
    **B.** Word2Vec is a **static** embedding — "bank" gets one vector regardless of
    meaning. BERT is **contextual**: the representation is computed from the whole
    sentence, so *river bank* and *investment bank* differ.

---

**4.** Which architecture family is most appropriate for producing text embeddings for
semantic search?

- A. Decoder-only, causal attention
- B. Encoder-only, bidirectional attention
- C. Diffusion model
- D. Generative adversarial network

??? success "Answer"
    **B.** Encoder models see the whole sequence in both directions, which is what a good
    whole-text representation requires. Decoders (A) are for generation. C and D are not
    text-representation architectures.

---

**5.** Training accuracy is 98%; validation accuracy is 61%. What is happening, and what
should you try first?

- A. Underfitting — increase model capacity
- B. Overfitting — add regularization, dropout, or more data
- C. Data leakage — remove the validation set
- D. The learning rate is too low — increase it

??? success "Answer"
    **B.** A large train–validation gap is the signature of overfitting. Remedies:
    dropout, weight decay, early stopping, data augmentation, more data, or a simpler
    model. Increasing capacity (A) makes it worse.

---

**6.** What does k-fold cross-validation primarily provide over a single train/validation
split?

- A. Faster training
- B. A lower-variance estimate of performance, plus a measure of its variability
- C. Elimination of overfitting
- D. Automatic hyperparameter selection

??? success "Answer"
    **B.** Averaging over *k* folds reduces dependence on one lucky or unlucky split and
    yields a standard deviation you can use to judge whether two models genuinely differ.
    It costs *k*× more compute, not less (A), and does not by itself remove overfitting (C).

---

**7.** Which chunking outcome is the most likely cause of a RAG system returning
fragments that lack enough context to answer?

- A. Chunks are too large
- B. Chunks are too small
- C. Overlap is too high
- D. The vector index uses HNSW

??? success "Answer"
    **B.** Small chunks match precisely but may not carry the surrounding context that
    makes the answer complete. Fixes: larger chunks, more overlap, or parent-document
    retrieval. Overly large chunks (A) cause the opposite problem — diluted embeddings and
    wasted context.

---

**8.** Which similarity metric is the conventional default for comparing text embeddings,
and why?

- A. Euclidean distance, because it accounts for magnitude
- B. Cosine similarity, because it measures angle and is insensitive to vector length
- C. Manhattan distance, because it is cheapest
- D. Jaccard similarity, because text is set-like

??? success "Answer"
    **B.** Document length inflates vector magnitude without changing topic; cosine
    compares direction only. Note that on **L2-normalized** vectors, cosine, dot product
    and Euclidean produce the same ranking.

---

**9.** A model must answer using a strict JSON schema, with minimal variation between
runs. Which decoding setting is most appropriate?

- A. Temperature 1.0, top-p 0.95
- B. Temperature 0 (or very low), with a stop sequence and schema validation
- C. Temperature 0.7 with a high frequency penalty
- D. Maximum top-k

??? success "Answer"
    **B.** Structured, factual output wants deterministic decoding. High temperature (A, C)
    is for creative diversity. Also validate the parsed JSON and retry on failure.

---

**10.** What is "in-context learning"?

- A. Fine-tuning the model on examples at inference time
- B. The model performing a task from examples in the prompt, with no weight updates
- C. Continual pretraining on a streaming corpus
- D. Caching key/value tensors across requests

??? success "Answer"
    **B.** Few-shot prompting works because large models infer the pattern from prompt
    examples. **No gradients, no weight changes.** D describes the KV cache.

---

**11.** Which is TRUE about positional encodings in transformers?

- A. They are unnecessary because attention preserves order
- B. They inject order information, because self-attention is permutation-invariant
- C. They replace the embedding layer
- D. They are only used in encoder-only models

??? success "Answer"
    **B.** Attention computes the same result regardless of token order, so position must
    be supplied explicitly — sinusoidal, learned absolute, RoPE or ALiBi.

---

**12.** Which Python library would you use to extract named entities from raw text in a
production pipeline?

- A. NumPy
- B. spaCy
- C. Matplotlib
- D. XGBoost

??? success "Answer"
    **B.** spaCy's default pipeline includes NER along with tokenization, POS tagging,
    dependency parsing and lemmatization, and is built for production throughput.

---

**13.** A team fine-tunes a model on 500 domain examples and finds it now performs poorly
on general tasks it previously handled. What is this called?

- A. Overfitting to the validation set
- B. Catastrophic forgetting
- C. Mode collapse
- D. Gradient explosion

??? success "Answer"
    **B.** Aggressive fine-tuning on a narrow dataset overwrites general capability.
    Mitigate with lower learning rates, fewer epochs, mixing in general data, or PEFT/LoRA
    which leaves base weights frozen.

---

**14.** According to the Chinchilla scaling result, for a fixed compute budget most large
models of that era were:

- A. Too small and trained on too much data
- B. Too large and trained on too little data
- C. Correctly sized
- D. Limited only by inference cost

??? success "Answer"
    **B.** Chinchilla showed models were **undertrained**: roughly 20 training tokens per
    parameter is compute-optimal. Smaller models trained longer beat larger, data-starved
    ones — and are cheaper to serve.

---

**15.** Which of the following is NOT a benefit of RAG over fine-tuning for factual
question answering?

- A. Knowledge can be updated without retraining
- B. Answers can carry citations to source documents
- C. Reduced per-query inference cost
- D. Private data can be used without embedding it in weights

??? success "Answer"
    **C.** RAG *increases* per-query cost — retrieval latency plus a much longer prompt.
    Its advantages are freshness, traceability, and keeping data out of the weights.

---

**16.** In the transformer block, what is the position-wise feed-forward network's inner
dimension, conventionally?

- A. Equal to `d_model`
- B. Roughly 4× `d_model`
- C. Equal to the vocabulary size
- D. Equal to the number of attention heads

??? success "Answer"
    **B.** The FFN expands to ~4× `d_model` and projects back. Most of a transformer's
    parameters live in these layers.

---

**17.** Which normalization is used inside transformer blocks, and why?

- A. Batch normalization, because it is faster
- B. Layer normalization, because it is independent of batch size and handles
  variable-length sequences
- C. Instance normalization, because sequences are images
- D. No normalization is used

??? success "Answer"
    **B.** Layer norm normalises across features within each token, so it works with any
    batch size and any sequence length. Modern models place it **before** the sub-layer
    (pre-LN) for training stability.

---

**18.** A retrieval system fails to find documents when users search by exact product
codes, though semantic queries work well. What is the best fix?

- A. Increase the embedding dimensionality
- B. Add hybrid search combining dense vectors with BM25 keyword matching
- C. Raise the temperature of the generator
- D. Reduce the chunk overlap

??? success "Answer"
    **B.** Dense embeddings are weak on rare exact tokens like identifiers. Hybrid search
    fuses semantic and lexical rankings (commonly via reciprocal rank fusion) and covers
    both cases.

---

**19.** Which statement about LLM pretraining is correct?

- A. It requires large volumes of human-labeled data
- B. It is self-supervised — labels are derived from the text itself
- C. It uses reinforcement learning from human feedback
- D. It always uses k-fold cross-validation

??? success "Answer"
    **B.** Next-token or masked-token prediction creates its own targets, which is why
    pretraining can scale to trillions of tokens. RLHF (C) comes later, in alignment.
    Cross-validation (D) is impractical at foundation-model scale.

---

**20.** A cross-encoder reranker is used in RAG because it:

- A. Encodes documents once, offline, for fast search
- B. Scores query and passage together for much higher relevance accuracy, at higher cost
- C. Compresses embeddings to save memory
- D. Replaces the need for a vector database

??? success "Answer"
    **B.** A **bi-encoder** embeds query and document separately (fast, precomputable) and
    is used for the initial wide retrieval; the **cross-encoder** reads the pair jointly
    and is far more accurate but too slow to run over the whole corpus. Retrieve wide,
    rerank narrow.

---

## Scoring

| Score | Verdict |
| --- | --- |
| 18–20 | Domain 1 is solid. Move on. |
| 14–17 | Re-read the pages behind your misses, then retry. |
| < 14 | Work through the chapter again — this is 30% of the exam. |
