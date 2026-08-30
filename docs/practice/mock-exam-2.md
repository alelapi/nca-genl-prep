# Mock Exam 2

**60 questions · 60 minutes · no notes.** Harder than Mock Exam 1, with more
scenario-style items. Take it a few days before the real thing.

---

## Domain 1 — Core ML & AI (questions 1–18)

**1.** A team wants a model that answers in a specific JSON schema, uses company
terminology, and never invents policy details. They have 4,000 labeled examples and an
internal document corpus. The best architecture is:

- A. Fine-tune only
- B. RAG only
- C. PEFT fine-tuning for format and terminology, plus RAG for the policy facts
- D. Increase the context window

??? success "Answer"
    **C.** Behaviour gap → fine-tuning; knowledge gap → RAG. When both exist, do both.

**2.** Self-attention has which computational complexity in sequence length *n*?

- A. O(n)
- B. O(n log n)
- C. O(n²)
- D. O(1)

??? success "Answer"
    **C.** Every token attends to every token — the root cause of context-length cost.

**3.** The position-wise feed-forward network in a transformer block typically has an inner
dimension of:

- A. `d_model / 4`
- B. `d_model`
- C. `4 × d_model`
- D. The vocabulary size

??? success "Answer"
    **C.** Most transformer parameters live in the FFN.

**4.** RoPE (rotary positional embeddings) differ from sinusoidal encodings mainly because
they:

- A. Are learned during training
- B. Encode *relative* position by rotating Q and K, generalising better to long contexts
- C. Replace the attention mechanism
- D. Are only used in encoder models

??? success "Answer"
    **B.**

**5.** Which pretraining objective does BERT use?

- A. Next-token prediction
- B. Masked language modeling
- C. Contrastive learning
- D. Reinforcement learning

??? success "Answer"
    **B.** Predict masked tokens using bidirectional context.

**6.** A RAG pipeline retrieves the correct document at rank 40 out of 50 candidates, so it
never reaches the prompt. The most targeted fix is:

- A. Increase chunk overlap
- B. Add a cross-encoder reranker
- C. Switch to a larger generator model
- D. Lower the temperature

??? success "Answer"
    **B.** Recall is fine; ranking is the problem. That is precisely what reranking fixes.

**7.** Grouped-query attention (GQA) primarily reduces:

- A. The number of layers
- B. The KV cache size, by sharing K/V projections across heads
- C. The vocabulary size
- D. Training time only

??? success "Answer"
    **B.**

**8.** Which statement about tokenization is correct?

- A. Each word maps to exactly one token
- B. Subword tokenization handles unseen words by decomposing them
- C. Tokenizers are interchangeable between model families
- D. Tokenization happens after embedding

??? success "Answer"
    **B.** C is dangerously false — a mismatched tokenizer produces garbage.

**9.** "Lost in the middle" refers to:

- A. Gradient vanishing in deep networks
- B. Models attending less to information in the middle of a long context
- C. Chunks split across document boundaries
- D. Tokens dropped by the tokenizer

??? success "Answer"
    **B.** A reason to prefer fewer, better-ranked chunks over stuffing the context.

**10.** HyDE (Hypothetical Document Embeddings) improves retrieval by:

- A. Compressing the index
- B. Having the LLM draft a hypothetical answer and embedding that as the search query
- C. Reranking with a cross-encoder
- D. Removing stop words from queries

??? success "Answer"
    **B.** It closes the vocabulary gap between short queries and long passages.

**11.** Which is TRUE about a bi-encoder versus a cross-encoder?

- A. Cross-encoders are faster because documents are precomputed
- B. Bi-encoders embed query and document separately, enabling precomputation and fast search
- C. They are the same architecture
- D. Bi-encoders cannot be used for retrieval

??? success "Answer"
    **B.**

**12.** Which is NOT an emergent capability associated with model scale?

- A. In-context learning
- B. Chain-of-thought reasoning
- C. Instruction following
- D. Backpropagation

??? success "Answer"
    **D.** Backpropagation is a training algorithm, not a model capability.

**13.** A residual connection in a transformer block exists mainly to:

- A. Reduce parameter count
- B. Allow gradients to flow directly through deep stacks
- C. Normalize activations
- D. Add positional information

??? success "Answer"
    **B.**

**14.** Prompt chaining is preferable to one large prompt when:

- A. Latency is the only concern
- B. The task decomposes into steps you want to debug and evaluate independently
- C. The model has a small context window only
- D. Temperature must be zero

??? success "Answer"
    **B.**

**15.** Which library would you use to load a pretrained model from the Hugging Face Hub
and run text generation?

- A. spaCy
- B. transformers
- C. Gensim
- D. cuGraph

??? success "Answer"
    **B.**

**16.** MTEB is the standard benchmark for:

- A. Code generation
- B. Text embedding models
- C. Multilingual translation
- D. GPU throughput

??? success "Answer"
    **B.**

**17.** Diffusion models generate by:

- A. Autoregressive next-token prediction
- B. Learning to reverse a gradual noising process
- C. Adversarial training between two networks
- D. Retrieval and recombination

??? success "Answer"
    **B.** (C describes GANs.)

**18.** A team changes their embedding model but does not re-index. The expected result is:

- A. A clear error at query time
- B. Silently poor, effectively random retrieval
- C. Slower but correct retrieval
- D. Improved recall

??? success "Answer"
    **B.** The most dangerous kind of failure — no error, just wrong answers.

---

## Domain 2 — Software Development (questions 19–32)

**19.** Full fine-tuning a 7B model with Adam in mixed precision needs roughly how much
memory for parameters, gradients, master weights and optimizer states?

- A. ~14 GB
- B. ~28 GB
- C. ~112 GB
- D. ~448 GB

??? success "Answer"
    **C.** ~16 bytes/param × 7B ≈ 112 GB, before activations. The reason PEFT exists.

**20.** QLoRA differs from LoRA in that it:

- A. Trains the base weights
- B. Applies LoRA on top of a 4-bit quantized frozen base model
- C. Requires no adapter matrices
- D. Only works on encoder models

??? success "Answer"
    **B.**

**21.** Tensor parallelism is normally confined within a single node because:

- A. It cannot cross network boundaries technically
- B. It communicates many times per layer and needs NVLink-class bandwidth
- C. It requires identical GPU models
- D. NCCL does not support multi-node

??? success "Answer"
    **B.** Pipeline parallelism, with communication only at stage boundaries, is what
    crosses nodes.

**22.** ZeRO stage 3 shards which of the following across data-parallel ranks?

- A. Optimizer states only
- B. Optimizer states and gradients
- C. Optimizer states, gradients and parameters
- D. Activations only

??? success "Answer"
    **C.** Stage 1 = optimizer states, stage 2 adds gradients, stage 3 adds parameters.

**23.** PagedAttention improves serving by:

- A. Compressing model weights
- B. Managing the KV cache in fixed-size blocks, removing fragmentation and enabling larger batches
- C. Skipping attention for short sequences
- D. Caching HTTP responses

??? success "Answer"
    **B.**

**24.** Which is the correct characterisation of BF16 versus FP16?

- A. BF16 has more mantissa bits and less range
- B. BF16 has the same exponent range as FP32 and fewer mantissa bits, making it more robust for training
- C. They are identical
- D. FP16 uses 8 bits

??? success "Answer"
    **B.**

**25.** A chatbot's p99 latency is 6 s while p50 is 400 ms. The most useful next step is:

- A. Report the mean instead
- B. Investigate what the slow tail has in common — long prompts, queue depth, cold starts
- C. Increase batch size
- D. Reduce model size immediately

??? success "Answer"
    **B.** p99 is where users actually suffer; diagnose before you change anything.

**26.** Which practice best prevents an LLM application from regressing when a prompt is
edited?

- A. Manual spot checks
- B. A versioned evaluation set run in CI, blocking on regressions
- C. Raising the temperature
- D. Increasing the context window

??? success "Answer"
    **B.**

**27.** NIM is best described as:

- A. A training framework
- B. Containerised, GPU-optimized model endpoints with a standard API, deployable in your own environment
- C. A vector database
- D. A data curation tool

??? success "Answer"
    **B.**

**28.** Model output is inserted directly into a SQL query. This maps to which OWASP LLM
risk?

- A. Overreliance
- B. Insecure output handling
- C. Model theft
- D. Training data poisoning

??? success "Answer"
    **B.** Treat model output like untrusted user input, always.

**29.** Which memory component often exceeds the model weights at long context and high
concurrency?

- A. The tokenizer vocabulary
- B. The KV cache
- C. The optimizer states
- D. Positional encodings

??? success "Answer"
    **B.** Optimizer states (C) exist only during training.

**30.** For a document-processing batch job run overnight, you should optimise:

- A. Time to first token
- B. Throughput — large batches, maximum GPU utilisation
- C. Streaming responsiveness
- D. Prompt caching only

??? success "Answer"
    **B.**

**31.** Knowledge distillation transfers capability by training the student on the
teacher's:

- A. Weights directly
- B. Soft output distributions, which carry more information than hard labels
- C. Gradients
- D. Attention masks

??? success "Answer"
    **B.**

**32.** Which is the right first response to "our GPU inference is too slow"?

- A. Buy more GPUs
- B. Right-size the model, quantize, compile with TensorRT-LLM, enable continuous batching — then scale out
- C. Increase temperature
- D. Switch to CPU

??? success "Answer"
    **B.** Optimise before you scale; horizontal scaling multiplies an unoptimised cost.

---

## Domain 3 — Experimentation (questions 33–45)

**33.** A candidate summary paraphrases the reference perfectly but shares few n-grams.
Which metric penalises it unfairly?

- A. BERTScore
- B. ROUGE
- C. Human evaluation
- D. LLM-as-a-judge

??? success "Answer"
    **B.** N-gram overlap metrics cannot credit paraphrase; BERTScore and judges can.

**34.** Which pair of fairness or evaluation metrics is mathematically incompatible when
base rates differ across groups?

- A. Precision and recall
- B. Equalized odds and predictive parity
- C. BLEU and ROUGE
- D. Recall@k and MRR

??? success "Answer"
    **B.** The impossibility result — fairness requires an explicit documented choice.

**35.** Statistical power of 80% means:

- A. 80% of results will be significant
- B. There is an 80% chance of detecting a real effect of the assumed size
- C. The p-value will be below 0.2
- D. 80% of users see the treatment

??? success "Answer"
    **B.**

**36.** A model's stated chain of thought should NOT be used as:

- A. A prompting technique
- B. An audit trail of how the answer was actually computed
- C. A way to improve multi-step reasoning
- D. Input to a verification step

??? success "Answer"
    **B.** Stated reasoning is generated text, not an introspection log.

**37.** Which of these is the strongest evidence that a benchmark score is inflated?

- A. The model is large
- B. The benchmark data appears in the model's training corpus
- C. The score improved over the previous version
- D. The benchmark is multiple choice

??? success "Answer"
    **B.** Contamination.

**38.** Constitutional AI reduces the need for:

- A. Pretraining data
- B. Large volumes of human preference labels
- C. Tokenization
- D. Evaluation

??? success "Answer"
    **B.** The model critiques and revises against written principles.

**39.** Reward hacking in RLHF means the policy:

- A. Fails to converge
- B. Maximises the proxy reward while violating the intent behind it
- C. Overfits the SFT data
- D. Refuses too often

??? success "Answer"
    **B.** The KL penalty against the reference model is the standard mitigation.

**40.** A team wants to compare two ranking systems with far less traffic than a standard
A/B test would need. The best option is:

- A. Shadow deployment
- B. Interleaving
- C. Multi-armed bandit
- D. Canary release

??? success "Answer"
    **B.** Interleaving blends both systems' results into one list and is much more
    sensitive per impression.

**41.** In a RAG evaluation, context relevance is high but faithfulness is low. This means:

- A. Retrieval is broken
- B. The right context was retrieved, but the model is not sticking to it
- C. The embedding model is mismatched
- D. Chunks are too small

??? success "Answer"
    **B.** Fix with a stricter grounding prompt, lower temperature, and fewer/cleaner chunks.

**42.** Which is the correct interpretation of a 95% confidence interval of [−0.5%, +4.2%]
for a treatment effect?

- A. The treatment definitely helps
- B. The effect is not statistically distinguishable from zero at this sample size
- C. The effect is exactly 1.85%
- D. The test was invalid

??? success "Answer"
    **B.** The interval includes zero.

**43.** Which is the best reason to prefer pairwise human preference over 1–5 Likert
scoring?

- A. It is faster to collect
- B. Humans are more reliable at comparison than at absolute judgement
- C. It requires fewer annotators
- D. It produces continuous scores

??? success "Answer"
    **B.** The same reason RLHF collects rankings.

**44.** Self-consistency improves answer reliability by:

- A. Lowering temperature to zero
- B. Sampling multiple reasoning paths and taking the majority answer
- C. Retrieving more documents
- D. Fine-tuning on the eval set

??? success "Answer"
    **B.**

**45.** Which statement about an evaluation set is correct?

- A. It should be regenerated for each experiment
- B. It should be fixed, versioned, representative, and grown from production failures
- C. Few-shot examples may be drawn from it
- D. It should contain only easy cases

??? success "Answer"
    **B.** C is leakage.

---

## Domain 4 — Data Analysis (questions 46–54)

**46.** Which RAPIDS library mirrors the scikit-learn API?

- A. cuDF
- B. cuML
- C. cuGraph
- D. cuSpatial

??? success "Answer"
    **B.**

**47.** Applying SMOTE before splitting into train and test causes:

- A. Nothing unusual
- B. Leakage — synthetic rows derived from training data end up in the test set
- C. A memory error
- D. Class imbalance to worsen

??? success "Answer"
    **B.**

**48.** Spearman correlation is preferable to Pearson when:

- A. The relationship is linear and the data is normal
- B. The relationship is monotonic but non-linear, or there are outliers
- C. The variables are categorical
- D. The sample is very large

??? success "Answer"
    **B.**

**49.** Simpson's paradox describes:

- A. Overfitting on small data
- B. A trend that reverses when subgroups are aggregated
- C. Vanishing gradients
- D. Class imbalance

??? success "Answer"
    **B.**

**50.** Adjusted R² is preferred over R² when:

- A. The target is categorical
- B. Comparing models with different numbers of predictors
- C. Data is imbalanced
- D. Using cross-validation

??? success "Answer"
    **B.** Plain R² never decreases when features are added, even useless ones.

**51.** Which preprocessing step is appropriate before feeding text to BERT?

- A. Stop-word removal and stemming
- B. Using the model's own tokenizer on clean, natural text
- C. Lowercasing plus punctuation stripping
- D. TF-IDF vectorization

??? success "Answer"
    **B.** The classical pipeline destroys signal that transformers use.

**52.** Back-translation is a technique for:

- A. Model compression
- B. Text data augmentation, producing natural paraphrases
- C. Evaluating translation quality
- D. Tokenization

??? success "Answer"
    **B.**

**53.** For a 30 MB dataset processed with heavy custom Python loops, RAPIDS will most
likely:

- A. Be much faster
- B. Be slower — kernel launch and transfer overhead dominates
- C. Fail to load the data
- D. Automatically parallelise the loops

??? success "Answer"
    **B.**

**54.** Which is the most robust measure of spread for a heavily skewed distribution?

- A. Standard deviation
- B. Interquartile range
- C. Variance
- D. Mean absolute deviation from the mean

??? success "Answer"
    **B.**

---

## Domain 5 — Trustworthy AI (questions 55–60)

**55.** Which is the strongest argument for RAG over fine-tuning when personal data is
involved?

- A. RAG is cheaper
- B. Deleting a document from the index takes effect immediately, honouring erasure requests
- C. RAG is more accurate
- D. Fine-tuning requires more GPUs

??? success "Answer"
    **B.** Removing an individual's influence from trained weights generally requires
    retraining.

**56.** "Data minimisation" means:

- A. Compressing datasets
- B. Collecting only the data necessary for the stated purpose
- C. Reducing model size
- D. Deleting logs weekly

??? success "Answer"
    **B.**

**57.** Which is the correct layered defence against prompt injection?

- A. A single strong system prompt
- B. Delimit untrusted input, filter inputs, validate outputs, least-privilege tools, human confirmation for consequential actions
- C. Raising the temperature
- D. Using a larger model

??? success "Answer"
    **B.** No prompt-only defence is sufficient, because the model sees one flat token
    stream.

**58.** Differential privacy provides:

- A. Encryption of data at rest
- B. A mathematical guarantee that any single individual's data does not meaningfully change the output
- C. Access control
- D. Data residency

??? success "Answer"
    **B.**

**59.** "Overreliance" as an OWASP LLM risk refers to:

- A. Excessive GPU usage
- B. Humans trusting fluent but incorrect output
- C. Too many API calls
- D. Model dependency on retrieval

??? success "Answer"
    **B.**

**60.** A model card should include, at minimum:

- A. Only the architecture and parameter count
- B. Intended use, out-of-scope uses, training data summary, evaluation results disaggregated by subgroup, and known limitations
- C. The full training dataset
- D. The model weights

??? success "Answer"
    **B.**

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

If you scored 52+ on both mocks, you are ready. Review the
[Exam-Day Strategy](../exam/exam-day.md) and the
[Flashcards](flashcards.md), and go book it.
