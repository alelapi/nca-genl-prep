# Hands-On Labs

The NCA-GENL exam has **no lab component** — it is 50–60 multiple-choice questions. So why
do labs?

Because the questions are written from a practitioner's perspective. "Which chunk size
would you choose?" is an abstract question until you have watched a badly chunked RAG
system return the wrong half of a sentence. Concepts you have executed are the ones you
still remember under time pressure.

Each lab is short (20–60 minutes), runs on **CPU or a modest GPU**, and uses only free,
open models.

<div class="grid cards" markdown>

- **[Lab 0 — Environment Setup](00-setup.md)** · 15 min

    Python environment, packages, verifying GPU availability.

- **[Lab 1 — spaCy & NumPy](01-spacy-numpy.md)** · 30 min

    NLP pipeline, NER, lemmatization vs. stemming, TF-IDF, vectorised similarity.
    *Domain 1 (1.6, 1.10), Domain 4.*

- **[Lab 2 — Transformers with Hugging Face](02-transformers-hf.md)** · 45 min

    Tokenizers, pipelines, encoder vs. decoder, attention weights, decoding parameters.
    *Domain 1.*

- **[Lab 3 — Embeddings & Vector Search](03-embeddings-vector-search.md)** · 40 min

    Sentence embeddings, cosine similarity, a FAISS index, chunking experiments.
    *Domain 1 (1.4, 1.8).*

- **[Lab 4 — Build a RAG Pipeline](04-rag.md)** · 60 min

    End-to-end RAG from scratch: chunk, embed, retrieve, rerank, generate, cite.
    *Domain 1 (1.3), Domain 2 (4.2).*

- **[Lab 5 — Evaluating an LLM](05-evaluation.md)** · 45 min

    Classification metrics, ROUGE/BLEU, perplexity, retrieval metrics, LLM-as-a-judge.
    *Domain 3.*

- **[Lab 6 — LoRA Fine-Tuning](06-peft-lora.md)** · 60 min

    PEFT with LoRA on a small model; parameter counts, adapter size, before/after.
    *Domain 2.*

</div>

!!! tip "If you only have time for two"
    Do **[Lab 4 (RAG)](04-rag.md)** and **[Lab 5 (Evaluation)](05-evaluation.md)**. Between
    them they touch the largest share of exam-relevant concepts.

## Ground rules

- **Type the code, do not paste it.** The friction is the point.
- **Break things on purpose** — set chunk size to 20 tokens, set temperature to 2.0, use
  the wrong embedding model for queries. Each lab suggests specific things to break.
- **Write down what surprised you.** That is your personal weak-spot list.
