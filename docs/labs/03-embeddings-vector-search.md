# Lab 3 — Embeddings & Vector Search

**Time:** 40 minutes · **Covers:** tasks 1.4, 1.8

## 1. Sentence embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim, small, fast
print("dimension:", model.get_sentence_embedding_dimension())
print("max seq length:", model.max_seq_length)     # note this — silent truncation lives here

sentences = [
    "The automobile is fast.",
    "The car is quick.",
    "GPUs accelerate deep learning.",
    "Graphics processors speed up neural network training.",
    "I ate pasta for dinner.",
]

emb = model.encode(sentences, normalize_embeddings=True)
print("shape:", emb.shape)
```

## 2. Cosine similarity — the metric that matters

```python
sim = emb @ emb.T          # normalized vectors → dot product == cosine
np.set_printoptions(precision=2, suppress=True)
print(sim)
```

Compare against Lab 1: TF-IDF scored "automobile/car" at ~0.0. Embeddings should score
them high. **That is the entire point of dense retrieval.**

```python
# Cosine vs. dot vs. euclidean on normalized vectors
a, b = emb[0], emb[1]
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
dot = a @ b
l2 = np.linalg.norm(a - b)
print(f"cosine={cos:.4f}  dot={dot:.4f}  euclidean={l2:.4f}")
print("cosine == dot on normalized vectors:", np.isclose(cos, dot))
```

## 3. Build a vector index with FAISS

```python
import faiss

corpus = [
    "NVIDIA Triton Inference Server hosts models in production with dynamic batching.",
    "TensorRT-LLM optimizes large language model inference with quantization and fused kernels.",
    "LoRA freezes pretrained weights and trains small low-rank matrices instead.",
    "Retrieval-augmented generation grounds LLM answers in retrieved documents.",
    "cuDF is the RAPIDS GPU DataFrame library, an accelerated replacement for pandas.",
    "Ring-AllReduce synchronizes gradients across GPUs with constant per-GPU communication.",
    "Perplexity is the exponentiated average cross-entropy of a language model.",
    "NeMo Guardrails adds input, dialog, retrieval, execution and output rails to LLM apps.",
]

vecs = model.encode(corpus, normalize_embeddings=True).astype("float32")

index = faiss.IndexFlatIP(vecs.shape[1])    # inner product == cosine on normalized vectors
index.add(vecs)
print("indexed:", index.ntotal)

def search(query, k=3):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    print(f"\nQ: {query}")
    for s, i in zip(scores[0], ids[0]):
        print(f"  {s:.3f}  {corpus[i]}")

search("How do I make inference faster?")
search("How do I fine-tune without lots of GPU memory?")
search("How do multiple GPUs share gradients?")
```

**Observe:** none of these queries share many literal words with the documents they
retrieve. That is semantic search working.

## 4. Where dense retrieval fails

```python
corpus2 = corpus + ["Error code E-4471 indicates a thermal shutdown of the GPU."]
vecs2 = model.encode(corpus2, normalize_embeddings=True).astype("float32")
idx2 = faiss.IndexFlatIP(vecs2.shape[1]); idx2.add(vecs2)

q = model.encode(["E-4471"], normalize_embeddings=True).astype("float32")
scores, ids = idx2.search(q, 3)
for s, i in zip(scores[0], ids[0]):
    print(f"{s:.3f}  {corpus2[i]}")
```

Dense retrieval is weak on **rare exact identifiers**. Now the keyword approach:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tv = TfidfVectorizer()
M = tv.fit_transform(corpus2)
kw = cosine_similarity(tv.transform(["E-4471"]), M)[0]
print("\nkeyword top hit:", corpus2[kw.argmax()], f"({kw.max():.3f})")
```

**This is why hybrid search exists.** Dense for meaning, sparse for exact tokens.

## 5. The embedding-mismatch disaster

```python
other = SentenceTransformer("paraphrase-MiniLM-L3-v2")   # different model, same dim

q_wrong = other.encode(["How do I make inference faster?"],
                       normalize_embeddings=True).astype("float32")
scores, ids = index.search(q_wrong, 3)     # index built with all-MiniLM-L6-v2!
print("Querying with the WRONG model:")
for s, i in zip(scores[0], ids[0]):
    print(f"  {s:.3f}  {corpus[i]}")
```

Note that it **does not error**. It returns confident nonsense. This is the failure mode
behind the rule: *same embedding model for indexing and querying, always.*

## 6. Chunking experiment

```python
doc = (
    "NVIDIA Triton Inference Server is an open-source serving solution. "
    "It supports multiple frameworks including TensorRT, PyTorch and ONNX Runtime. "
    "Triton provides dynamic batching, which groups incoming requests to improve throughput. "
    "It also supports concurrent model execution on a single GPU. "
    "Model versioning allows hot-swapping without restarting the server. "
    "Prometheus metrics are exposed for latency, throughput and GPU utilisation. "
    "Triton runs on Kubernetes, in the cloud, on-premises and at the edge."
)

def fixed_chunks(text, size, overlap):
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i+size]) for i in range(0, len(words), step)]

for size, overlap in [(8, 0), (20, 5), (60, 10)]:
    chunks = fixed_chunks(doc, size, overlap)
    cv = model.encode(chunks, normalize_embeddings=True).astype("float32")
    ix = faiss.IndexFlatIP(cv.shape[1]); ix.add(cv)
    q = model.encode(["What does dynamic batching do?"],
                     normalize_embeddings=True).astype("float32")
    s, i = ix.search(q, 1)
    print(f"\nsize={size} overlap={overlap} → {len(chunks)} chunks, score {s[0][0]:.3f}")
    print(f"  {chunks[i[0][0]]}")
```

Read the retrieved text at each setting. At size 8 the chunk is too fragmentary to answer;
at size 60 it is diluted with unrelated sentences. **The trade-off is now something you
have seen, not something you memorised.**

## Break it on purpose

1. Embed a 2,000-word document as a single chunk. Check `model.max_seq_length` — how much
   was silently discarded?
2. Build the index **without** `normalize_embeddings=True` and use `IndexFlatIP`. Watch
   long documents dominate every result.
3. Add 5,000 random unrelated sentences to the corpus and re-run the searches. Does
   precision hold?

## Takeaways

- Embeddings capture meaning; TF-IDF captures tokens.
- **Cosine == dot product on normalized vectors** — normalize, then use inner product.
- Dense retrieval fails on exact identifiers → **hybrid search**.
- **Mismatched embedding models fail silently.** Same model for index and query.
- Chunk size trades context against precision; overlap prevents cutting answers in half.
