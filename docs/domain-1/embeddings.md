# Embeddings & Vector Search

*Covers tasks 1.4 (curate and embed content datasets for RAGs) and 1.8 (select and use models
to create text embeddings).*

Embeddings are the bridge between text and mathematics. Once you can turn a sentence into a
vector whose *geometry* encodes its *meaning*, you can search, cluster, deduplicate, classify
and retrieve using nothing but arithmetic. Everything in [RAG](rag.md) rests on this page.

---

## 1. The problem with counting words

Start with what came before, because it makes the motivation concrete.

The classical way to turn text into numbers is to count words. A document becomes a vector as
long as the vocabulary, with one entry per word. **TF-IDF** weights those counts by how
distinctive each word is (see [ML Fundamentals](ml-fundamentals.md)).

Now try to compare these two sentences:

```text
A: "The automobile is fast."
B: "The car is quick."
```

They mean the same thing. But as TF-IDF vectors, `automobile` and `car` are two entirely
different dimensions, as are `fast` and `quick`. The only word they share is "the", which
TF-IDF assigns a weight of zero because it appears everywhere. **Their cosine similarity is
approximately 0.0.** By this representation, these sentences are as unrelated as one about
GPUs and one about pasta.

This is the fundamental limit of count-based representations: they are **sparse** — nearly all
zeros, one dimension per vocabulary word — and they match only on **exact tokens**. They have
no concept of meaning. Every synonym, paraphrase and rephrasing is invisible to them.

!!! note "You can see this yourself"
    [Lab 1](../labs/01-spacy-numpy.md) computes exactly this comparison and shows TF-IDF
    scoring ~0.0, and [Lab 3](../labs/03-embeddings-vector-search.md) reruns it with embeddings.
    The contrast is the entire motivation for the field.

---

## 2. What an embedding is

An **embedding** is a dense, fixed-length vector of floats that represents a piece of content
in a space where **geometric closeness corresponds to semantic similarity**.

```text
sparse (TF-IDF), vocabulary size 50,000:
  [0, 0, 0, 0, 0.71, 0, 0, 0, 0, 0, 0, 0.43, 0, 0, ... 49,988 more zeros ...]
  ↑ one dimension per word, meaning nothing on its own

dense (embedding), 384 dimensions:
  [0.12, -0.44, 0.87, 0.03, -0.29, 0.55, 0.71, ... 377 more ...]
  ↑ every dimension carries a fraction of the meaning; no single one is interpretable
```

Three properties define it:

- **Dense** — nearly every value is non-zero, and the information is distributed across all of
  them rather than concentrated in one.
- **Fixed-length** — a three-word sentence and a three-hundred-word paragraph both produce a
  vector of the same size. This is what makes them comparable.
- **Semantic** — position in the space reflects meaning. "dog" lands near "puppy" and far from
  "database".

Typical dimensionalities: 384, 768, 1024, 1536, 4096. The same idea applies to words,
sentences, whole documents, images, audio and code.

### Meaning as direction

The famous demonstration from Word2Vec:

$$ \text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen}) $$

Read that as a statement about geometry. The *direction* you travel to get from "man" to
"woman" is roughly the same direction that takes you from "king" to "queen" — the model has
learned a "gender" direction in the space without ever being told that gender exists. The same
holds for `Paris − France + Italy ≈ Rome` (a "capital of" direction) and for singular→plural,
present→past tense, and many others.

Nothing supervised this. It emerges because words that appear in similar contexts get similar
vectors, and the structure of language makes these relationships systematic.

!!! warning "The same mechanism produces bias"
    If the training corpus systematically associates certain occupations with certain genders,
    that association becomes a *direction in the embedding space* just like the legitimate
    ones. This is precisely how societal bias enters models mechanically, and it is why
    [Bias & Fairness](../domain-5/bias-fairness.md) is a certification topic rather than a
    footnote.

---

## 3. The evolution — and the distinction the exam tests

| Generation | Examples | What it does | Key limitation |
| --- | --- | --- | --- |
| Count-based | Bag of words, TF-IDF | Sparse token counts | No semantics at all |
| **Static** word embeddings | **Word2Vec**, GloVe, fastText | Dense, semantic | **One vector per word, regardless of context** |
| **Contextual** embeddings | ELMo, **BERT** | Vector depends on the whole sentence | Per-token, not per-sentence |
| **Sentence/document** embeddings | **Sentence-BERT**, E5, BGE, NV-Embed | Optimised so whole texts compare directly | — |

### Static vs. contextual, concretely

Word2Vec assigns exactly one vector to the word "bank". That single vector has to
simultaneously represent:

```text
"I sat on the river bank and watched the water."      ← geography
"I deposited the cheque at the bank downtown."        ← finance
```

It cannot. The best it can do is a blurry average of both senses — which is close to neither.

BERT computes each token's representation **from the whole sentence** through attention. The
word "bank" in the first sentence attends to "river" and "water" and produces one vector; in
the second it attends to "cheque" and "deposited" and produces a different one. Same input
token, different output vectors.

!!! tip "A near-guaranteed exam question"
    **Word2Vec produces a single static vector per word. BERT and successors produce
    contextual vectors that depend on the surrounding sentence.** That is the key advance, and
    distractors will reverse it.

    [Lab 2](../labs/02-transformers-hf.md) measures this directly — cosine similarity between
    "bank" in a river sentence and "bank" in a finance sentence.

### Why sentence embeddings need their own models

A subtlety worth knowing. BERT gives you a vector per *token*. To get one vector for a whole
sentence, the obvious moves are to take the `[CLS]` token or to average all token vectors — and
both work surprisingly badly for similarity search, because BERT was never trained to make
those vectors comparable across sentences.

**Sentence-BERT** and its successors fix this by fine-tuning with a **contrastive objective**:
show the model pairs of sentences and train it so that related pairs end up close together and
unrelated pairs far apart. The vectors are then explicitly optimised for the thing you want to
do with them.

This is why you should use a purpose-built embedding model rather than pooling a general LLM's
hidden states.

---

## 4. Similarity metrics

Once text is a vector, "similar" becomes a distance calculation. Three candidates:

| Metric | Formula | Range | Character |
| --- | --- | --- | --- |
| **Cosine similarity** | `A·B / (‖A‖‖B‖)` | [−1, 1] | Measures the **angle**. Ignores magnitude. **The default for text** |
| **Dot product** | `A·B` | (−∞, ∞) | Fast (no normalisation). Equals cosine when vectors are unit length |
| **Euclidean (L2)** | `‖A − B‖₂` | [0, ∞) | Straight-line distance. Magnitude-sensitive |

### Why cosine, specifically

Consider a one-sentence note about GPU memory and a twenty-page document about GPU memory. The
document's embedding will typically have a **larger magnitude** — more content, more activated
features — but it points in essentially the same *direction* in the space, because it is about
the same thing.

Euclidean distance would call them far apart, because the vectors have very different lengths.
Cosine ignores length entirely and asks only "do these point the same way?" — which is exactly
the question you want answered.

### The normalisation shortcut

If you **L2-normalise** every vector to unit length first, then:

$$ \|A\| = \|B\| = 1 \quad\Longrightarrow\quad \cos(A,B) = \frac{A \cdot B}{1 \times 1} = A \cdot B $$

Cosine similarity **becomes** the dot product. And Euclidean distance becomes a monotonic
function of it, so all three metrics produce the **identical ranking**.

This is why production code almost always normalises at encoding time and then uses a
dot-product index: you get cosine semantics at dot-product speed. `IndexFlatIP` in FAISS
("inner product") is the standard choice for exactly this reason.

!!! tip "Exam-ready"
    Cosine is the conventional text metric because it is insensitive to vector length. **On
    normalised vectors, cosine, dot product and Euclidean rank identically** — a detail that
    appears in answer options.

---

## 5. Choosing an embedding model (task 1.8)

The blueprint asks you to "select and use models to create text embeddings". Here is what
actually drives the choice, roughly in order of importance.

### Symmetric vs. asymmetric — the one people get wrong

This distinction matters more than dimensionality or leaderboard position.

- **Symmetric** tasks compare two things of the same kind and length: sentence ↔ sentence
  similarity, duplicate detection, clustering.
- **Asymmetric** tasks compare a **short query** against a **long passage**. That is what
  retrieval is: "how do I reduce inference latency?" (8 words) must match a 200-word paragraph
  about batching and quantization.

Those are different geometric problems, and models are trained for one or the other. Many
retrieval models expect an **instruction prefix** — literally prepending `query: ` to queries
and `passage: ` to documents — and omitting it measurably degrades results while producing no
error whatsoever.

### The rest of the checklist

| Criterion | What to check |
| --- | --- |
| **Domain** | General-purpose vs. legal, biomedical, code. Domain models win substantially on domain text |
| **Language** | Monolingual vs. multilingual. A multilingual model can match a French query to an English passage |
| **Dimensionality** | Higher is usually marginally better and costs storage, memory and search time **linearly**. 384 and 768 are common sweet spots |
| **Max sequence length** | Often 512 tokens. Anything longer is **truncated silently** — no error, just lost content |
| **Benchmark** | **MTEB** (Massive Text Embedding Benchmark) is the standard leaderboard. Check the *retrieval* subset, not the average |
| **Deployment** | Self-hosted open model vs. hosted API. NVIDIA ships retrieval embedding and reranking models via **NeMo Retriever** and **NIM** |

### The rule you cannot break

!!! danger "Same model for indexing and querying. Always."
    Embeddings from two different models live in unrelated coordinate systems. Dimension 47
    means something completely different in each. Comparing across them is not "less accurate"
    — it is **meaningless**, like comparing a temperature in Celsius to a pressure in pascals.

    What makes this genuinely dangerous is that **nothing errors**. If the dimensions happen to
    match, the search runs, returns results with plausible-looking scores, and they are
    essentially random. You discover it through mysteriously bad answers, not through a stack
    trace.

    The corollary: **changing your embedding model means re-indexing the entire corpus.** Budget
    for it. [Lab 3](../labs/03-embeddings-vector-search.md) reproduces this failure so you can
    see how quietly it happens.

---

## 6. Curating and embedding a dataset (task 1.4)

The blueprint says "curate and embed content datasets for RAGs". Here is the full pipeline.

```text
raw docs → extract → clean → deduplicate → chunk → embed → store with metadata → index
```

### Extract

PDF, HTML, DOCX, Confluence, Markdown → plain text. The unglamorous step where most real
projects lose quality.

Preserve structure where you can: headings tell you what a section is about, tables lose all
meaning when flattened into a word soup, and code blocks need their formatting. A PDF with
two-column layout extracted naively interleaves the columns line by line and produces text
that is grammatically incoherent — and no downstream embedding model can recover from that.

### Clean

Strip navigation menus, headers, footers, cookie banners, "share this page" boilerplate. Fix
encoding damage (`â€™` instead of `'`). Normalise whitespace.

Boilerplate is actively harmful, not merely wasteful: if the same 200-word footer appears in
every chunk, every chunk's embedding is pulled toward the footer's meaning, and they all become
more similar to each other and less distinguishable.

### Deduplicate

Exact duplicates via hashing; near-duplicates via **MinHash/LSH** or embedding similarity.

Why it matters for retrieval: if a document is indexed five times, a query matching it consumes
five of your top-10 slots with identical content, crowding out other relevant material. You
have effectively reduced `k` from 10 to 6.

**NeMo Curator** is NVIDIA's GPU-accelerated tool for this at corpus scale — extraction,
quality filtering, exact and fuzzy deduplication, PII removal.

### Chunk

Documents are too long to embed as a unit — they exceed the model's sequence limit, and a single
vector cannot meaningfully represent fifty pages about twelve different topics. So you split.

| Strategy | How | Trade-off |
| --- | --- | --- |
| **Fixed-size** | N tokens with overlap | Simple and predictable; cuts mid-sentence |
| **Sentence / paragraph** | Split on natural boundaries | Preserves meaning; variable sizes |
| **Recursive character** | Try paragraph → sentence → word until it fits | The practical default; respects structure while bounding size |
| **Semantic** | Split where consecutive-sentence embedding similarity drops | Best coherence; requires embedding everything twice |
| **Document-structure aware** | Split on headings and sections | Excellent for technical documentation and legal text |

**Overlap** — typically 10–20% — means consecutive chunks share their boundary text. Its
purpose is specific: without it, a sentence that answers the question can be sliced in half,
with the subject in chunk 3 and the predicate in chunk 4, so neither chunk retrieves well and
neither answers the question. Overlap guarantees any short span appears intact in at least one
chunk.

### The chunk-size trade-off

This is the highest-leverage decision in a RAG system and a reliable exam topic.

```text
TOO SMALL (e.g. 50 tokens)              TOO LARGE (e.g. 2000 tokens)
─────────────────────────               ────────────────────────────
"...dynamic batching improves"          [a chunk covering batching, versioning,
                                         metrics, Kubernetes deployment and
Retrieved. But improves what?            security, all in one]
By how much? The context that
made it meaningful is in the            Retrieved. The embedding is an average
neighbouring chunk.                     of five topics, so it matches every
                                        query weakly and none strongly.
→ precise match, insufficient           Precision drops, and you burn context
  information                           window on mostly-irrelevant text.
```

Typical starting point: **256–512 tokens with 10–20% overlap**, then tune empirically against a
retrieval evaluation set. There is no universally correct value — it depends on your documents
and your queries, which is why measurement matters more than the number.

### Embed and store

Batch the chunks through the model on GPU. Normalise if you plan to use dot-product search.

**Store metadata alongside every vector** — this is not optional:

```python
{
  "vector":     [0.12, -0.44, ...],
  "text":       "Dynamic batching groups incoming requests...",
  "source":     "triton-guide.pdf",
  "chunk_id":   "triton-guide.pdf#12",
  "title":      "Performance Tuning",
  "section":    "Dynamic Batching",
  "updated_at": "2026-03-14",
  "acl":        ["engineering", "support"]
}
```

Metadata is what enables three things you will otherwise wish you had:

- **Citations** — you cannot say "according to triton-guide.pdf, page 12" without storing it.
  Citations are what make a RAG answer verifiable, which is why NVIDIA files RAG under
  [Trustworthy AI](../domain-5/index.md).
- **Filtered search** — "only documents from 2026", "only the EU policy set".
- **Access control** — filter by the user's permissions **at query time**. Never rely on the
  model to decline to mention something it was shown; if a chunk reaches the prompt, treat it
  as disclosed.

---

## 7. Vector databases

A **vector database** stores embeddings and answers nearest-neighbour queries: "give me the `k`
vectors most similar to this one."

### Exact vs. approximate search

**Exact (brute force)** compares the query against every stored vector. Perfectly accurate,
O(N) per query. With 10,000 vectors this is a single fast matrix multiplication and completely
fine. With 100 million it is not.

**Approximate nearest neighbour (ANN)** trades a small amount of recall for orders of magnitude
of speed, by avoiding comparison against most of the corpus.

| Index | How it works | Character |
| --- | --- | --- |
| **Flat** | Brute force | Exact; the correctness baseline |
| **IVF** | Cluster vectors; search only the nearest few clusters | Tunable via `nprobe` (how many clusters to check) |
| **HNSW** | A navigable multi-layer graph; greedily hop toward the query | Excellent recall/latency; **high memory** |
| **PQ / IVF-PQ** | Compress vectors via product quantization | Large memory savings, some accuracy loss |

The recall/speed dial is real and worth understanding: `nprobe` in IVF or `efSearch` in HNSW
directly trade query latency against the probability of finding the true nearest neighbours.
Set them too aggressively and your retrieval silently starts missing documents.

### The options you should recognise

**FAISS** (a library, not a server — Meta), **Milvus**, **Qdrant**, **Weaviate**, **Chroma**,
**pgvector** (a PostgreSQL extension — often the right answer when you already run Postgres and
have under a few million vectors), **Pinecone** (managed), **Elasticsearch/OpenSearch**.

NVIDIA accelerates ANN search on GPUs through **cuVS/RAFT**, which is what backs FAISS-GPU and
Milvus's GPU indexes.

---

## 8. The two techniques that fix dense retrieval's weaknesses

### Hybrid search

Dense embeddings are excellent at meaning and **weak at rare exact tokens**. Search for error
code `E-4471` or part number `NV-8823-B`, and the embedding model has likely never seen that
string; its tokenizer shatters it into meaningless fragments and the resulting vector is noise.

Meanwhile **BM25** — a refined TF-IDF — handles it perfectly, because it matches literally.

**Hybrid search** runs both and fuses the rankings, usually with **Reciprocal Rank Fusion**,
which combines by rank position rather than by score (avoiding the problem that the two systems'
scores are on incomparable scales):

$$ \text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)} $$

This is the standard production answer, and a recurring exam scenario: *"users search by SKU
and get poor results"* → **hybrid search**.

### Reranking

A two-stage retrieval architecture, and the vocabulary is worth getting exactly right.

**Bi-encoder** (what your index uses): query and document are embedded **separately**. Document
vectors can therefore be computed once, offline, and searched in milliseconds. The cost is that
the model never sees the query and document together, so it must produce a vector that is good
for *all possible* queries.

**Cross-encoder** (the reranker): query and document are fed through the model **together**, as
one concatenated input, so attention can compare them token by token. Far more accurate. But
it requires a full forward pass per candidate, so scoring a million documents is impossible.

```text
1M documents ──[bi-encoder ANN search]──► top 50 ──[cross-encoder]──► top 5 ──► LLM
               fast, approximate                    slow, accurate
```

**Retrieve wide, rerank narrow.** [Lab 4](../labs/04-rag.md) implements both stages and shows
the ordering change directly.

---

## 9. Embeddings beyond retrieval

Retrieval dominates the exam, but the same vectors power several other things worth
recognising:

- **Clustering** — group documents by embedding, then label each cluster with an LLM. This has
  largely replaced classical topic modeling for "what is in this corpus?"
- **Classification** — embed your training texts, train a simple logistic regression on the
  vectors. Extraordinarily effective with a few hundred labeled examples, and far cheaper to
  serve than an LLM.
- **Deduplication** — near-duplicates are near neighbours.
- **Anomaly detection** — outliers are far from every cluster.
- **Recommendation** — items close in embedding space are similar items.
- **Few-shot example selection** — retrieve the most similar labeled examples to use as
  few-shot demonstrations. See [Zero- & Few-Shot](../domain-3/zero-few-shot.md).

---

## 10. Recap

- Count-based representations (TF-IDF) are sparse and match only exact tokens; "car" and
  "automobile" score ~0.0. **Embeddings map meaning to geometry**, which is the whole point.
- Semantic relationships become **directions** in the space — which is also how bias enters
  mechanically.
- **Word2Vec = static** (one vector per word). **BERT = contextual** (depends on the sentence).
  **Sentence-BERT** adds a contrastive objective so whole texts compare properly.
- **Cosine** is the default text metric because it ignores vector length. On **normalised**
  vectors, cosine, dot product and Euclidean rank identically — so normalise and use dot
  product.
- Choose an embedding model by **task symmetry** (retrieval is asymmetric), domain, language,
  max sequence length and MTEB retrieval scores.
- **Same model for index and query, always.** Mismatches fail silently, and changing the model
  forces a full re-index.
- Chunking is the highest-leverage RAG knob: too small loses context, too large dilutes the
  embedding. Start at 256–512 tokens with 10–20% overlap, then measure.
- **Store metadata** — it is what enables citations, filtering and access control.
- ANN indexes (HNSW, IVF, PQ) trade recall for speed. **Hybrid search** fixes exact-identifier
  failures; **cross-encoder reranking** fixes ranking precision.
