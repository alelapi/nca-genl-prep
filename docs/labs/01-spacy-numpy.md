# Lab 1 — spaCy & NumPy

**Time:** 30 minutes · **Covers:** tasks 1.6, 1.10, and Domain 4 preprocessing

## 1. The spaCy pipeline

```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = ("NVIDIA released the H100 GPU in 2022. Jensen Huang announced it at GTC "
        "in San Jose, and analysts at Goldman Sachs raised their price target to $500.")

doc = nlp(text)

print(f"{'TOKEN':<14}{'LEMMA':<14}{'POS':<8}{'DEP':<12}{'STOP'}")
for t in doc[:15]:
    print(f"{t.text:<14}{t.lemma_:<14}{t.pos_:<8}{t.dep_:<12}{t.is_stop}")

print("\nEntities:")
for ent in doc.ents:
    print(f"  {ent.text:<18}{ent.label_:<8}{spacy.explain(ent.label_)}")

print("\nSentences:")
for s in doc.sents:
    print(" -", s.text)
```

**Observe:** the pipeline components that ran — tokenizer, tagger, parser, NER,
lemmatizer — and that `lemma_` returns real dictionary words.

## 2. Stemming vs. lemmatization

This is the comparison NVIDIA's reading list points at. Run it and remember the output.

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
wnl = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

words = ["studies", "studying", "running", "ran", "better", "geese",
         "was", "children", "universities", "caring"]

print(f"{'WORD':<15}{'PORTER STEM':<15}{'WORDNET LEMMA':<15}{'SPACY LEMMA'}")
for w in words:
    spacy_lemma = nlp(w)[0].lemma_
    print(f"{w:<15}{stemmer.stem(w):<15}{wnl.lemmatize(w, pos='v'):<15}{spacy_lemma}")
```

**What to notice:**

- `studies → studi` — the stem is **not a word**.
- `better → good` and `was → be` — lemmatization uses linguistic knowledge; stemming
  cannot do this.
- Stemming is fast and destructive; lemmatization is slower and correct.

## 3. TF-IDF and the classical pipeline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

docs = [
    "GPUs accelerate deep learning training workloads.",
    "Deep learning models require large amounts of training data.",
    "The restaurant served excellent pasta and wine.",
    "Wine pairs well with pasta at any good restaurant.",
]

vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(docs)

print("vocabulary:", vec.get_feature_names_out())
print("matrix shape:", X.shape, "| sparsity:", f"{1 - X.nnz / np.prod(X.shape):.1%}")

# Highest-weighted term per document
terms = vec.get_feature_names_out()
for i, row in enumerate(X.toarray()):
    top = terms[np.argsort(-row)[:3]]
    print(f"doc {i}: {list(top)}")
```

**Observe** the sparsity — this is the representation embeddings replace.

## 4. Similarity: sparse vs. dense

```python
from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity(X)
np.set_printoptions(precision=2, suppress=True)
print(sim)
```

The two GPU documents should score high with each other, the two restaurant documents
likewise, and across the groups near zero. Now try the failure case:

```python
pair = ["The automobile is fast.", "The car is quick."]
Xp = TfidfVectorizer().fit_transform(pair)
print("TF-IDF similarity:", cosine_similarity(Xp)[0, 1])   # ~0.0 — no shared words
```

**This is the whole motivation for embeddings** (Lab 3): TF-IDF sees no relationship
between "automobile/car" and "fast/quick".

## 5. NumPy: vectorised similarity search

```python
import numpy as np

rng = np.random.default_rng(42)
query = rng.random(768)
corpus = rng.random((10_000, 768))

def normalize(a, axis=-1):
    return a / np.linalg.norm(a, axis=axis, keepdims=True)

scores = normalize(corpus) @ normalize(query)      # (10000,) — no Python loop
top5 = np.argsort(-scores)[:5]
print("top 5 indices:", top5)
print("top 5 scores :", scores[top5].round(4))
```

Compare against a loop to feel the difference:

```python
import time

t0 = time.perf_counter()
_ = normalize(corpus) @ normalize(query)
vec_time = time.perf_counter() - t0

t0 = time.perf_counter()
qn = query / np.linalg.norm(query)
_ = [float(c @ qn / np.linalg.norm(c)) for c in corpus]
loop_time = time.perf_counter() - t0

print(f"vectorised: {vec_time*1000:.2f} ms")
print(f"loop:       {loop_time*1000:.2f} ms  ({loop_time/vec_time:.0f}x slower)")
```

## Break it on purpose

1. Remove `stop_words="english"` from the vectorizer and re-inspect the vocabulary. How
   much of it is now noise?
2. Set `ngram_range=(1,2)` and watch the vocabulary size grow. Why does `min_df` matter?
3. Run the spaCy pipeline on lowercase text with punctuation stripped. What happens to NER?
   (This is why you do **not** pre-clean text for transformers.)

## Takeaways

- spaCy gives you tokenization, POS, dependencies, **NER** and **lemmatization** in one
  pipeline.
- Stemming may produce non-words; lemmatization returns real dictionary forms.
- TF-IDF is sparse and matches only on shared tokens — it cannot see synonyms.
- Vectorised NumPy is orders of magnitude faster than looping, which is why embeddings
  search works at all.
