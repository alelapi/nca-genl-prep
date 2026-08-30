# Python NLP & ML Stack

*Covers tasks 1.6 and 1.10 (and 4.3, 4.6 in Software Development): "familiarity with the
capabilities of Python natural language packages (spaCy, NumPy, vector databases, etc.)"
and "use Python packages (spaCy, NumPy, Keras, etc.) to implement specific traditional
machine learning analyses".*

!!! tip "How this is examined"
    You will not be asked to write code. You will be asked **which library you would
    reach for**, and what a named library is *for*. Learn the one-line identity of each.

## The map

| Library | One-line identity |
| --- | --- |
| **NumPy** | N-dimensional arrays and vectorised numerical computing — the substrate everything else sits on |
| **pandas** | Labeled tabular data: DataFrames, joins, groupby, I/O |
| **Polars** | Faster, multi-threaded DataFrame alternative to pandas |
| **scikit-learn** | Classical ML: preprocessing, classical models, metrics, cross-validation, pipelines |
| **SciPy** | Scientific computing: statistics, optimisation, sparse matrices, signal processing |
| **Matplotlib** | Foundational plotting library |
| **Seaborn** | Statistical plots on top of Matplotlib, sensible defaults |
| **Plotly** | Interactive, web-based charts |
| **spaCy** | Industrial-strength **NLP pipeline**: tokenization, POS, dependency parsing, **NER**, lemmatization |
| **NLTK** | Academic/teaching NLP toolkit: tokenizers, stemmers, corpora, WordNet |
| **Gensim** | Topic modeling (LDA) and **Word2Vec/Doc2Vec** embeddings |
| **Hugging Face `transformers`** | Pretrained transformer models + `pipeline()` + `Trainer` |
| **Hugging Face `datasets`** | Efficient dataset loading, streaming, memory-mapping |
| **Hugging Face `tokenizers`** | Fast BPE/WordPiece/SentencePiece tokenization |
| **`sentence-transformers`** | Sentence/document **embeddings** and cross-encoder rerankers |
| **PEFT** | LoRA/QLoRA and other parameter-efficient fine-tuning |
| **PyTorch** | The dominant deep-learning framework; dynamic graphs, research standard |
| **TensorFlow** | Google's DL framework; production tooling (TF Serving, TFLite) |
| **Keras** | High-level neural-network API (`Sequential`, `Model`, `.fit()`); runs on TF/JAX/PyTorch |
| **XGBoost / LightGBM** | Gradient-boosted decision trees — still the best default for tabular data |
| **LangChain / LlamaIndex** | Orchestration for LLM apps: chains, retrievers, agents, RAG plumbing |
| **LangGraph** | Graph-structured, stateful agent orchestration |
| **FAISS / Milvus / Qdrant / Chroma / pgvector** | Vector storage and similarity search |
| **RAPIDS (cuDF, cuML, cuGraph)** | GPU-accelerated pandas/scikit-learn/NetworkX equivalents |
| **NeMo** | NVIDIA's framework to build, customize, align and export generative AI models |
| **TensorRT / TensorRT-LLM** | Inference optimization and runtime |
| **Triton Inference Server** | Production model serving |
| **MLflow / Weights & Biases** | Experiment tracking, model registry |

## NumPy — why it matters here

NumPy is named in the blueprint twice. The point is not `np.array`; it is that
**embeddings are NumPy arrays** and that vectorised operations run orders of magnitude
faster than Python loops.

```python
import numpy as np

a = np.random.rand(768)          # an embedding
b = np.random.rand(768)

cosine = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

# Batch similarity: one query against 10,000 documents, no loops
docs = np.random.rand(10_000, 768)
docs_n = docs / np.linalg.norm(docs, axis=1, keepdims=True)
q_n = a / np.linalg.norm(a)
scores = docs_n @ q_n            # (10000,)
top5 = np.argsort(-scores)[:5]
```

Concepts to recognise: `ndarray`, `dtype`, `shape`, **broadcasting**, axis-wise
reductions, vectorisation, views vs. copies.

## spaCy — the NLP pipeline

spaCy is named more than any other NLP library in the blueprint. Its identity:
**a fast, production-oriented pipeline that turns raw text into linguistic annotations.**

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("NVIDIA released new GPUs in Santa Clara in March 2026.")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.is_stop)

for ent in doc.ents:
    print(ent.text, ent.label_)      # NVIDIA ORG · Santa Clara GPE · March 2026 DATE
```

The default pipeline components: **tokenizer → tagger → parser → NER →
lemmatizer** (plus optional components).

| spaCy gives you | Meaning |
| --- | --- |
| Tokenization | Rule-based, language-specific splitting |
| **Lemmatization** | "running" → "run" — dictionary-based, returns a real word |
| POS tagging | Part of speech |
| Dependency parsing | Grammatical structure between tokens |
| **NER** | Named entities: PERSON, ORG, GPE, DATE, MONEY… |
| Sentence segmentation | `doc.sents` |
| Similarity / vectors | Static vectors in `md`/`lg` models |

!!! note "spaCy vs. NLTK"
    **spaCy** = production, opinionated, fast, one best algorithm per task, object-oriented
    `Doc`/`Token` API. **NLTK** = research/teaching, many alternative algorithms, string
    in/string out, includes **stemmers** (spaCy deliberately offers only lemmatization).

    **spaCy vs. Hugging Face**: spaCy for classical linguistic annotation at scale;
    `transformers` for pretrained deep models. They compose — `spacy-transformers` puts a
    transformer inside a spaCy pipeline.

## scikit-learn — traditional ML (task 1.10)

The canonical shape of a classical text-classification pipeline:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000)),
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="f1_macro").mean())
pipe.fit(X_tr, y_tr)
print(classification_report(y_te, pipe.predict(X_te)))
```

Why `Pipeline` matters conceptually: it fits every transform **inside** each
cross-validation fold, which is what prevents [data leakage](ml-fundamentals.md).

Key modules: `preprocessing` (scalers, encoders), `feature_extraction.text`
(`CountVectorizer`, `TfidfVectorizer`), `model_selection` (splits, CV, `GridSearchCV`),
`metrics`, `decomposition` (PCA), `cluster` (k-means).

## Keras / PyTorch — deep learning

**Keras** is named in the blueprint. Recognise its high-level API:

```python
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(784,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

**PyTorch** is the framework the LLM ecosystem is actually built on (Hugging Face, NeMo,
Megatron). Recognise `nn.Module`, `forward()`, `loss.backward()`, `optimizer.step()`,
`DataLoader`, `.to("cuda")`.

## Hugging Face — the LLM workhorse

```python
from transformers import pipeline

clf = pipeline("sentiment-analysis")
gen = pipeline("text-generation", model="gpt2")
ner = pipeline("ner", aggregation_strategy="simple")
qa  = pipeline("question-answering")
summ = pipeline("summarization")
```

And embeddings:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vecs = model.encode(["first chunk", "second chunk"], normalize_embeddings=True)
```

The **Hub** hosts models, datasets and Spaces; `AutoModel`/`AutoTokenizer` load anything
by name. NVIDIA's course objective names this explicitly: *"find, pull in, and experiment
with the HuggingFace model repository and Transformers API."*

## Orchestration: LangChain and LangGraph

Named in two separate NVIDIA course objectives — *"be proficient in using LangChain to
organize and compose LLM workflows"* and *"explore using LangChain and LangGraph for
orchestrating data pipelines and environment-enabled agents."*

What LangChain provides: **document loaders**, **text splitters**, **embedding wrappers**,
**vector-store adapters**, **retrievers**, **prompt templates**, **chains** (composed
calls), **memory**, **agents and tools**, **output parsers**. Its value is that it
standardises the interfaces so components are swappable — not that it does anything you
could not write yourself.

**LangGraph** models an agent as a **graph of nodes with shared state**, allowing cycles,
branching and human-in-the-loop checkpoints — things a linear chain cannot express.

**LlamaIndex** is the alternative, more focused on indexing and retrieval structures.

## Key takeaways

- **NumPy** = arrays and vectorised math (embeddings live here). **pandas** = tables.
- **spaCy** = production NLP pipeline (tokenize, POS, parse, **NER**, lemmatize);
  **NLTK** = academic toolkit with stemmers; **Gensim** = Word2Vec and topic models.
- **scikit-learn** = classical ML + TF-IDF + cross-validation; use `Pipeline` to avoid leakage.
- **Keras** = high-level DL API; **PyTorch** = what the LLM stack is built on.
- **Hugging Face** `transformers` + `sentence-transformers` = pretrained models and embeddings.
- **LangChain/LangGraph** = orchestration; **FAISS/Milvus/Qdrant/pgvector** = vector storage.
- **RAPIDS** = the GPU versions of pandas/scikit-learn/NetworkX.
