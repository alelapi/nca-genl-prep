# Python NLP & ML Stack

*Covers tasks 1.6 and 1.10 (and 4.3, 4.6 under Software Development): "familiarity with the
capabilities of Python natural language packages (spaCy, NumPy, vector databases, etc.)" and
"use Python packages (spaCy, NumPy, Keras, etc.) to implement specific traditional machine
learning analyses".*

!!! tip "How this is actually examined"
    You will not write code in the exam. You will be asked **which library you would reach
    for**, and **what a named library is for**. The failure mode is not "I couldn't write the
    code" — it is "I couldn't tell cuML from cuGraph" or "I thought Gensim was a plotting
    library".

    So the goal of this page is a **one-line identity** for each library that you can recall
    instantly, plus enough context on the important ones that the identity sticks.

---

## 1. The map

Learn this table. It is the highest ratio of marks to effort in Domain 1.

| Library | One-line identity |
| --- | --- |
| **NumPy** | N-dimensional arrays and vectorised numerical computing — the substrate everything else sits on |
| **pandas** | Labeled tabular data: DataFrames, joins, groupby, I/O |
| **Polars** | Faster, multi-threaded DataFrame alternative to pandas |
| **scikit-learn** | Classical ML: preprocessing, classical models, metrics, cross-validation, pipelines |
| **SciPy** | Scientific computing: statistics, optimisation, sparse matrices, signal processing |
| **Matplotlib** | The foundational plotting library |
| **Seaborn** | Statistical plots on top of Matplotlib, with good defaults |
| **Plotly** | Interactive, web-based charts |
| **spaCy** | Industrial-strength **NLP pipeline**: tokenization, POS, dependency parsing, **NER**, lemmatization |
| **NLTK** | Academic/teaching NLP toolkit: tokenizers, **stemmers**, corpora, WordNet |
| **Gensim** | Topic modeling (LDA) and **Word2Vec/Doc2Vec** embeddings |
| **Hugging Face `transformers`** | Pretrained transformer models, `pipeline()`, `Trainer` |
| **Hugging Face `datasets`** | Efficient dataset loading, streaming, memory-mapping |
| **Hugging Face `tokenizers`** | Fast BPE / WordPiece / SentencePiece tokenization |
| **`sentence-transformers`** | Sentence and document **embeddings**, plus cross-encoder rerankers |
| **PEFT** | LoRA, QLoRA and other parameter-efficient fine-tuning |
| **PyTorch** | The dominant deep-learning framework; what the LLM ecosystem is built on |
| **TensorFlow** | Google's DL framework; strong production tooling (TF Serving, TFLite) |
| **Keras** | High-level neural-network API (`Sequential`, `.fit()`); runs on TF / JAX / PyTorch |
| **XGBoost / LightGBM** | Gradient-boosted decision trees — still the best default for tabular data |
| **LangChain / LlamaIndex** | Orchestration for LLM apps: chains, retrievers, agents, RAG plumbing |
| **LangGraph** | Graph-structured, stateful agent orchestration |
| **FAISS / Milvus / Qdrant / Chroma / pgvector** | Vector storage and similarity search |
| **RAPIDS (cuDF, cuML, cuGraph)** | GPU-accelerated pandas / scikit-learn / NetworkX |
| **NeMo** | NVIDIA's framework to build, customize, align and export generative AI models |
| **TensorRT / TensorRT-LLM** | Inference optimization and runtime |
| **Triton Inference Server** | Production model serving |
| **MLflow / Weights & Biases** | Experiment tracking, model registry |

---

## 2. NumPy — why the blueprint names it twice

The point of NumPy in an LLM context is not `np.array`. It is that **an embedding is a NumPy
array**, and that similarity search is a matrix multiplication.

Two concepts underpin everything:

**Vectorisation.** NumPy operations run in compiled C over whole arrays instead of looping in
Python. The difference is not marginal — it is typically 50–200×, which is the difference
between a search that returns in milliseconds and one that returns in a minute.

```python
import numpy as np

query = np.random.rand(768)              # one embedding
docs  = np.random.rand(10_000, 768)      # a corpus of embeddings

def normalize(a, axis=-1):
    return a / np.linalg.norm(a, axis=axis, keepdims=True)

scores = normalize(docs) @ normalize(query)   # (10000,) — one matmul, no Python loop
top5   = np.argsort(-scores)[:5]
```

That `@` is the entire similarity search from [Embeddings](embeddings.md), and it is why cosine
similarity over normalised vectors is the standard: it reduces to a single dot product, which
is the operation hardware is best at.

**Broadcasting.** NumPy automatically expands compatible shapes so operations between arrays of
different dimensions work without explicit loops — `(10000, 768)` divided by `(10000, 1)`
normalises every row.

Terms to recognise: `ndarray`, `dtype`, `shape`, broadcasting, axis-wise reductions,
vectorisation, views vs. copies.

[Lab 1](../labs/01-spacy-numpy.md) times the vectorised version against a Python loop so you
see the gap.

---

## 3. spaCy — the NLP pipeline

spaCy is named more often in the blueprint than any other NLP library. Its identity:
**a fast, production-oriented pipeline that turns raw text into linguistic annotations.**

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("NVIDIA released new GPUs in Santa Clara in March 2026.")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.is_stop)

for ent in doc.ents:
    print(ent.text, ent.label_)
    # NVIDIA      ORG
    # Santa Clara GPE
    # March 2026  DATE
```

Calling `nlp(text)` runs a pipeline of components in order:

```text
raw text ─► tokenizer ─► tagger ─► parser ─► NER ─► lemmatizer ─► Doc object
              │            │         │         │        │
              │            │         │         │        └─ "running" → "run"
              │            │         │         └────────── PERSON, ORG, GPE, DATE, MONEY
              │            │         └──────────────────── grammatical structure
              │            └────────────────────────────── part of speech
              └─────────────────────────────────────────── language-specific splitting
```

Everything hangs off the resulting `Doc`: `doc.ents` for entities, `doc.sents` for sentences,
`doc.noun_chunks` for noun phrases, and per-token attributes for the rest.

### spaCy vs. NLTK vs. Hugging Face

A comparison that gets tested, because all three are "NLP libraries".

| | **spaCy** | **NLTK** | **Hugging Face `transformers`** |
| --- | --- | --- | --- |
| Built for | Production | Teaching and research | Deep learning models |
| Philosophy | One best algorithm per task | Many alternative algorithms to compare | Access to pretrained models |
| API style | Object-oriented (`Doc`, `Token`, `Span`) | Functions on strings | Model + tokenizer objects |
| Speed | Fast | Slow | Depends on the model and hardware |
| **Stemming** | **Deliberately not provided** | Porter, Snowball, Lancaster | Not applicable |
| Lemmatization | Yes | Yes (WordNet) | Not applicable |
| Typical use | NER, parsing, large-scale text processing | Learning NLP, quick experiments | Classification, generation, embeddings |

That "deliberately not provided" row is a genuine design statement: spaCy's authors consider
stemming a crude approximation and offer only lemmatization. See
[Data Preprocessing](../domain-4/data-preprocessing.md) for why the distinction matters.

The three are complementary, not competing. In a real system you might use spaCy to segment and
extract entities, then a Hugging Face model to embed the results.

---

## 4. scikit-learn — traditional ML (task 1.10)

The canonical shape of a classical text-classification pipeline, which is exactly what task 1.10
describes:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")),
    ("clf",   LogisticRegression(max_iter=1000)),
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="f1_macro")
print(f"{scores.mean():.3f} ± {scores.std():.3f}")     # report variance, not just the mean

pipe.fit(X_tr, y_tr)
print(classification_report(y_te, pipe.predict(X_te)))
```

!!! important "Why `Pipeline` is a correctness feature, not a convenience"
    Inside `cross_val_score`, the `Pipeline` fits the **TF-IDF vectorizer separately within each
    fold**, using only that fold's training data.

    Do it manually — vectorise everything, then cross-validate — and the vocabulary and IDF
    weights were computed using the validation data. That is **[data
    leakage](ml-fundamentals.md#data-leakage)**, your scores are inflated, and nothing warns
    you.

    A question describing "fitting the scaler on the full dataset before splitting" is asking
    about exactly this.

The modules worth recognising: `preprocessing` (scalers, encoders), `feature_extraction.text`
(`CountVectorizer`, `TfidfVectorizer`), `model_selection` (splits, CV, `GridSearchCV`),
`metrics`, `decomposition` (PCA), `cluster` (k-means), `ensemble` (random forest).

---

## 5. Deep learning frameworks

### Keras — named in the blueprint

Keras is the high-level API. Recognise its shape:

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(784,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

Note how directly this maps onto [Neural Networks](neural-networks.md): `Dense` layers,
`relu` hidden activation, `Dropout` regularization, `softmax` output for multi-class,
`sparse_categorical_crossentropy` loss, `adam` optimizer. If you can read this block and name
what each argument is doing, you have the concepts.

### PyTorch — what the ecosystem actually runs on

Hugging Face, NeMo and Megatron are all PyTorch. Recognise the idioms:

```python
model.train()                      # enables dropout / batchnorm training behaviour
for batch in dataloader:
    optimizer.zero_grad()          # clear gradients from the previous step
    loss = criterion(model(batch.x), batch.y)
    loss.backward()                # backpropagation — compute gradients
    optimizer.step()               # apply the update

model.eval()                       # disables dropout — forgetting this is a real bug
with torch.no_grad():              # skip gradient tracking; saves memory at inference
    preds = model(x.to("cuda"))
```

**TensorFlow vs. PyTorch:** TensorFlow has stronger deployment tooling (TF Serving, TFLite);
PyTorch dominates research and the LLM ecosystem. For this exam, know that **PyTorch is what
the generative AI stack is built on**.

---

## 6. Hugging Face — the LLM workhorse

NVIDIA's course objective names it directly: *"find, pull in, and experiment with the
HuggingFace model repository and Transformers API."*

**The `pipeline()` abstraction** — one line per task, tokenizer and model handled for you:

```python
from transformers import pipeline

clf  = pipeline("sentiment-analysis")
gen  = pipeline("text-generation", model="gpt2")
ner  = pipeline("ner", aggregation_strategy="simple")
qa   = pipeline("question-answering")
summ = pipeline("summarization")
zs   = pipeline("zero-shot-classification")
```

**The `Auto` classes** — load any model from the Hub by name:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
```

The `AutoModelFor...` suffix selects the task head: `ForCausalLM` (generation),
`ForSequenceClassification`, `ForTokenClassification` (NER), `ForQuestionAnswering`,
`AutoModel` (raw hidden states, for embeddings).

**`sentence-transformers`** for embeddings — this is the library you actually use for RAG:

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

embedder = SentenceTransformer("all-MiniLM-L6-v2")
vectors  = embedder.encode(chunks, normalize_embeddings=True)     # bi-encoder → index

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores   = reranker.predict([(query, c) for c in candidates])     # cross-encoder → rerank
```

Note the two classes correspond exactly to the bi-encoder/cross-encoder distinction from
[Embeddings](embeddings.md).

**The Hub** hosts models, datasets and Spaces; `datasets` handles loading and streaming large
corpora without loading them into RAM.

---

## 7. Orchestration: LangChain and LangGraph

Named in two separate NVIDIA course objectives — *"be proficient in using LangChain to organize
and compose LLM workflows"* and *"explore using LangChain and LangGraph for orchestrating data
pipelines and environment-enabled agents."*

**What LangChain provides:** document loaders, text splitters, embedding wrappers, vector-store
adapters, retrievers, prompt templates, chains (composed calls), memory, agents and tools,
output parsers.

Its value is **standardised interfaces**, not novel capability. Every component is swappable —
change three lines and your retriever goes from Chroma to Milvus, or your model from OpenAI to
a local NIM endpoint. You could write all of it yourself; the framework saves you from writing
it five times.

**LangGraph** models an agent as a **graph of nodes over shared state**, which allows cycles,
conditional branching and human-in-the-loop checkpoints — things a linear chain structurally
cannot express. "Retrieve, check if sufficient, retrieve again if not" is a cycle, and that is
the case LangGraph exists for.

**LlamaIndex** is the main alternative, more focused on indexing and retrieval structures.

!!! note "A practical caveat worth having an opinion on"
    Frameworks add abstraction between you and the model, which helps when you are assembling
    known components and hurts when you are debugging why retrieval is bad. [Lab
    4](../labs/04-rag.md) deliberately builds RAG without a framework, so that you know what the
    framework is doing for you before you let it do it.

---

## 8. Choosing the right tool — worked scenarios

The exam asks "which library would you use to…", so practise the mapping:

| Task | Reach for |
| --- | --- |
| Extract company and person names from 100k documents | **spaCy** (NER) |
| Compute cosine similarity between 50k embeddings | **NumPy** (vectorised matmul) |
| Classify support tickets with 2,000 labeled examples | **scikit-learn** (TF-IDF + logistic regression) — or embeddings + logistic regression |
| Generate text from a pretrained model | **transformers** (`AutoModelForCausalLM`) |
| Turn document chunks into vectors for RAG | **sentence-transformers** |
| Store and search 5M vectors | **FAISS / Milvus / Qdrant / pgvector** |
| Rerank the top 50 retrieved candidates | **sentence-transformers `CrossEncoder`** |
| Fine-tune a 7B model on one GPU | **PEFT** (LoRA/QLoRA) + **transformers** |
| Train Word2Vec on a domain corpus | **Gensim** |
| Predict churn from a tabular dataset | **XGBoost** |
| Speed up a pandas pipeline on 50 GB | **cuDF** (RAPIDS) |
| Compute PageRank over a 10M-edge graph | **cuGraph** (RAPIDS) |
| Chain retrieval, prompting and tool calls | **LangChain / LangGraph** |
| Track experiments across many runs | **MLflow / Weights & Biases** |
| Optimize a trained model for GPU inference | **TensorRT / TensorRT-LLM** |
| Serve several models in production | **Triton Inference Server** |

---

## 9. Recap

- **NumPy** = arrays and vectorised math — embeddings live here, and similarity search is one
  matrix multiplication. **pandas** = tables.
- **spaCy** = production NLP pipeline (tokenize → POS → parse → **NER** → lemmatize), and it
  offers **lemmatization only, no stemming**. **NLTK** = academic toolkit *with* stemmers.
  **Gensim** = Word2Vec and topic models.
- **scikit-learn** = classical ML, TF-IDF, cross-validation. Use `Pipeline` — it prevents
  leakage, which is a correctness issue, not a style one.
- **Keras** = high-level DL API; **PyTorch** = what the LLM stack is actually built on.
- **Hugging Face**: `transformers` for models, `sentence-transformers` for embeddings and
  reranking, `datasets` for data, `PEFT` for LoRA.
- **LangChain/LangGraph** = orchestration and standardised interfaces; **FAISS/Milvus/Qdrant/
  pgvector** = vector storage.
- **RAPIDS**: cuDF → pandas, cuML → scikit-learn, cuGraph → NetworkX.
