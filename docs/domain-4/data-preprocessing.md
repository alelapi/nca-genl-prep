# Data Preprocessing & Text Cleaning

*Covers task 2.3, and NVIDIA's suggested reading "Stemming and Lemmatizing With sklearn
Vectorizers".*

## The general pipeline

```text
raw data → inspect → clean → handle missing → handle outliers → transform
        → encode → split → (augment) → model-ready
```

Two rules that produce most of the exam's correct answers:

1. **Fit every transform on the training set only**, then apply to validation/test.
   Fitting a scaler or vectorizer on all the data is [leakage](../domain-1/ml-fundamentals.md#data-splits).
2. **Split before you preprocess** — or use a `Pipeline`, which enforces the order for you.

## Cleaning tabular data

**Missing values**

| Strategy | When |
| --- | --- |
| Drop rows | Few missing, plenty of data |
| Drop columns | The column is mostly empty |
| Mean/median imputation | Numeric; median when skewed or outlier-prone |
| Mode imputation | Categorical |
| Forward/backward fill | Time series |
| Model-based (kNN, iterative) | Missingness is structured and the column matters |
| **Missingness indicator** | When *the fact that it is missing* is itself informative |

**Outliers** — detect with the IQR rule (outside `Q1 − 1.5·IQR` … `Q3 + 1.5·IQR`),
z-scores (|z| > 3), or model-based methods (isolation forest). Then decide: remove
(if erroneous), cap/winsorize, transform (log), or keep — a genuine extreme value is data,
not noise.

**Duplicates** — remove exact duplicates, and detect near-duplicates. In LLM datasets this
is critical: duplication causes memorisation, wastes compute, and corrupts evaluation when
duplicates straddle the train/test boundary.

**Type and consistency fixes** — dates parsed as strings, numbers with thousands
separators, inconsistent categories ("USA"/"U.S.A."/"us"), encoding errors, whitespace.

## Text preprocessing

The classical NLP pipeline, still fully examinable:

1. **Lowercasing** — reduces vocabulary; loses acronym/proper-noun signal.
2. **Noise removal** — HTML tags, URLs, emails, boilerplate.
3. **Punctuation and special characters** — task-dependent (sentiment analysis may need
   "!!!" and emoji).
4. **Unicode normalization** — NFKC; consistent quotes and dashes.
5. **Tokenization** — split into units.
6. **Stop-word removal** — drop "the", "is", "and". Useful for TF-IDF/topic models;
   **harmful** for transformers, which need the full sentence.
7. **Stemming or lemmatization** — see below.
8. **Vectorization** — bag-of-words, TF-IDF, or embeddings.

!!! danger "Do NOT apply this pipeline before a transformer"
    Modern transformer models expect **raw, natural text**. Lowercasing, stop-word removal
    and stemming destroy information their tokenizers and attention layers rely on. The
    classical pipeline belongs to classical models (TF-IDF + logistic regression, topic
    models, keyword search).

    A question describing "preprocessing text before feeding BERT" wants **the model's own
    tokenizer**, nothing more.

### Stemming vs. lemmatization

Named explicitly in NVIDIA's reading list — expect a question.

| | **Stemming** | **Lemmatization** |
| --- | --- | --- |
| Method | Chops affixes using heuristic rules | Dictionary + morphological analysis, uses part of speech |
| Output | A truncated stem, **may not be a real word** | A valid dictionary word (the **lemma**) |
| Speed | Fast | Slower |
| Accuracy | Cruder | More accurate |
| Examples | "studies" → "studi", "running" → "run", "better" → "better" | "studies" → "study", "running" → "run", "better" → "good" |
| Tools | NLTK (Porter, Snowball, Lancaster) | spaCy, NLTK WordNet lemmatizer |

!!! tip "The exam-ready contrast"
    **Stemming is fast and crude and can produce non-words. Lemmatization is slower,
    linguistically informed, and always produces a real word.**

    Note that **spaCy deliberately provides only lemmatization** — a design choice worth
    remembering.

### Vectorization with scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

bow = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.9)
tfidf = TfidfVectorizer(sublinear_tf=True, max_features=50_000)
```

Key parameters worth recognising: `ngram_range` (capture short phrases), `min_df`
(drop rare terms — noise and vocabulary bloat), `max_df` (drop near-universal terms),
`max_features` (cap vocabulary size), `stop_words`.

A custom analyzer or preprocessor is where stemming/lemmatization gets plugged into these
vectorizers — precisely the topic of NVIDIA's suggested reading.

## Preparing data for LLMs

Beyond the classical steps, LLM corpora need:

- **Extraction** — clean text out of PDF/HTML/DOCX while preserving structure.
- **Quality filtering** — drop boilerplate, machine-generated spam, very short or
  very repetitive documents; heuristic and classifier-based filters.
- **Exact and fuzzy deduplication** — hashing plus MinHash/LSH.
- **PII removal** — names, emails, phone numbers, IDs. See
  [Privacy & Consent](../domain-5/privacy-consent.md).
- **Toxicity and safety filtering.**
- **Language identification** — keep the languages you intend.
- **Decontamination** — remove documents overlapping evaluation benchmarks.
- **Chunking** — for RAG; see [Embeddings](../domain-1/embeddings.md).

**NeMo Curator** is NVIDIA's GPU-accelerated tool for exactly this pipeline at scale.

## Data augmentation

Named in NVIDIA's course objectives (*"enhance datasets through data augmentation to
improve model accuracy"*).

- **Images** — crop, flip, rotate, colour jitter, mixup, cutout.
- **Text** — synonym replacement, random insertion/swap/deletion (EDA),
  **back-translation** (translate to another language and back — produces natural
  paraphrases), paraphrasing with an LLM, and **synthetic generation** with a stronger
  model.
- **Purpose:** more effective training data, better generalisation, less overfitting, and
  a way to rebalance rare classes.
- **Caveat:** augmentation must preserve the label. Randomly swapping words in a sentiment
  example can flip its meaning; back-translation can drop negations.

## Handling class imbalance

- **Resampling** — oversample the minority (**SMOTE** synthesises interpolated examples),
  or undersample the majority.
- **Class weights** — penalise minority errors more in the loss. Usually the first thing
  to try.
- **Threshold tuning** — move the decision threshold instead of the data.
- **Metrics** — always report precision/recall/F1/PR-AUC, never accuracy alone.

!!! warning "Resample only the training set"
    Oversampling before splitting puts synthetic copies of training rows into the test set,
    producing spectacular and completely fake scores.

## Key takeaways

- Split first, fit transforms on training data only — everything else is leakage.
- **Stemming** = fast, rule-based, may produce non-words. **Lemmatization** = dictionary
  and POS-aware, always a real word.
- Stop-word removal and stemming help classical models and **hurt transformers**.
- Deduplication and decontamination are non-negotiable for LLM corpora.
- **Back-translation** is the classic text augmentation technique.
- Handle imbalance with class weights or resampling — **on the training set only**.
