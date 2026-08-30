# Data Preprocessing & Text Cleaning

*Covers task 2.3, and NVIDIA's suggested reading "Stemming and Lemmatizing With sklearn
Vectorizers".*

---

## 1. The two rules that generate most correct answers

Before any technique, two principles that appear in question after question.

**Rule 1: Split before you preprocess.** Every transform — scaler, encoder, vectorizer, imputer —
must be **fitted on the training set only**, then applied to validation and test.

**Rule 2: Fit inside the fold.** In cross-validation, the transform must be refitted within each
fold, using only that fold's training portion.

Both rules exist for the same reason: violating them is
[data leakage](../domain-1/ml-fundamentals.md#data-leakage). If you compute a TF-IDF vocabulary
over your whole dataset before splitting, your training features were shaped by the test
documents. Your reported score is inflated, nothing errors, and you will not discover it until
production.

The structural defence is **scikit-learn's `Pipeline`**, which enforces the ordering
automatically. It is a correctness feature, not a convenience one.

```text
❌ WRONG                              ✓ RIGHT
scaler.fit(X_all)                     X_tr, X_te = split(X)
X = scaler.transform(X_all)           scaler.fit(X_tr)
X_tr, X_te = split(X)                 X_tr = scaler.transform(X_tr)
                                      X_te = scaler.transform(X_te)
test statistics leaked into           test set never influenced training
the training features
```

---

## 2. Cleaning tabular data

### Missing values

Every strategy encodes an assumption, and choosing badly quietly biases the model.

| Strategy | The assumption you are making |
| --- | --- |
| Drop the rows | The missing rows are a small, **unbiased** subset |
| Drop the column | The column is mostly empty and not worth rescuing |
| Mean imputation | The distribution is roughly symmetric |
| **Median** imputation | The distribution is skewed or outlier-prone |
| Mode imputation | The variable is categorical |
| Forward/backward fill | The data is a time series and adjacent values are similar |
| Model-based (k-NN, iterative) | Missingness is predictable from the other columns |
| **Add a "was missing" indicator** | **The fact of missingness is itself informative** |

!!! tip "The strategy people forget"
    Data is often **not missing at random**. If income is missing disproportionately for people
    who declined to answer, then "declined to answer" is a **real signal** about that person.

    Mean-imputing throws that signal away and replaces it with a fabricated average. Adding a
    binary `income_was_missing` column preserves it and lets the model decide whether it matters.

### Outliers

**Detect** with the IQR rule (outside `Q1 − 1.5·IQR` to `Q3 + 1.5·IQR`), z-scores (|z| > 3), or
model-based methods (isolation forest).

**Then decide** — and this is a judgement call, not a procedure:

- **Remove** if it is a genuine error (a person aged 340, a negative price).
- **Cap / winsorize** to a percentile if the value is real but you do not want it to dominate.
- **Transform** (log) if the distribution is simply skewed rather than contaminated.
- **Keep it.** A genuine extreme value is data. In fraud detection, anomaly detection and
  reliability engineering, the outliers **are the thing you are trying to find**. Removing them
  removes the signal.

### Duplicates and consistency

**Duplicates** — remove exact duplicates, and detect near-duplicates. In LLM corpora this is
critical: duplication causes memorisation, wastes compute, and corrupts evaluation when copies
straddle the train/test split.

**Type and consistency fixes** — dates parsed as strings, numbers with thousands separators,
inconsistent categories (`"USA"` / `"U.S.A."` / `"us"`), encoding damage, stray whitespace. Dull,
and responsible for a large share of real model failures.

---

## 3. Text preprocessing — the classical pipeline

This is the pipeline that TF-IDF and classical models expect.

```text
raw text
   │
   ├─► 1. lowercasing            reduces vocabulary; loses acronym/proper-noun signal
   ├─► 2. noise removal          HTML tags, URLs, emails, boilerplate
   ├─► 3. punctuation removal    task-dependent — "!!!" matters for sentiment
   ├─► 4. unicode normalization  NFKC; consistent quotes and dashes
   ├─► 5. tokenization           split into units
   ├─► 6. stop-word removal      drop "the", "is", "and"
   ├─► 7. stemming OR lemmatization
   └─► 8. vectorization          bag-of-words, TF-IDF
```

!!! danger "Do NOT run this pipeline before a transformer"
    Modern transformer models expect **raw, natural text**. They were pretrained on it, their
    tokenizers were built for it, and their attention layers use exactly the signal this pipeline
    destroys:

    - **Stop words** carry grammatical structure that attention relies on. "The cat sat on the
      mat" and "cat sat mat" are not equivalent inputs.
    - **Casing** distinguishes "Apple" from "apple", and "US" from "us".
    - **Punctuation** carries sentence boundaries, tone and structure.
    - **Full morphology** — "running" and "ran" have different tenses, which matters.

    A question describing "preprocessing text before feeding it to BERT" wants **the model's own
    tokenizer, and nothing else**. Cleaning genuine noise (HTML tags, encoding damage,
    boilerplate) is still fine. Linguistic normalisation is not.

    The classical pipeline belongs to classical models: TF-IDF plus logistic regression, topic
    models, keyword search.

### Stemming vs. lemmatization

Named explicitly in NVIDIA's reading list, so expect a question.

| | **Stemming** | **Lemmatization** |
| --- | --- | --- |
| Method | Chops affixes using heuristic rules | Dictionary + morphological analysis, uses part of speech |
| Output | A truncated stem — **may not be a real word** | A valid dictionary word (the **lemma**) |
| Speed | Fast | Slower |
| Accuracy | Cruder | More accurate |
| Tools | NLTK (Porter, Snowball, Lancaster) | spaCy, NLTK WordNet lemmatizer |

Run the comparison and the difference is obvious:

```text
word           Porter stem     lemma
─────────      ───────────     ─────
studies        studi           study        ← stem is not a word
studying       studi           study
running        run             run
ran            ran             run          ← stemming can't do this
better         better          good         ← needs a dictionary
was            wa              be           ← needs morphology
geese          gees            goose
universities   univers         university
```

Two observations worth carrying into the exam:

- **Stemming produces non-words** (`studi`, `wa`, `univers`). That is fine if you only need
  consistent keys for matching, and useless if a human will read the output.
- **Lemmatization requires linguistic knowledge** — `better → good` and `was → be` are impossible
  with rule-based affix removal, because there are no affixes to remove.

!!! tip "The one-line exam contrast"
    **Stemming is fast, rule-based, and can produce non-words. Lemmatization is slower,
    dictionary- and POS-aware, and always produces a real word.**

    And note: **spaCy deliberately provides lemmatization only, no stemming.** NLTK provides both.

[Lab 1](../labs/01-spacy-numpy.md) runs this exact comparison so the table becomes something you
have seen rather than memorised.

### Vectorization with scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

bow   = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.9)
tfidf = TfidfVectorizer(sublinear_tf=True, max_features=50_000)
```

| Parameter | Effect |
| --- | --- |
| `ngram_range=(1,2)` | Include bigrams — captures "not good", which unigrams get backwards |
| `min_df=2` | Drop terms appearing in fewer than 2 documents — removes typos and noise |
| `max_df=0.9` | Drop terms appearing in >90% of documents — they carry no discriminative signal |
| `max_features` | Cap the vocabulary at the most frequent N |
| `stop_words` | Remove common function words |

A custom `analyzer` or `preprocessor` is where stemming or lemmatization gets plugged into these
vectorizers — which is precisely the topic of NVIDIA's suggested reading.

---

## 4. Preparing data for LLMs

Beyond the classical steps, LLM corpora need a different pipeline:

**Extraction** — clean text out of PDF, HTML, DOCX while preserving structure. Headings and tables
carry meaning that is destroyed by naive flattening.

**Quality filtering** — drop boilerplate, machine-generated spam, very short documents, and
highly repetitive text. Heuristic filters plus classifier-based ones.

**Exact and fuzzy deduplication** — hashing for exact matches, MinHash/LSH for near-duplicates.

**PII removal** — names, emails, phone numbers, identifiers. See
[Privacy & Consent](../domain-5/privacy-consent.md).

**Toxicity and safety filtering.**

**Language identification** — keep the languages you intend to support.

**Decontamination** — remove documents overlapping your evaluation benchmarks, or your metrics
measure memorisation rather than capability.

**Chunking** — for RAG; see [Embeddings](../domain-1/embeddings.md).

**NeMo Curator** is NVIDIA's GPU-accelerated tool for exactly this pipeline at corpus scale.

!!! important "Why deduplication matters more for LLMs than for classical ML"
    Three separate harms, and they compound:

    1. **Memorisation.** Content repeated many times gets reproduced verbatim, which is both a
       quality problem and a [privacy problem](../domain-5/privacy-consent.md).
    2. **Wasted compute.** You pay to train on the same text repeatedly.
    3. **Corrupted evaluation.** Duplicates straddling the train/test boundary mean your test set
       is partly a memorisation test.

    In a RAG index the harm is different but still real: a document indexed five times consumes
    five of your top-10 retrieval slots, effectively reducing *k*.

---

## 5. Data augmentation

Named in NVIDIA's course objectives: *"enhance datasets through data augmentation to improve
model accuracy."* Synthesise new training examples by transforming existing ones.

**Images** — crop, flip, rotate, colour jitter, mixup, cutout.

**Text** — harder, because meaning is fragile:

| Technique | How |
| --- | --- |
| Synonym replacement | Swap words for synonyms |
| Random insertion / swap / deletion (EDA) | Perturb the token sequence |
| **Back-translation** | Translate to another language and back — produces **natural paraphrases** |
| LLM paraphrasing | Ask a model to rewrite |
| Synthetic generation | Generate wholly new examples with a stronger model |

**Back-translation is the technique to remember** — it is the classic answer, and it works because
a round trip through another language forces a genuine rephrasing while a competent translation
system preserves the meaning.

!!! warning "Augmentation must preserve the label"
    - Flipping a photo of a cat still shows a cat. Flipping a photo of the digit "2" does not
      still show a "2".
    - Random word deletion can delete a **negation**: *"This is not good"* → *"This is good"*,
      with the sentiment label unchanged. You have just taught the model something false.
    - Back-translation occasionally drops negations or hedges for the same reason.

    Augmented data needs spot-checking, not blind trust.

---

## 6. Class imbalance

| Approach | How it works |
| --- | --- |
| **Class weights** | Penalise minority-class errors more heavily in the loss. **Usually the first thing to try** — no data is changed |
| **Oversampling** | Duplicate minority examples, or synthesise new ones with **SMOTE** (interpolating between neighbours) |
| **Undersampling** | Discard majority examples — throws away real data |
| **Threshold tuning** | Move the decision threshold rather than touching the data |
| **Metrics** | Report precision, recall, F1 or PR-AUC — **never accuracy alone** |

!!! danger "Resample the training set only — after splitting"
    Oversampling **before** splitting puts synthetic examples derived from training rows into the
    test set. Your model is then evaluated on interpolations of data it trained on, and the
    scores are spectacular and completely meaningless.

    This is a specific, common, and entirely avoidable form of leakage, and it appears as an exam
    scenario.

---

## 7. Recap

- **Split first; fit transforms on training data only; refit inside each CV fold.** Everything
  else is leakage. `Pipeline` enforces this.
- Missing-value strategy encodes an assumption — and **"was missing" is sometimes the most
  informative feature**.
- Outliers may be errors or may be the signal. Decide, do not reflexively delete.
- **Stemming**: fast, rule-based, may produce non-words. **Lemmatization**: dictionary and
  POS-aware, always a real word. spaCy offers lemmatization only.
- **The classical text pipeline helps classical models and hurts transformers.** Feed transformers
  raw text through their own tokenizer.
- LLM corpora need **deduplication** (memorisation, compute, evaluation) and **decontamination**.
- **Back-translation** is the classic text augmentation; augmentation must preserve the label.
- Handle imbalance with **class weights** first, and **resample the training set only, after
  splitting**.
