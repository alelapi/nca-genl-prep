# Domain 4 Quiz — Data Analysis

14 exam-style questions. Target: **12/14**.

---

**1.** Which RAPIDS library is the GPU equivalent of pandas?

- A. cuML
- B. cuDF
- C. cuGraph
- D. CuPy

??? success "Answer"
    **B.** cuDF is the GPU DataFrame library. cuML replaces scikit-learn, cuGraph replaces
    NetworkX, and CuPy replaces NumPy.

---

**2.** What distinguishes lemmatization from stemming?

- A. Stemming always produces a valid dictionary word; lemmatization does not
- B. Lemmatization uses dictionary and morphological analysis and returns a real word; stemming chops affixes heuristically and may not
- C. They are identical
- D. Stemming requires part-of-speech tags

??? success "Answer"
    **B.** "studies" → "studi" (stem) vs. "study" (lemma); "better" → "better" (stem) vs.
    "good" (lemma). Stemming is faster and cruder. spaCy provides only lemmatization.

---

**3.** R² is best described as:

- A. The correlation between two variables
- B. The proportion of variance in the target explained by the model
- C. The mean squared error
- D. The probability the model is correct

??? success "Answer"
    **B.** `R² = 1 − SS_res/SS_tot`. It can be negative on held-out data, meaning the model
    is worse than simply predicting the mean.

---

**4.** Which chart best shows the relationship between two continuous variables?

- A. Bar chart
- B. Histogram
- C. Scatter plot
- D. Pie chart

??? success "Answer"
    **C.** Histogram shows one variable's distribution; bar charts compare categories;
    scatter plots show relationships between two continuous variables.

---

**5.** A dataset has a mean of 45,000 and a median of 28,000. What does this indicate?

- A. The data is normally distributed
- B. The data is right-skewed, with large values pulling the mean up
- C. The data is left-skewed
- D. There is an error in the calculation

??? success "Answer"
    **B.** Mean ≫ median indicates right (positive) skew. Report the median, and consider a
    log transform.

---

**6.** Which loss function is most robust to outliers?

- A. MSE
- B. MAE
- C. Cross-entropy
- D. Hinge loss

??? success "Answer"
    **B.** MAE weights all errors linearly, so a single huge error does not dominate. MSE
    squares errors and chases outliers. Huber loss is the middle ground.

---

**7.** A Pearson correlation of 0 between X and Y means:

- A. X and Y are statistically independent
- B. There is no *linear* relationship, but a non-linear one may exist
- C. X causes Y with no effect
- D. The data is invalid

??? success "Answer"
    **B.** A perfect U-shaped relationship yields r ≈ 0. Always plot the data; consider
    Spearman for monotonic non-linear relationships.

---

**8.** A team applies SMOTE oversampling to the full dataset and then splits into train and
test. What is wrong?

- A. Nothing
- B. Synthetic copies derived from training rows leak into the test set, inflating results
- C. SMOTE should only be used for regression
- D. SMOTE requires GPU acceleration

??? success "Answer"
    **B.** Resampling must happen **after** the split, on the training set only. Otherwise
    the test set contains information synthesised from training data.

---

**9.** Which preprocessing step should NOT be applied before feeding text to a
transformer model like BERT?

- A. Using the model's own tokenizer
- B. Removing stop words and stemming
- C. Fixing encoding errors
- D. Removing HTML boilerplate

??? success "Answer"
    **B.** Transformers are trained on natural text and rely on function words and full
    morphology. Stop-word removal and stemming belong to classical TF-IDF pipelines.

---

**10.** cuML provides GPU-accelerated equivalents of which library?

- A. pandas
- B. NetworkX
- C. scikit-learn
- D. Matplotlib

??? success "Answer"
    **C.** cuML mirrors the scikit-learn API for regression, clustering, PCA, t-SNE, UMAP,
    k-NN and more.

---

**11.** Which statement about t-SNE is correct?

- A. It is a linear dimensionality reduction technique suitable for model input
- B. It is for visualization; distances between clusters in the plot are not meaningful
- C. It preserves global structure better than PCA
- D. It is deterministic across runs

??? success "Answer"
    **B.** t-SNE is non-linear, stochastic and tuned by perplexity. Use it (or UMAP) to
    *look* at embeddings, never as features. PCA is the linear option that can be used as
    model input.

---

**12.** A trend appears in every customer segment but reverses when all segments are
pooled. This is:

- A. Multicollinearity
- B. Simpson's paradox
- C. Survivorship bias
- D. The central limit theorem

??? success "Answer"
    **B.** Simpson's paradox — the reason aggregate analysis must be checked against
    segmented analysis before drawing conclusions.

---

**13.** Which is TRUE about using RAPIDS for a 50 MB dataset with heavy custom Python
looping?

- A. It will be dramatically faster than pandas
- B. Overhead will likely dominate; CPU pandas is probably the better choice
- C. RAPIDS cannot read data that small
- D. It requires converting the data to ONNX

??? success "Answer"
    **B.** GPU acceleration pays off on large, vectorised workloads. Small data plus
    sequential Python logic means kernel-launch and transfer overhead outweighs any gain.

---

**14.** XGBoost differs from random forest in that it:

- A. Trains many independent trees in parallel and averages them
- B. Builds trees sequentially, each correcting the errors of the previous ensemble
- C. Is a deep neural network
- D. Cannot be GPU-accelerated

??? success "Answer"
    **B.** XGBoost is **boosting** (sequential, bias reduction); random forest is
    **bagging** (parallel, variance reduction). XGBoost has native GPU support and is
    still the strongest default for tabular data.

---

## Scoring

| Score | Verdict |
| --- | --- |
| 13–14 | Solid. |
| 10–12 | Re-read RAPIDS and Statistics. |
| < 10 | Rework the chapter — these are among the easiest points on the exam. |
