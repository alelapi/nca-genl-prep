# Privacy, Consent & Data Governance

*Covers task 5.2: "describe the balance between data privacy and the importance of data consent."*

Note the blueprint's word: **balance**. Questions here are about **trade-offs**, not absolutes.
The answer is essentially never "collect everything" and essentially never "collect nothing".

---

## 1. The tension

AI systems improve with more data. Individuals have a right to control data about themselves.
Both claims are legitimate, and they genuinely pull against each other.

| Pulls toward MORE data | Pulls toward LESS |
| --- | --- |
| Better accuracy and coverage | Individual privacy rights |
| **Fewer blind spots for minority groups** | Regulatory obligations (GDPR, CCPA, HIPAA) |
| More useful, personalised systems | Breach and misuse risk |
| Faster iteration | Loss of user trust, which is not recoverable |

That second row on the left is worth pausing on, because it is the genuinely hard part: **privacy
protection and fairness can conflict.** Collecting less demographic data protects privacy — and
makes it impossible to measure whether your model works equally well across groups. You cannot
run the [disaggregated evaluation](bias-fairness.md) that fairness requires without the attribute
that privacy suggests you should not collect.

There is no clean resolution. There are careful choices: collect the minimum needed for the
measurement, keep it separated and access-controlled, use it only for fairness auditing, and
document all of that.

!!! tip "The shape of the correct exam answer"
    **Collect the minimum necessary, with informed consent, for a stated purpose, retained for a
    limited time, with technical safeguards.**

    Answer options describing that shape are usually right. Options at either extreme usually are
    not.

---

## 2. Consent

**Informed consent** means the person understands and freely agrees to a specific use. Five
properties, all of which can be tested:

- **Informed** — plain language, not buried on page 34 of a terms-of-service document. If nobody
  can reasonably be expected to have understood it, it is not informed.
- **Specific** — tied to a stated purpose. Consent for order fulfilment is **not** consent for
  model training.
- **Freely given** — not coerced by making the service unusable otherwise.
- **Revocable** — withdrawal must be possible **and must actually take effect**.
- **Documented** — you must be able to demonstrate what was consented to, and when.

**Opt-in vs. opt-out.** Opt-in requires an explicit affirmative action and is the
privacy-preserving default that regulation increasingly requires. Opt-out infers consent from
silence, which conflates "agreed" with "did not notice".

### Secondary use — the recurring problem

Data collected for one purpose, later used to train a model, is a **new purpose** and generally
requires a new legal basis.

*"We already have the data"* is a statement about your database, not a justification. This is the
single most common real-world consent failure, and it is easy to fall into precisely because the
data is right there and the new use feels harmless.

!!! danger "Consent and LLM training corpora"
    Web-scraped corpora contain personal data, copyrighted work, and content whose authors never
    contemplated model training as a use. This is an unresolved legal and ethical question, and
    it is a large part of why NVIDIA lists privacy and consent as a certification topic at all.

    The defensible professional position: **know the provenance of your training data**, respect
    robots.txt and licences, honour deletion and opt-out requests, and document your basis. "It
    was on the internet" is not a legal basis.

---

## 3. Personal data and PII

**PII** identifies a person either **directly** (name, email, phone, national ID, biometrics) or
**indirectly in combination**. That second category is under-appreciated: postcode + date of birth
+ gender is famously sufficient to uniquely identify a large majority of a population, despite
none of those three being identifying on its own.

**Special categories** requiring stronger protection: health, biometric and genetic data, racial
or ethnic origin, political opinions, religious beliefs, trade union membership, sexual
orientation.

### Handling PII across the LLM pipeline

The point is that PII appears at **every** stage, not just in the training set:

| Stage | What to do |
| --- | --- |
| **Training data** | Detect and redact — regex plus NER (spaCy, Presidio). **NeMo Curator** includes PII removal |
| **RAG index** | Same, plus store access-control metadata per chunk |
| **Inference — input** | Filter prompts for PII before they reach a third-party model |
| **Inference — output** | Filter responses, in case the model reproduces something it should not |
| **Logs and traces** | **Redact before storing.** See below |
| **In transit and at rest** | Encrypt (TLS, disk encryption) |
| **Retrieval** | Enforce per-user permissions as **metadata filters at query time** |

!!! warning "Two failures teams reliably make"
    **1. Prompt logs.** You log the full request trace for debugging — prompt, retrieved chunks,
    response. That log now contains every piece of personal data any user ever typed, in
    plaintext, in a system with looser access controls than your production database. Redact
    before storing, and set a retention policy.

    **2. Trusting the model to keep secrets.** Access control belongs in the **retrieval layer**,
    as metadata filters, not in the prompt. Instructing a model "do not reveal HR documents to
    non-HR staff" while placing an HR document in its context is not access control — it is a
    request. **If a chunk reaches the prompt, treat it as disclosed.**

---

## 4. Regulatory vocabulary

You do not need legal expertise. You need to recognise these terms in answer options.

| Concept | Meaning |
| --- | --- |
| **Data minimisation** | Collect only what is necessary for the stated purpose |
| **Purpose limitation** | Use it only for that purpose |
| **Storage limitation** | Keep it only as long as needed |
| **Right of access** | People can see what you hold about them |
| **Right to erasure** ("right to be forgotten") | People can require deletion |
| **Right to rectification** | Correction of inaccurate data |
| **Data portability** | Export in a usable format |
| **Right not to be subject to solely automated decisions** | Human review of significant automated decisions |
| **Privacy by design / by default** | Build protection in from the start; the default setting is the private one |
| **DPIA** | Data Protection Impact Assessment, for high-risk processing |
| **Data residency / sovereignty** | Data must remain within a jurisdiction |

Frameworks by name: **GDPR** (EU), **CCPA/CPRA** (California), **HIPAA** (US health), **EU AI
Act** (risk-tiered AI regulation), **NIST AI Risk Management Framework**, **ISO/IEC 42001**.

### The right to erasure meets trained weights

!!! important "A conflict with no clean technical solution"
    Deleting a person's record from your database is trivial. Removing their **influence from a
    trained model's weights** is not — their data contributed to millions of gradient updates,
    diffused across billions of parameters. There is no "delete this person" operation.

    Retraining is the only reliable remedy, and retraining a foundation model to honour one
    deletion request is not viable. **Machine unlearning** — approximating deletion without full
    retraining — is an active research area, not a solved problem.

    **This is a strong, practical argument for RAG over fine-tuning wherever personal data is
    involved.** Delete the document from the index and the system stops using it **immediately
    and verifiably**. That property is worth a great deal in a regulated environment, and it is
    exactly the sort of trade-off this exam asks about.

---

## 5. Memorisation and extraction

LLMs can **memorise and reproduce training data verbatim**. Demonstrated extraction attacks have
recovered names, addresses, phone numbers and API keys from publicly released models.

**The main driver is duplication.** Content that appeared many times in the corpus is far more
likely to be memorised than content that appeared once — the model saw the same exact token
sequence repeatedly and learned it as a sequence rather than as a pattern. Highly distinctive
content (a specific 16-digit number) is also more vulnerable than generic text.

**Mitigations, in order of effect:**

1. **Deduplicate the training corpus.** The single most effective measure, and it also improves
   quality and saves compute — see [Data Preprocessing](../domain-4/data-preprocessing.md).
2. **Remove PII before training.**
3. **Differential privacy** during training.
4. **Filter outputs** at inference.
5. **Red-team for extraction** — actively try to make your model leak, and treat every success as
   a regression test.

---

## 6. Privacy-preserving techniques

| Technique | The idea | The caveat |
| --- | --- | --- |
| **Anonymisation** | Irreversibly remove identifiers | Genuinely hard — re-identification by combining quasi-identifiers is common and well-documented |
| **Pseudonymisation** | Replace identifiers with tokens | **Still personal data under GDPR** — it is reversible with the key |
| **Aggregation** | Report only group-level statistics | Small groups can still be re-identified |
| **Differential privacy** | Add calibrated noise so that no individual record measurably changes the output | A **formal mathematical guarantee**; costs accuracy (the privacy budget ε) |
| **Federated learning** | Train across devices or institutions; **only model updates leave**, never raw data | Updates can still leak information — combine with DP and secure aggregation |
| **Secure enclaves / confidential computing** | Compute on encrypted data in hardware-protected memory | Performance and platform constraints |
| **Homomorphic encryption** | Compute directly on encrypted data | Still impractical for large models |
| **Synthetic data** | Train on generated data resembling the real distribution | Can leak the original if the generator memorised it |

!!! tip "The two to be able to define on sight"
    **Differential privacy** — a mathematical guarantee that the presence or absence of any single
    individual's data does not meaningfully change the result. You can therefore say something
    formally provable about what an attacker can learn.

    **Federated learning** — the data **never leaves** the device or institution; only model
    updates are shared and aggregated centrally. The canonical use case is several hospitals
    training a shared model without any of them sending patient records anywhere.

    The distinction that gets tested: **DP adds noise; federated learning moves the computation.**
    They address different threats and are frequently used together.

---

## 7. Deploying with privacy in mind

- **Self-hosting** (NIM, Triton, on-prem NeMo) keeps prompts and documents inside your perimeter.
  For regulated industries this is often the deciding architectural factor, ahead of cost or
  quality. See [Deployment](../domain-2/deployment.md).
- **Hosted APIs** mean data leaves your perimeter. Check retention periods and whether inputs are
  used for training.
- **Never put secrets, keys or credentials in prompts.** They are logged, cached, potentially
  leaked back out via prompt extraction, and often transmitted to a third party.
- **Tell users** when their input may be used for training, and give them a real choice.

---

## 8. Recap

- The **balance** is: minimum necessary data + informed, specific, revocable consent + purpose
  limitation + retention limits + technical safeguards.
- Privacy and **fairness can conflict** — you need demographic data to measure fairness.
- Consent must be **informed, specific, freely given, revocable and documented**. **Secondary use
  requires a new basis** — "we already have the data" is not one.
- PII is direct **or indirect in combination** (postcode + DOB + gender is identifying).
- Handle PII at training, indexing, inference **and in logs**. Enforce access control at
  **retrieval**, never by instructing the model.
- The **right to erasure** cannot be honoured in trained weights — a strong argument for **RAG
  over fine-tuning** with personal data.
- **Deduplication** is the primary defence against training-data memorisation.
- **Differential privacy adds noise; federated learning moves the computation.** Know both.
