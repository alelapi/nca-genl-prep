# Privacy, Consent & Data Governance

*Covers task 5.2: "describe the balance between data privacy and the importance of data
consent." Note the blueprint's word — **balance**. Questions here are about trade-offs, not
absolutes.*

## The tension

AI systems improve with more data; individuals have a right to control data about
themselves. Both are legitimate, and they pull against each other:

| Pulls toward more data | Pulls toward less |
| --- | --- |
| Better accuracy and coverage | Individual privacy rights |
| Fewer blind spots for minority groups | Regulatory obligations (GDPR, CCPA, HIPAA) |
| Personalised, more useful systems | Breach and misuse risk |
| Faster iteration | Loss of user trust once violated |

!!! tip "The exam's framing"
    The trustworthy answer is almost never "collect everything" or "collect nothing". It is
    **collect the minimum necessary, with informed consent, for a stated purpose, retained
    for a limited time, with technical safeguards.** Options describing that shape are
    usually correct.

## Consent

**Informed consent** means the person understands and freely agrees to a specific use. It
must be:

- **Informed** — plain language, not buried in a 40-page agreement.
- **Specific** — tied to a stated purpose. Consent for order fulfilment is not consent for
  model training.
- **Freely given** — not coerced by making the service unusable otherwise.
- **Revocable** — withdrawal must be possible, and must actually take effect.
- **Documented** — you must be able to show what was consented to, and when.

**Opt-in vs. opt-out** — opt-in (explicit affirmative action) is the privacy-preserving
default and what regulation increasingly requires. Opt-out assumes consent from silence.

**Secondary use** is the recurring problem: data collected for one purpose later used to
train a model. That is a **new purpose** and generally requires a new legal basis. "We
already have the data" is not a justification.

!!! danger "Consent and LLM training data"
    Web-scraped corpora contain personal data, copyrighted work and content whose authors
    never consented to model training. This is an unresolved legal and ethical issue and is
    exactly why NVIDIA lists privacy and consent as a certification topic. The practical
    professional stance: know the provenance of your training data, respect robots.txt and
    licences, honour deletion requests, and document your basis.

## Personal data and PII

**PII** — data that identifies a person directly (name, email, phone, national ID,
biometrics) or indirectly in combination (postcode + birth date + gender is famously
re-identifying).

**Special categories** requiring stronger protection: health, biometric and genetic data,
racial or ethnic origin, political opinions, religion, trade union membership, sexual
orientation.

**Handling PII in an LLM pipeline:**

- **Detect and redact** before training or indexing — regex plus NER (spaCy, Presidio),
  and NVIDIA's **NeMo Curator** includes PII removal.
- **Filter prompts and outputs** at inference — a guardrail responsibility.
- **Redact logs** — traces of prompts and responses are a major, frequently overlooked
  leak.
- **Encrypt** in transit (TLS) and at rest.
- **Access control** — RAG retrieval must respect per-user document permissions, enforced
  with metadata filters at query time, not by hoping the model behaves.
- **Retention limits** — define and enforce them.

## Regulatory concepts worth recognising

You do not need legal expertise, but the vocabulary appears in answer options:

| Concept | Meaning |
| --- | --- |
| **Data minimisation** | Collect only what is necessary for the stated purpose |
| **Purpose limitation** | Use it only for that purpose |
| **Storage limitation** | Keep it only as long as needed |
| **Right of access** | People can see what you hold about them |
| **Right to erasure ("right to be forgotten")** | People can require deletion |
| **Right to rectification** | Correction of inaccurate data |
| **Data portability** | Export in a usable format |
| **Right not to be subject to solely automated decisions** | Human review of significant automated decisions |
| **Privacy by design / by default** | Build protection in from the start; the default setting is the private one |
| **DPIA** | Data Protection Impact Assessment for high-risk processing |
| **Data residency / sovereignty** | Data must remain in a jurisdiction |

Frameworks by name: **GDPR** (EU), **CCPA/CPRA** (California), **HIPAA** (US health),
**EU AI Act** (risk-tiered AI regulation), **NIST AI Risk Management Framework**,
**ISO/IEC 42001** (AI management systems).

!!! warning "The right to erasure vs. trained weights"
    Deleting a person's record from your database is straightforward. Removing their
    influence from a **trained model's weights** is not — retraining is the only reliable
    remedy, and "machine unlearning" remains an open research area.

    This is a strong practical argument for **RAG over fine-tuning** when personal data is
    involved: delete the document from the index and the system stops using it immediately.

## Memorisation and extraction

LLMs can **memorise and regurgitate** training data verbatim — especially content that was
duplicated many times or is unusual enough to be distinctive. Demonstrated attacks have
recovered names, addresses, phone numbers and API keys from public models.

Mitigations: **deduplicate** the training corpus (duplication is the main driver of
memorisation), **remove PII before training**, apply **differential privacy** during
training, filter outputs, and test for extraction with red-teaming.

## Privacy-preserving techniques

| Technique | Idea | Caveat |
| --- | --- | --- |
| **Anonymisation** | Irreversibly remove identifiers | Genuinely hard; re-identification via combination is common |
| **Pseudonymisation** | Replace identifiers with tokens | Still personal data under GDPR — reversible with the key |
| **Aggregation** | Report only group-level statistics | Small groups can still be re-identified |
| **Differential privacy** | Add calibrated noise so no individual record measurably changes the output | Formal guarantee; costs accuracy (the privacy budget ε) |
| **Federated learning** | Train across devices/sites; **only model updates leave**, never raw data | Updates can still leak; combine with DP and secure aggregation |
| **Secure enclaves / confidential computing** | Compute on encrypted data in hardware-protected memory | Performance and platform constraints |
| **Homomorphic encryption** | Compute directly on encrypted data | Still impractical for large models |
| **Synthetic data** | Train on generated data resembling the real distribution | Can leak the original if the generator memorised it |

!!! tip "Two you should be able to define on sight"
    **Differential privacy** — a mathematical guarantee that the presence or absence of any
    single individual's data does not meaningfully change the result.
    **Federated learning** — the data never leaves the device or institution; only model
    updates are shared and aggregated.

## Deploying with privacy in mind

- **Self-hosting** (NIM, Triton, on-prem NeMo) keeps prompts and documents inside your
  perimeter — often the deciding factor for regulated industries.
- **Hosted APIs** mean data leaves it; check retention and training-use terms.
- **Never put secrets in prompts** — they are logged, cached and can be leaked back out.
- **Tell users** when their input may be used for training, and give them a real choice.

## Key takeaways

- **Balance** = minimum necessary data + informed, specific, revocable consent + purpose
  limitation + retention limits + technical safeguards.
- Consent must be informed, specific, freely given, revocable and documented; **secondary
  use needs a new basis**.
- Detect and redact PII at training, indexing, inference **and in logs**; enforce
  permissions at retrieval time.
- The **right to erasure** is nearly impossible to honour in trained weights — a strong
  argument for RAG over fine-tuning with personal data.
- **Deduplication** is the main defence against training-data memorisation.
- Know **differential privacy** and **federated learning** by definition.
