# Lab 2 — Transformers with Hugging Face

**Time:** 45 minutes · **Covers:** Domain 1 (transformers, tokenization, decoding)

## 1. Tokenization

```python
from transformers import AutoTokenizer

bert = AutoTokenizer.from_pretrained("bert-base-uncased")      # WordPiece
gpt2 = AutoTokenizer.from_pretrained("gpt2")                   # BPE

text = "Tokenization handles unbelievable out-of-vocabulary words like antidisestablishmentarianism."

for name, tok in [("BERT/WordPiece", bert), ("GPT-2/BPE", gpt2)]:
    ids = tok.encode(text)
    print(f"\n{name}: {len(ids)} tokens")
    print(tok.convert_ids_to_tokens(ids)[:25])
```

**Observe:**

- Long rare words are split into **subwords** — this is how a fixed vocabulary handles any
  input, and why there is no true out-of-vocabulary problem.
- BERT adds `[CLS]` and `[SEP]`; GPT-2 does not.
- BERT's `##` prefix marks continuation; GPT-2's `Ġ` marks a preceding space.

```python
# The token/word ratio that governs cost estimates
words = len(text.split())
print(f"\nwords={words}, gpt2 tokens={len(gpt2.encode(text))}, "
      f"ratio={len(gpt2.encode(text))/words:.2f}")
```

## 2. Pipelines: one line per task

```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")
print(sentiment("The latency improvements are outstanding, but the cost is brutal."))

ner = pipeline("ner", aggregation_strategy="simple")
print(ner("Jensen Huang founded NVIDIA in 1993 in Santa Clara."))

zeroshot = pipeline("zero-shot-classification")
print(zeroshot(
    "The GPU ran out of memory during fine-tuning.",
    candidate_labels=["hardware", "billing", "user interface", "documentation"],
))
```

!!! note "How zero-shot classification works"
    No labeled training data was used. The model frames each candidate label as an
    **entailment hypothesis** against your text as the premise. This is the mechanism
    behind NVIDIA's course objective *"use encoder models for zero-shot classification"*.

## 3. Encoder vs. decoder, concretely

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch

# ENCODER — produces representations, sees the whole sequence bidirectionally
enc_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = AutoModel.from_pretrained("bert-base-uncased")
out = enc(**enc_tok("The bank raised interest rates.", return_tensors="pt"))
print("encoder hidden states:", out.last_hidden_state.shape)   # (1, seq, 768)

# DECODER — generates text, causal (masked) attention
dec_tok = AutoTokenizer.from_pretrained("gpt2")
dec = AutoModelForCausalLM.from_pretrained("gpt2")
ids = dec_tok("The future of GPU computing is", return_tensors="pt")
gen = dec.generate(**ids, max_new_tokens=25, do_sample=False)
print("\ndecoder output:", dec_tok.decode(gen[0], skip_special_tokens=True))
```

Now demonstrate that BERT embeddings are **contextual**:

```python
import torch.nn.functional as F

def token_vector(sentence, word):
    enc_in = enc_tok(sentence, return_tensors="pt")
    tokens = enc_tok.convert_ids_to_tokens(enc_in["input_ids"][0])
    idx = tokens.index(word)
    with torch.no_grad():
        h = enc(**enc_in).last_hidden_state[0, idx]
    return h

river = token_vector("I sat on the river bank and watched the water.", "bank")
money = token_vector("I deposited the cheque at the bank downtown.", "bank")
money2 = token_vector("The bank approved my mortgage application.", "bank")

print("river-bank vs money-bank :", F.cosine_similarity(river, money, dim=0).item())
print("money-bank vs money-bank2:", F.cosine_similarity(money, money2, dim=0).item())
```

The same word, different vectors. **Word2Vec cannot do this** — it has one vector per word.

## 4. Decoding parameters

```python
prompt = "The three main benefits of retrieval-augmented generation are"
ids = dec_tok(prompt, return_tensors="pt")

configs = {
    "greedy (temp=0 equivalent)": dict(do_sample=False),
    "temperature 0.3":            dict(do_sample=True, temperature=0.3, top_p=1.0),
    "temperature 1.5":            dict(do_sample=True, temperature=1.5, top_p=1.0),
    "top_p 0.9":                  dict(do_sample=True, temperature=1.0, top_p=0.9),
    "top_k 5":                    dict(do_sample=True, temperature=1.0, top_k=5),
}

torch.manual_seed(0)
for label, cfg in configs.items():
    out = dec.generate(**ids, max_new_tokens=30, pad_token_id=dec_tok.eos_token_id, **cfg)
    print(f"\n--- {label} ---\n{dec_tok.decode(out[0], skip_special_tokens=True)}")
```

Run greedy twice — identical output. Run temperature 1.5 twice — different every time.
**That is the determinism trade-off in one experiment.**

## 5. Looking at attention

```python
enc_attn = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager",
                                     output_attentions=True)
sent = "The cat sat on the mat because it was warm."
inp = enc_tok(sent, return_tensors="pt")
with torch.no_grad():
    res = enc_attn(**inp)

attn = res.attentions            # tuple: one tensor per layer
print("layers:", len(attn), "| shape per layer:", attn[0].shape)  # (batch, heads, seq, seq)

tokens = enc_tok.convert_ids_to_tokens(inp["input_ids"][0])
layer, head = 5, 3
it = tokens.index("it")
weights = attn[layer][0, head, it]
top = torch.topk(weights, 5)
print(f"\n'it' (layer {layer}, head {head}) attends most to:")
for score, i in zip(top.values, top.indices):
    print(f"  {tokens[i]:<12}{score.item():.3f}")
```

Different heads specialise. Some track syntax, some coreference, some just attend to
`[SEP]`. **Attention is suggestive, not a causal explanation** — a Trustworthy AI point.

## Break it on purpose

1. Feed BERT a 600-token document. Watch it truncate at 512 **silently**. This is exactly
   how chunking bugs hide.
2. Set `temperature=2.5` and generate. Read the output. That is what "flatten the
   distribution" means.
3. Use `bert-base-uncased`'s tokenizer with a GPT-2 model. Observe the garbage — the same
   category of error as mismatched embedding models in RAG.

## Takeaways

- Subword tokenization (BPE/WordPiece) gives a fixed vocabulary that can spell anything.
- **Encoder = representations** (bidirectional), **decoder = generation** (causal).
- BERT embeddings are **contextual**; the same word gets different vectors.
- Greedy decoding is reproducible; sampling is not. Temperature controls the trade-off.
- Attention heads specialise, but attention is not an explanation.
