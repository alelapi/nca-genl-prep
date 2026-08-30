# Lab 6 — LoRA Fine-Tuning

**Time:** 60 minutes · **Covers:** Domain 2 (customization, PEFT)

Small enough to run on CPU (slowly) or any GPU. The goal is not a great model — it is to
**see the parameter counts** and understand why PEFT changed everything.

## 1. Baseline: count the parameters

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "distilgpt2"            # 82M params — swap for a bigger one if you have a GPU

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL)

total = sum(p.numel() for p in model.parameters())
print(f"total parameters: {total:,}")
print(f"FP32 weights    : {total * 4 / 1e6:.1f} MB")
print(f"FP16 weights    : {total * 2 / 1e6:.1f} MB")
print(f"full FT states  : {total * 16 / 1e6:.1f} MB  (params+grads+fp32 master+Adam)")
```

That last line is the number from
[Hardware Sizing](../domain-2/hardware-sizing.md): **~16 bytes per parameter** for full
fine-tuning. Scale it mentally to 7B and you get ~112 GB — which is why LoRA exists.

## 2. Apply LoRA

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                     # rank — the key knob
    lora_alpha=16,           # scaling (alpha/r applied to BA)
    lora_dropout=0.05,
    target_modules=["c_attn"],   # attention projections in GPT-2 style models
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
```

Expected output shape:

```text
trainable params: 294,912 || all params: 82,207,104 || trainable%: 0.3587
```

**0.36% of the parameters.** Vary the rank and watch it move:

```python
from transformers import AutoModelForCausalLM

for r in [2, 4, 8, 16, 32, 64]:
    base = AutoModelForCausalLM.from_pretrained(MODEL)
    m = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=r,
                                        lora_alpha=2*r, target_modules=["c_attn"]))
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"r={r:<3} trainable={trainable:>9,}  ({trainable/total*100:.3f}%)")
```

Trainable parameters scale **linearly with rank** — this is the memory/expressivity dial.

## 3. A tiny training set

```python
from datasets import Dataset

EXAMPLES = [
    {"text": "Q: What does Triton do?\nA: [OPS] Triton serves models in production with dynamic batching and versioning."},
    {"text": "Q: What is TensorRT?\nA: [OPS] TensorRT compiles and optimizes trained networks for fast GPU inference."},
    {"text": "Q: What is LoRA?\nA: [OPS] LoRA freezes base weights and trains small low-rank adapter matrices."},
    {"text": "Q: What is cuDF?\nA: [OPS] cuDF is the RAPIDS GPU DataFrame library replacing pandas."},
    {"text": "Q: What is NCCL?\nA: [OPS] NCCL implements optimized multi-GPU collective communication."},
    {"text": "Q: What is quantization?\nA: [OPS] Quantization lowers numeric precision to cut memory and speed up inference."},
] * 40      # repeat so a tiny run has something to learn

ds = Dataset.from_list(EXAMPLES)

def tokenize(batch):
    out = tok(batch["text"], truncation=True, padding="max_length", max_length=64)
    out["labels"] = out["input_ids"].copy()
    return out

ds = ds.map(tokenize, batched=True, remove_columns=["text"])
```

The `[OPS]` tag is a deliberately artificial style marker — it makes the behavioural change
visible in a short run.

## 4. Train

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

args = TrainingArguments(
    output_dir="./lora-out",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,          # note: much higher than full FT (1e-5..5e-5)
    logging_steps=20,
    save_strategy="no",
    report_to=[],
)

trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)
trainer.train()
```

!!! note "Why the learning rate is higher"
    LoRA trains freshly initialised low-rank matrices, not delicately pretrained weights.
    Typical LoRA learning rates are **1e-4 to 3e-4** — roughly 10× a full fine-tune.

## 5. Compare before and after

```python
def generate(m, prompt, n=40):
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)

prompt = "Q: What is NeMo Guardrails?\nA:"

base = AutoModelForCausalLM.from_pretrained(MODEL).eval()
print("--- BASE ---\n", generate(base, prompt))
print("\n--- LoRA-TUNED ---\n", generate(peft_model.eval(), prompt))
```

The tuned model should adopt the `[OPS]` prefix and the terse answer format — even for a
question it never saw. **That is style/behaviour transfer, which is what fine-tuning is
for.** It has not learned any new facts about NeMo Guardrails, which is what
[RAG](04-rag.md) is for.

## 6. Adapter size

```python
import os

peft_model.save_pretrained("./lora-adapter")
adapter_bytes = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, fs in os.walk("./lora-adapter") for f in fs
)
print(f"adapter on disk : {adapter_bytes/1e6:.2f} MB")
print(f"full model FP32 : {total*4/1e6:.1f} MB")
print(f"ratio           : {total*4/adapter_bytes:.0f}x smaller")
```

This is why you can serve **dozens of task-specific adapters against one loaded base
model** and swap them per request.

## 7. Merge for zero-latency inference

```python
merged = peft_model.merge_and_unload()      # folds BA into W
print(type(merged))
print("trainable after merge:",
      sum(p.numel() for p in merged.parameters() if p.requires_grad))
print(generate(merged.eval(), prompt))
```

After merging, the model is architecturally identical to the base — **no extra layers, no
added inference latency**. That is LoRA's structural advantage over adapter layers.

## Break it on purpose

1. Set `r=1`. Does the style still transfer? Where does it break?
2. Raise `num_train_epochs` to 20 and ask an unrelated question ("Q: Write a poem about
   the sea.\nA:"). Watch **catastrophic forgetting** appear.
3. Set `learning_rate=1e-6`. Nothing changes — the classic "why isn't it learning" bug.
4. Add `target_modules=["c_attn", "c_proj", "c_fc"]` and compare trainable parameter counts.

## Takeaways

- Full fine-tuning costs **~16 bytes/parameter** in state; LoRA trains **0.1–1%** of them.
- Trainable parameters scale **linearly with rank `r`**.
- LoRA learning rates are ~10× higher than full fine-tuning.
- Adapters are **megabytes** — many adapters, one base model.
- `merge_and_unload()` folds `BA` into `W`: **zero added inference latency**.
- Fine-tuning changed **behaviour and style**, not knowledge. Knowledge is RAG's job.
