# Exam-Day Strategy

## Before you start

**The week before**

- [ ] Confirm your Certiverse booking and time zone
- [ ] Run the system/proctoring check from the same machine and network you will use
- [ ] Have government-issued photo ID ready

**The day of**

- [ ] Clear your desk completely — no paper, no phone, no second monitor
- [ ] Close every application; disable notifications
- [ ] Use a private, quiet, well-lit room; nobody else may enter
- [ ] Plug in the laptop; use wired ethernet if you have it
- [ ] Be seated 15 minutes early — the proctor check-in eats real time

!!! warning "Remote proctoring is strict"
    Reading questions out loud, looking away from the screen for long stretches, or
    another person entering the room can void the session. Keep your face in frame and
    your eyes on the screen.

## Time budget

**60 minutes, 50–60 questions → ~60 seconds per question.**

| Phase | Time | What you do |
| --- | --- | --- |
| Pass 1 | ~40 min | Answer everything you know in under 45 s. Flag anything slower and move on. |
| Pass 2 | ~15 min | Work the flagged questions with the full time they need. |
| Pass 3 | ~5 min | Verify no question is blank. Never leave one unanswered — there is no penalty for a wrong guess. |

The single biggest cause of failure at Associate level is **spending four minutes on
one hard question** and then rushing twelve easy ones.

## How to attack the question types

**Definition / recognition** — *"What is the primary purpose of X?"*
You either know it or you do not. Answer in 20 seconds and bank the time.

**Scenario / best-choice** — *"A team needs Y. Which approach is most appropriate?"*
Find the **constraint** in the stem first: no labeled data, must not retrain, model
does not fit in GPU memory, needs current information, latency budget. The constraint
usually eliminates two options immediately.

**Trade-off** — *"Which technique reduces memory at the cost of accuracy?"*
Map to the axis being tested: quality ↔ cost, latency ↔ throughput, memory ↔ precision,
generality ↔ specialisation.

**"Which is NOT…" / "least likely"** — Slow down and re-read. Reversed-polarity
questions are where careless points are lost. Mentally mark the three true statements,
and the leftover is your answer.

## Elimination heuristics

- **Absolutes are usually wrong.** "always", "never", "guarantees", "completely
  eliminates". LLM answers are probabilistic; NVIDIA writes correct answers with hedged
  language: "reduces", "typically", "helps mitigate".
- **The vendor-neutral concept usually beats the product name** — unless the stem
  explicitly asks for an NVIDIA technology, in which case pick the correct NVIDIA one
  (Triton = serving, TensorRT/TensorRT-LLM = inference optimization, NeMo = build and
  customize, NeMo Guardrails = safety rails, RAPIDS = GPU dataframes/ML, NIM = packaged
  inference microservice).
- **When two options are near-synonyms, neither is usually right** — the answer is one
  of the other two.
- **Match the scale of the fix to the scale of the problem.** If the stem says "quick
  proof of concept with no training budget", the answer is prompting or RAG, not
  pretraining from scratch.

## The five reflexes that earn the most points

1. **Missing/outdated knowledge → RAG.** Not fine-tuning.
2. **Missing style, format, tone or domain behaviour → fine-tuning (PEFT/LoRA first).**
3. **Model does not fit in GPU memory → quantization, then tensor/pipeline parallelism.**
4. **Need throughput on a server → batching (in-flight/continuous), Triton, TensorRT-LLM.**
5. **Need to compare two variants in production → A/B test with a defined metric and
   sufficient sample size.**

## After the exam

Results normally appear immediately. If you pass, your digital badge arrives by email;
the certification is valid for **two years**. If you do not, note which domains were
weak — NVIDIA reports per-domain performance — and rework those chapters before
rebooking.
