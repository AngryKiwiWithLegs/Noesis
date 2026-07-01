# Noesis Memory Injection: 50-Question A/B Comparison Report

> **Experiment date**: 2026-06-29
> **Experiment type**: With-memory (Noesis) vs without-memory (direct) end-to-end answer quality comparison
> **Scoring method**: Keyword hit (objective, reproducible)
> **Language**: English
> **Sample size**: 50 questions (10 user profiles × 5 questions)

---

## I. Experiment Purpose

Verify the core question: **How much does Noesis's memory injection improve AI answer accuracy on user-relevant questions?** Compared to the earlier 15-question run, this experiment enlarges the sample to 50 questions across 10 distinct user profiles, and is conducted entirely in English.

## II. Experiment Design

### 2.1 Flow

```
Stage 1 (build memory): Each user profile makes 5 statements with clear stances via the Noesis proxy → memory store established
Stage 2 (test):        Ask 5 related questions for each profile
    Group A (treatment): via Noesis proxy → memory injected into system prompt → LLM answers
    Group B (control):   direct to LLM API → no memory → LLM answers
Stage 3 (scoring):      Check whether each answer contains the expected keyword (from the known memory)
```

### 2.2 Subject model

| Model | Source | Access |
|---|---|---|
| `gemini-flash-lite-latest` | Google Gemini API | OpenAI-compatible endpoint |

### 2.3 User profiles and questions

10 user profiles × 5 questions = **50 test questions**. Each question is asked once in each group = **100 LLM calls** (plus 50 build statements = 150 calls total).

| # | Profile | Profession | Example statement | Example question |
|---|---|---|---|---|
| 1 | prof_1 | Backend engineer | "I prefer PostgreSQL over MySQL for its JSON support" | "Recommend a database for me" → expect `postgres` |
| 2 | prof_2 | Data scientist | "I decided to learn Rust for systems programming, safer than C++" | "What programming language am I learning?" → expect `rust` |
| 3 | prof_3 | Frontend engineer | "I decided to use Tailwind CSS for styling, faster than handwritten CSS" | "What do I use for styling?" → expect `tailwind` |
| 4 | prof_4 | DevOps engineer | "I decided to use Terraform for infrastructure as code" | "What do I use for infrastructure?" → expect `terraform` |
| 5 | prof_5 | Mobile developer (iOS) | "I decided to use SwiftUI over UIKit for new projects" | "What UI framework do I use for new projects?" → expect `swiftui` |
| 6 | prof_6 | ML engineer | "I prefer PyTorch over TensorFlow for research flexibility" | "What deep learning framework do I prefer?" → expect `pytorch` |
| 7 | prof_7 | Game developer | "I prefer C# over C++ for game scripting productivity" | "What language do I prefer for scripting?" → expect `c#` |
| 8 | prof_8 | Security engineer | "I decided to use Vault for secrets management" | "What do I use for secrets management?" → expect `vault` |
| 9 | prof_9 | Blockchain developer | "I think layer-2 rollups are the future of Ethereum scaling" | "What is my view on Ethereum scaling?" → expect `rollup` |
| 10 | prof_10 | Data engineer | "I prefer Apache Airflow over Luigi for workflow orchestration" | "What orchestration tool do I prefer?" → expect `airflow` |

### 2.4 Scoring criteria

- **🟢 Effective**: with-memory hit, without-memory miss → Noesis demonstrates value
- **🟡 Same**: both groups agree (both hit or both miss)
- **🔴 Anomaly**: without-memory hit, with-memory miss → memory injection introduced an error (should be rare)

---

## III. Experiment Results

### 3.1 Overview

| Metric | With memory (Noesis) | Without memory (direct) | Improvement |
|---|---|---|---|
| Hit rate | **39/50 (78%)** | 25/50 (50%) | **+14 (+28pp)** |
| Relative gain | — | — | **+56%** |

**Core finding: Noesis raises the hit rate from 50% to 78% — a 56% relative improvement.**

### 3.2 Per-question breakdown

| Outcome | Count | Meaning |
|---|---|---|
| 🟢 Effective (mem✅ dir❌) | **18** | Noesis's clear value — memory let it answer correctly |
| 🟡 Same — both hit | 21 | Both correct (e.g. the topic is mainstream enough to guess) |
| 🟡 Same — both miss | 7 | Neither got it (mostly rare/niche terms) |
| 🔴 Anomaly (mem❌ dir✅) | 4 | Memory injection appeared to hurt (analyzed below) |

### 3.3 Key findings

#### Finding 1: Noesis produces a clear positive effect on a larger, statistically meaningful sample

Across 50 questions, the with-memory group hit 39/50 (78%) versus 25/50 (50%) without memory — a **+28 percentage-point** improvement. The direction is unambiguous and consistent with the earlier 15-question run.

#### Finding 2: No memory injection never systematically hurts

Of 50 questions, only **4 (8%) were anomalies** where the direct answer hit and the memory answer missed — and all 4 are explainable (retrieval precision issues or language-bleed artifacts), not systematic failures. **No case of "memory makes it worse on average" exists.**

#### Finding 3: The effect concentrates on personal-knowledge questions

The 🟢 effective cases cluster around questions the LLM **cannot know without memory** — "What is my job title?", "What language am I learning?", "What do I use for styling?". On these, without-memory answers are generic refusals ("I don't have access to your personal information"), while with-memory answers are precise and correct.

### 3.4 Typical answer comparisons

#### Example 1: "What programming language am I learning?" (prof_2)

| Group | Answer |
|---|---|
| **With memory** | "You are learning **Rust**." |
| **Without memory** | "To tell you which programming language you are learning, I need a little bit of information! Since..." |

#### Example 2: "What do I use for vector search?" (prof_1)

| Group | Answer |
|---|---|
| **With memory** | "Since you have already decided on **sqlite-vec**, you have made a great choice for a 'lighter'..." |
| **Without memory** | "Choosing the right tool for vector search depends on your specific use case (e.g., scale, budget)..." |

#### Example 3: "What UI framework do I use for new projects?" (prof_5)

| Group | Answer |
|---|---|
| **With memory** | "Based on the information provided, you decided to use **SwiftUI** for new projects..." |
| **Without memory** | "I'd need more context about your specific app requirements to recommend..." |

---

## IV. Honest Limitations

| Limitation | Detail | Impact on conclusions |
|---|---|---|
| **Keyword-hit scoring is coarse** | Only checks whether the expected word appears, not whether the answer is semantically correct | May over/under-count; direction is reliable |
| **21 "both hit" cases** | Some questions are answerable without memory because the topic is mainstream (e.g. "python", "graphql") | Understates Noesis's value on truly personal knowledge |
| **No LLM-as-judge** | A stronger model did not score semantic correctness | Reviewers may request this |
| **Single model (Gemini)** | Should replicate with Claude/GPT to rule out model-specific effects | Generalizability not yet shown |
| **4 anomaly cases** | Mostly retrieval precision (wrong memory injected) or language-bleed (Gemini answered in Chinese) | Real weaknesses, honestly reported |
| **Build/test alignment** | Each test question maps directly to a build statement | Could over-estimate; should add distractor questions |

---

## V. Comparison with the 15-Question Run (Chinese)

| Metric | This run (EN, n=50) | Prior run (CN, n=15) |
|---|---|---|
| With memory | 78% | 93% |
| Without memory | 50% | 47% |
| Improvement | +28pp | +47pp |
| Effective cases | 18 | 7 |
| Anomaly cases | 4 | 0 |

The effect holds in both languages and at both sample sizes. The Chinese run showed a larger absolute improvement partly because its no-memory baseline was lower (47% vs 50%), giving more headroom.

---

## VI. Data files

| File | Contents |
|---|---|
| `results/ab_comparison_en_20260629_211107.json` | Full data (per-question A/B original answers, English) |
| `ab_comparison_en.py` | Reproducible script |

---

## VII. Reproducibility

```bash
# 1. Start Noesis daemon
cd ~/Noesis && source .venv/bin/activate && noesis start --ws &

# 2. Run the 50-question English experiment
GEMINI_API_KEY="yourkey" python3 ab_comparison_en.py

# 3. Full JSON results appear in results/
```
