# DeepScribe Evaluation Suite

### Evaluating the Quality of Automatically Generated SOAP Notes

This repository implements a **scalable, statistically sound evaluation framework** for clinical note generation systems.  
It identifies **missing findings**, **hallucinated facts**, and **clinical accuracy issues** using both **reference-based** and **non-reference-based** methods.

---

## Context

given a dataset of SOAP notes — each with:

- **Transcript:** the raw doctor–patient conversation  
- **Reference note:** the clinician-edited ground-truth  
- **Generated note:** produced by an AI model (e.g., Gemma-2B or Mistral-7B)

task is to **measure note quality** using a combination of **deterministic metrics** and **LLM-based judgment** — fast enough to integrate into production model evaluations.

---

## Framework Overview

### Core Goals
| Goal | Description |
|------|--------------|
| **1. Move fast** | Evaluate models quickly across 100–1000 notes without waiting for human review. |
| **2. Understand production quality** | Quantify how notes perform “in the wild,” identifying factual gaps or hallucinations that impact safety. |

---

## System Design

### Architecture Overview

```
dataset → generate_notes_gpu_opt.py 
        → [generated_notes.jsonl]
        ↓
+-------------------+
| Reference-based   | → ROUGE, BLEU  → score_vs_reference.py
+-------------------+
| Non-reference LLM | → Missing / Hallucination / Conflict  → run_eval_hf.py
+-------------------+
        ↓
summarize_ref.py, summarize_nonref.py → summary_overall.json
```

### Evaluation Dimensions

| Dimension | Description | Type |
|------------|--------------|------|
| **Missing findings** | Facts from transcript omitted in the note. | LLM-judge |
| **Hallucinations** | Statements not grounded in transcript. | LLM-judge |
| **Clinical conflicts** | Contradictory or misleading content. | LLM-judge |
| **ROUGE / BLEU** | Overlap with clinician reference. | Deterministic |
| **Quality Score** | Subjective 0–1 LLM rating (overall coherence & completeness). | LLM-judge |

---

## Final Results (Gemma-2B-IT, 250 Samples)

| Metric | Value | Interpretation |
|--------|--------|----------------|
| **ROUGE-L** | **0.199** | ~20% long-sequence overlap with clinician note — moderate semantic match. |
| **BLEU** | **8.28** | Low lexical overlap, expected for abstractive summarization. |
| **Quality (LLM)** | **0.828** | Majority of notes are good or acceptable. |
| **Missing Findings** | **0.86** | 86% of notes omit at least one key clinical detail — main improvement area. |
| **Hallucinations** | **0.00** | Excellent factual precision — no unsupported facts detected. |
| **95% CI** | [0.40, 1.00] | Reflects moderate variability in quality across notes. |

---

## How This Meets DeepScribe’s Goals

###  Goal 1: Move Fast
- ✅ **GPU-accelerated + 8-bit quantization** for small models (Gemma-2B, Mistral-7B).  
- ✅ **End-to-end in hours**, not days.  
- ✅ **Modular pipeline** (`src/`) makes swapping models trivial.  
- ✅ **Resumable JSONL output** enables streaming and parallel processing.

###  Goal 2: Understand Production Quality
- ✅ **Reference-based metrics** (ROUGE/BLEU) track regression vs. clinician notes.  
- ✅ **LLM-as-judge** evaluates quality when no ground truth exists.  
- ✅ **Statistical summaries** (mean + CI) provide reliability bounds.  
- ✅ **JSON-based summaries** integrate easily with dashboards and alerts.

---

##  Combined Summary Output

```json
{
  "rougeL_mean": 0.1991,
  "bleu_mean": 8.2777,
  "n": 250,
  "quality_mean": 0.828,
  "missing_mean": 0.86,
  "halluc_mean": 0.0,
  "quality_ci": [0.40, 1.00]
}
```

---

## Eval-of-Eval: Measuring the Evaluator’s Quality

| Test | Setup | Expected Outcome | Result |
|------|--------|------------------|--------|
| **Shuffle test** | Randomly mismatched transcripts ↔ notes. | Quality ↓, Missing ↑ | ✅ Discriminates correctly |
| **Injection test** | Manually deleted facts in sample notes. | Missing ↑ to 1.0 | ✅ Detected all missing |
| **Hallucination test** | Added unsupported statement. | Hallucination ↑ | ✅ Detected all |
| **Paraphrase test** | Reworded note with same facts. | Scores stable | ✅ Few false positives |
| **Judge stability** | Compared Mistral vs Gemma judges. | Same trends | ✅ High agreement |

These confirm the evaluation suite reacts to *real factual changes*, not style or wording — satisfying the “How do you know your eval works?” requirement.

---

## STEPS

### 1️⃣ Create Environment
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### 2️⃣ Generate SOAP Notes
```bash
python -m src.generate_notes_gpu_opt --model google/gemma-2b-it --split test --n 250
```

### 3️⃣ Run Evaluations
**(Reference-based)**  
```bash
python -m src.score_vs_reference --gen generated_notes_gpu.jsonl --out ref_eval.jsonl
python -m src.summarize_ref --inp ref_eval.jsonl --out summary_ref.json
```

**(Non-reference, LLM-as-judge)**  
```bash
python -m src.run_eval_hf --gen generated_notes_gpu.jsonl --out results_omi_det.jsonl
python -m src.summarize_nonref --inp results_omi_det.jsonl --out summary_omi.json
```

### 4️⃣ Combine Summaries
```bash
python -m src.combined_summary
# → summary_overall.json
```

### 5️⃣ (Optional) Run Eval-of-Eval
```bash
python -m src.sanity_check --n 30 --report sanity_report.json
```

---

### Artifacts Produced
| File | Description |
|------|--------------|
| `generated_notes_gpu.jsonl` | Model-generated SOAP notes |
| `results_omi_det.jsonl` | Non-reference evaluation results |
| `ref_eval.jsonl` | Reference-based results |
| `summary_omi.json` | Non-ref summary |
| `summary_ref.json` | Reference summary |
| `summary_overall.json` | Combined report |
| `sanity_report.json` | Eval-of-eval diagnostics |

---
