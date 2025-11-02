import json
from tqdm import tqdm
from .hf_loaders import load_hf

# Simple heuristic “deterministic” evaluator
def eval_nonref(dataset, split, out_path):
    results = []
    for ex in tqdm(load_hf(dataset, split)):
        transcript = ex["transcript"].lower()
        note = ex.get("gen_note", ex.get("ref_note", "")).lower()

        missing = 0
        for keyword in ["pain", "fever", "rash", "swelling", "nausea"]:
            if keyword in transcript and keyword not in note:
                missing += 1

        hallucinated = 0
        for keyword in ["cancer", "fracture", "heart failure"]:
            if keyword not in transcript and keyword in note:
                hallucinated += 1

        results.append({
            "id": ex["id"],
            "missing_findings": missing,
            "hallucinations": hallucinated,
            "quality": max(0, 1 - 0.2 * (missing + hallucinated))
        })

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    eval_nonref("omi-health/medical-dialogue-to-soap-summary", "test", "results_omi_det.jsonl")
