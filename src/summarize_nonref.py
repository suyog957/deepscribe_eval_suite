# src/summarize_nonref.py
import json, numpy as np

def summarize(path, out):
    data = [json.loads(l) for l in open(path)]
    quality = [d.get("quality", 0) for d in data]
    missing = [d.get("missing_findings", 0) for d in data]
    halluc = [d.get("hallucinations", 0) for d in data]

    summary = {
    "n": int(len(data)),
    "quality_mean": float(np.mean(quality)),
    "missing_mean": float(np.mean(missing)),
    "halluc_mean": float(np.mean(halluc)),
    "quality_ci": [float(x) for x in np.percentile(quality, [2.5, 97.5])]
}
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)

if __name__ == "__main__":
    summarize("results_omi_det.jsonl", "summary_omi.json")
