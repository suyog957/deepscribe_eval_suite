# src/combined_summary.py
import json

def merge(ref_path="summary_ref_gemma.json",
          nonref_path="summary_omi.json",
          out_path="summary_overall.json"):
    ref = json.load(open(ref_path))
    nonref = json.load(open(nonref_path))
    combined = {**ref, **nonref}
    json.dump(combined, open(out_path, "w"), indent=2)
    print("✅ Combined summary saved to", out_path)
    print(json.dumps(combined, indent=2))

if __name__ == "__main__":
    merge()
