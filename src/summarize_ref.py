# src/summarize_ref.py
import json

def main():
    infile = "ref_eval_gemma.jsonl"
    data = json.load(open(infile))
    summary = {
        "rougeL_mean": data.get("rougeL", 0),
        "bleu_mean": data.get("bleu", 0)
    }
    json.dump(summary, open("summary_ref_gemma.json", "w"), indent=2)
    print("✅ Summary written to summary_ref_gemma.json")

if __name__ == "__main__":
    main()
