# src/score_vs_reference.py
import json
import evaluate
from tqdm import tqdm

def compute_metrics(preds, refs):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    rouge_res = rouge.compute(predictions=preds, references=refs)
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    return {"rougeL": rouge_res["rougeL"], "bleu": bleu_res["score"]}

def main():
    infile = "generated_notes_gpu.jsonl"
    outfile = "ref_eval_gemma.jsonl"
    results = []

    preds, refs = [], []
    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            preds.append(j["gen_note"])
            refs.append(j["ref_note"])

    print("Evaluating metrics...")
    metrics = compute_metrics(preds, refs)
    print(metrics)

    with open(outfile, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Results saved to:", outfile)

if __name__ == "__main__":
    main()
