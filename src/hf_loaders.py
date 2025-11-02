from datasets import load_dataset

def load_hf(dataset, split="test"):
    ds = load_dataset(dataset, split=split)
    for ex in ds:
        yield {
            "id": ex.get("id", ""),
            "transcript": ex.get("dialogue", ""),
            "ref_note": ex.get("soap", ""),
            # optional placeholder for model outputs
            "gen_note": ex.get("generated_soap", ""),  # if exists
        }