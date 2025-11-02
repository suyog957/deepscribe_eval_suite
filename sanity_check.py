
from src.hf_loaders import load_hf
dataset = "omi-health/medical-dialogue-to-soap-summary"
split = "test"
it = load_hf(dataset, split)
for i, ex in enumerate(it):
    if i == 5: break
    print("="*80)
    print("ID:", ex["id"])
    print("GEN NOTE:\n", ex.get("gen_note"))
    print("\nREF NOTE:\n", ex.get("ref_note"))

# To run this sanity check, save the above code snippet to a file named `sanity_check.py`