from datasets import load_dataset

dataset = load_dataset("omi-health/medical-dialogue-to-soap-summary", split="test")
print(dataset[0].keys())
print("\nExample:\n")
print("DIALOGUE:\n", dataset[0]["dialogue"][:400], "...")
print("\nSOAP:\n", dataset[0]["soap"][:400], "...")
print("\nPROMPT:\n", dataset[0]["prompt"][:200], "...")