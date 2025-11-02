# src/generate_notes_gpu_opt.py
# Optimized batched generation on GPU with 4-bit quantization for google/gemma-2b-it
# Produces: generated_notes_gpu.jsonl containing {id, dialogue, gen_note, ref_note}

import os
import json
import math
import time
from typing import List, Dict

import torch
from datasets import load_dataset
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# -----------------------------
# Configuration
# -----------------------------
MODEL_ID = "google/gemma-2b-it"     # <- your requested model
DATASET = "omi-health/medical-dialogue-to-soap-summary"
SPLIT   = "test"                    # change if you want train/validation
OUT_PATH = "generated_notes_gpu.jsonl"

# Generation settings (tweak as needed)
MAX_INPUT_TOKENS   = 3072           # leave headroom for generation
MAX_NEW_TOKENS     = 320
TEMPERATURE        = 0.2
TOP_P              = 0.9
DO_SAMPLE          = True

# Batch size: start conservative for 8GB + 4bit; bump to 8 if VRAM allows
BATCH_SIZE         = 4

# Safety: if you hit OOM, we’ll auto-retry with smaller batch
MIN_BATCH_SIZE     = 1

# -----------------------------
# Utilities
# -----------------------------
def count_existing_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        for i, _ in enumerate(f, 1):
            pass
    return i if "i" in locals() else 0

def append_jsonl(path: str, rows: List[Dict]):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def print_gpu_info():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        print(f"🟢 GPU: {name} | total VRAM: {total_gb:.2f} GB")
    else:
        print("⚠️ CUDA not available; running on CPU.")

def build_prompt(tokenizer, dialogue: str) -> str:
    """
    Use chat template if available; fall back to a simple instruction prompt.
    """
    user_msg = (
        "Create a concise, structured SOAP note in sections S:, O:, A:, P: "
        "from the following doctor–patient conversation. Avoid extra commentary.\n\n"
        "<dialogue>\n" + dialogue.strip() + "\n</dialogue>"
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # fallback
    return f"Instruction: {user_msg}\nAnswer:"

def tokenize_batch(tokenizer, prompts: List[str]) -> Dict[str, torch.Tensor]:
    # Left padding lets us generate in batches cleanly for causal LMs
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    return encoded

def generate_batch(
    model, tokenizer, batch_prompts: List[str]
) -> List[str]:
    encoded = tokenize_batch(tokenizer, batch_prompts)
    input_ids = encoded["input_ids"].to(model.device)
    attn_mask = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode ONLY the generated portion (exclude the prompt portion per sample)
    # Determine per-sample prompt lengths from attention masks.
    prompt_lengths = attn_mask.sum(dim=1)
    results = []
    for i in range(outputs.size(0)):
        gen_tokens = outputs[i, prompt_lengths[i] : ]  # slice after prompt
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        results.append(text.strip())
    return results

# -----------------------------
# Main
# -----------------------------
def main():
    torch.backends.cudnn.benchmark = True

    print("Loading model with 4-bit quantization (bnb + NF4)...")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=torch.float16,
    )
    model.eval()

    print_gpu_info()

    print("Loading dataset...")
    ds = load_dataset(DATASET, split=SPLIT)

    # Basic sanity on the fields that exist in OMI dataset
    sample0 = ds[0]
    required = ["dialogue", "soap"]
    for k in required:
        if k not in sample0:
            raise KeyError(f"Dataset missing required key '{k}'. Keys: {list(sample0.keys())}")

    # Resume support
    done = count_existing_lines(OUT_PATH)
    total = len(ds)
    if done > 0:
        print(f"Resuming from line {done} (of {total})...")
    else:
        print(f"Writing to {OUT_PATH}")

    # Iterate in batches with auto OOM fallback
    start_idx = done
    remaining = total - start_idx
    if remaining <= 0:
        print("✅ Nothing to do; all samples already generated.")
        return

    pbar = tqdm(total=remaining, initial=0, desc="Generating", ncols=100)

    idx = start_idx
    current_bs = BATCH_SIZE
    while idx < total:
        try:
            upper = min(idx + current_bs, total)
            batch = ds.select(range(idx, upper))

            prompts = [build_prompt(tokenizer, ex["dialogue"]) for ex in batch]
            gen_texts = generate_batch(model, tokenizer, prompts)

            rows = []
            for ex, gen in zip(batch, gen_texts):
                rows.append({
                    "id": ex.get("id", ""),
                    "dialogue": ex["dialogue"],
                    "gen_note": gen,
                    "ref_note": ex["soap"],  # gold reference
                })
            append_jsonl(OUT_PATH, rows)

            pbar.update(len(rows))
            idx = upper  # advance window
            # If we had reduced batch size earlier, try to increase back up a bit
            if current_bs < BATCH_SIZE and current_bs < 8:
                current_bs += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_bs == MIN_BATCH_SIZE:
                # As a last resort, try smaller new tokens
                global MAX_NEW_TOKENS
                if MAX_NEW_TOKENS > 128:
                    MAX_NEW_TOKENS = max(128, MAX_NEW_TOKENS // 2)
                    print(f"OOM. Reducing MAX_NEW_TOKENS to {MAX_NEW_TOKENS} and retrying...")
                    continue
                else:
                    raise
            current_bs = max(MIN_BATCH_SIZE, math.floor(current_bs / 2))
            print(f"OOM. Reducing batch size to {current_bs} and retrying...")

        except Exception as e:
            # Log and skip problematic sample(s)
            print(f"--------------Error at index {idx}: {repr(e)}")
            # Try to skip a single record to keep going
            try:
                bad = ds.select([idx])
                rows = [{
                    "id": bad[0].get("id", ""),
                    "dialogue": bad[0].get("dialogue", ""),
                    "gen_note": "",
                    "ref_note": bad[0].get("soap", ""),
                    "error": str(e),
                }]
                append_jsonl(OUT_PATH, rows)
                pbar.update(1)
                idx += 1
            except Exception as e2:
                print(f"------------------Failed to skip record at {idx}: {repr(e2)}")
                # As a last resort, skip forward anyway.
                idx += 1
                pbar.update(1)

    pbar.close()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        used = torch.cuda.max_memory_reserved(0) / (1024**3)
        print(f"✅ Done. Peak reserved VRAM: {used:.2f} GB")

if __name__ == "__main__":
    main()
