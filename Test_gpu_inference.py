from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
print("🚀 Loading model on GPU with quantization + CPU offload...")

# ---- Quantization setup --------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                         # quantize to 8-bit
    llm_int8_enable_fp32_cpu_offload=True,     # allow spill-over to CPU
)

# ---- Model + tokenizer ---------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---- Pipeline ------------------------------------------------------------
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

# ---- Test inference ------------------------------------------------------
prompt = "Summarize this doctor–patient conversation: Patient reports sore throat for 2 days."
out = gen(prompt, max_new_tokens=80, temperature=0.4, top_p=0.9)
print("\n🩺 Generated SOAP-style summary:\n", out[0]["generated_text"])

# ---- Diagnostics ---------------------------------------------------------
if torch.cuda.is_available():
    print(f"\n✅ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB total")
else:
    print("⚠️  CUDA not available — running on CPU.")
