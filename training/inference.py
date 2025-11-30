import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== Configuration ====================
MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.2
TOP_P = 0.95

# =================== Parsing Arguments ====================
parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset")
parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
parser.add_argument("--output-path", type=str, required=True, help="Path to save the output predictions")
parser.add_argument("--device", type=str, required=True, help="Device to run the model on")
args = parser.parse_args()

DATA_PATH = args.dataset_path
OUTPUT_PATH = args.output_path
MODEL_PATH = args.model_path

# ==================== Load Model ====================
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=args.device,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# ==================== Load Data ====================
print(f"Loading data: {DATA_PATH}")
with open(DATA_PATH, "r") as f:
    data = [json.loads(line) for line in f]

print(f"Total {len(data)} test samples loaded.")

# ==================== Define Helper Functions ====================
def extract_answer(text: str) -> str:
    if "```python" in text and "```" in text:
        return text.split("```python")[-1].split("```")[0].strip()
    return text.strip()

# ==================== Evaluation ====================
results = []

for entry in tqdm(data, desc="Generating predictions"):
    input_prompt = entry["input_prompt"]

    # Construct chat messages
    messages = [
        {"role": "user", "content": input_prompt}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(args.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    # Construct output format (keep original data, add predictions)
    answer = extract_answer(generated_text)
    result = {
        **entry,  # Keep all original fields
        "responses": [generated_text],  # Add model-generated responses
        "solutions": [answer]  # Add extracted answer
    }
    
    results.append(result)
    
    # Save intermediate results every 10 samples
    if len(results) % 10 == 0:
        print(f"\nProcessed {len(results)}/{len(data)} samples, saving intermediate results...")
        with open(OUTPUT_PATH, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ==================== Save Final Results ====================
print(f"\nSaving results to: {OUTPUT_PATH}")
with open(OUTPUT_PATH, "w") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print("Evaluation completed!")
print(f"Total {len(results)} predictions generated.")

