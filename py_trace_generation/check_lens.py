import json
from transformers import AutoTokenizer

# Load the tokenizer (must match the one used for training)
# tokenizer = AutoTokenizer.from_pretrained("/data/pretrained_weights/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Read the dataset
with open("trace_train.jsonl", "r") as file:
    data = [json.loads(line) for line in file]

cutoff_len = 8192
total = len(data)
exceed_count = 0
length_stats = []

print(f"Total samples: {total}")
print(f"Cutoff length: {cutoff_len}")
print("-" * 60)

for i, entry in enumerate(data):
    # Compute the total token length of the full conversation
    conversation_text = ""
    for msg in entry["conversations"]:
        conversation_text += msg["value"] + " "
    
    tokens = tokenizer.encode(conversation_text)
    token_len = len(tokens)
    length_stats.append(token_len)
    
    if token_len > cutoff_len:
        exceed_count += 1
        print(f"Sample {i+1}: {token_len} tokens (exceeds by {token_len - cutoff_len})")

print("-" * 60)
print("\nStatistics:")
print(f"  Total samples: {total}")
print(f"  Samples exceeding cutoff: {exceed_count} ({exceed_count/total*100:.1f}%)")
print(f"  Samples within cutoff: {total - exceed_count} ({(total-exceed_count)/total*100:.1f}%)")
print("\nLength statistics:")
print(f"  Min length: {min(length_stats)} tokens")
print(f"  Max length: {max(length_stats)} tokens")
print(f"  Average length: {sum(length_stats)/len(length_stats):.0f} tokens")
print(f"  Median: {sorted(length_stats)[len(length_stats)//2]} tokens")