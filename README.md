# StateLens
Official repository for the paper "StateLens: Line-by-Line Execution State Supervision for Code LLMs"

> This repository is forked from [CodeCrash](https://github.com/cuhk-arise/CodeCrash) and extends it with execution trace generation and model training capabilities.

## 🛠️ Installation
```bash
git clone <repository-url>
cd StateLens
conda create -n statelens python=3.10
conda activate statelens
pip install -r requirements.txt
```

For training functionality, also install LLaMA-Factory:
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

## 📂 Project Structure

This project consists of three main components:

```
StateLens/
├── gpt_trace_generation/     # GPT-based trace generation
├── py_trace_generation/      # Python-based automatic trace generation
└── training/                 # Model training and inference
```

## 🔍 Trace Generation

### 1. GPT-Based Trace Generation (`gpt_trace_generation/`)

Generate execution traces using GPT models for variable tracking.

#### Generate Traces with GPT
```bash
cd gpt_trace_generation
python trace_generation.py
```

This script:
- Uses GPT-5 to generate line-by-line variable state traces
- Extracts local variables and tracks their values through execution
- Validates outputs against expected results
- Saves traces to `gpt5-variable-tracking.jsonl`

#### Convert to Training Format
```bash
python construct_data.py
```

Converts GPT-generated traces into training data format:
- Parses markdown tables from GPT responses
- Formats as conversational training examples
- Outputs to `data_gpt_trace.jsonl`

**Output Files:**
- `gpt5-variable-tracking.jsonl`: Raw GPT-generated traces
- `data_gpt_trace.jsonl`: Formatted training data

### 2. Python-Based Trace Generation (`py_trace_generation/`)

Automatically generate execution traces using Python's `sys.settrace` mechanism.

#### Basic Trace Generation
```bash
cd py_trace_generation
python generate_traces.py \
    --input cruxeval.jsonl \
    --output data_py_trace.jsonl \
    --limit 1000  # optional
```

This generates full execution traces with all loop iterations.

#### Loop-Optimized Trace Generation
```bash
python generate_traces_skip.py \
    --input cruxeval.jsonl \
    --output data_py_trace_skip.jsonl \
    --limit 1000  # optional
```

For loops with many iterations, this version:
- Shows iterations 1 and 2 in full
- Skips intermediate iterations
- Shows the final iteration
- Significantly reduces trace length for training efficiency

**Features:**
- Automatic execution tracing without LLM API calls
- Line-by-line variable state tracking
- Support for nested loops and complex control flow
- Handles special cases (e.g., rot13 encoding)

**Output Format:**
Both scripts produce JSONL files with:
- User prompt with code and function call
- Assistant response with `<think>` (execution trace) and `<answer>` (output) tags
- Training-ready conversation format

## 🚀 Model Training (`training/`)

Train models using generated execution traces.

### Fine-tuning with LLaMA-Factory

Edit `train.sh` to configure paths and hyperparameters:

```bash
#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python -m llamafactory.cli train \
  --stage sft \
  --model_name_or_path <path-to-base-model> \
  --dataset_dir <path-to-data-directory> \
  --dataset data_gpt_trace \  # or data_py_trace_skip
  --template qwen \
  --finetuning_type lora \
  --lora_target q_proj,v_proj,k_proj,o_proj,up_proj,down_proj,gate_proj \
  --lora_r 32 --lora_alpha 16 --lora_dropout 0.1 \
  --output_dir <path-to-output> \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 6 \
  --learning_rate 1e-4 \
  --cutoff_len 4096 \
  --bf16 \
  --val_size 0.1
```

Run training:
```bash
cd training
bash train.sh
```

### Inference

Run inference on trained models:

```bash
python inference.py \
  --dataset-path <path-to-test-data> \
  --model-path <path-to-trained-model> \
  --output-path predictions.jsonl \
  --device cuda:0
```

**Parameters:**
- `--dataset-path`: Input test dataset (JSONL format)
- `--model-path`: Path to trained model checkpoint
- `--output-path`: Where to save predictions
- `--device`: Device to run inference on (cuda:0, cpu, etc.)

## 🔑 API Configuration

For GPT-based trace generation, configure API keys in `.env`:

```bash
OPENAI_API_KEY="<your_openai_api_key>"
```

## 📊 Workflow

1. **Generate Training Data**
   - Option A: Use `gpt_trace_generation/` for LLM-generated traces
   - Option B: Use `py_trace_generation/` for automatic Python traces

2. **Train Model**
   - Configure `training/train.sh` with your paths
   - Run training with LLaMA-Factory

3. **Evaluate**
   - Use `training/inference.py` to generate predictions
   - Evaluate using the CodeCrash evaluation framework

## 🙏 Acknowledgement
- [CodeCrash](https://github.com/cuhk-arise/CodeCrash) - Base framework for code perturbations
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - Efficient model training framework
- [CruxEval](https://github.com/facebookresearch/cruxeval) - Code execution reasoning benchmark
