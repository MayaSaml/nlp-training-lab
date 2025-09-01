# QLoRA on TinyLlama — Detailed Notes

This document captures the full, detailed write‑up for your TinyLlama fine‑tuning project using **QLoRA** (4‑bit quantization + LoRA adapters). It’s intended for the `docs/` folder in your repo.

---

## Part 1: Model Loading and Quantization (The "Q" in QLoRA)

**Objective:** Load a large language model into a memory‑constrained GPU environment (like a T4) without sacrificing significant performance. Achieved via **4‑bit quantization** with the `bitsandbytes` library.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# 4-bit quantization configuration - Q in QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

# Load the model to train on the GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"
```

### Technical Breakdown
- **`BitsAndBytesConfig`**: Control panel for quantization, guiding how model weights are loaded into GPU memory.
- **`load_in_4bit=True`**: Core QLoRA switch. Loads weights in 4‑bit (vs. float16/bfloat16), reducing memory by ~75%.
- **`bnb_4bit_quant_type="nf4"`**: NormalFloat4, a specialized 4‑bit type that preserves distributions of typical NN weights by allocating more precision near zero.
- **`bnb_4bit_compute_dtype="float16"`**: Compute in higher precision (fp16). Weights are dequantized on‑the‑fly for matmuls to preserve accuracy.
- **`bnb_4bit_use_double_quant=True`**: Nested quantization—also quantizes quantization constants (e.g., scales), saving ~0.4 bits/param on average.
- **`AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config, device_map="auto")`**: Loads and places layers across available devices (typically `cuda:0`).
- **Tokenizer preparation**:
  - If a `pad_token` is missing (common for base LLaMA‑style models), set `pad_token = eos_token` for batching.
  - `padding_side = "left"` is best practice for decoder‑only models to avoid attention issues during generation.

---

## Part 2: LoRA Configuration (The "LoRA" in QLoRA)

**Objective:** Define how to train the quantized (and effectively frozen) model using **LoRA**—small trainable adapters inserted into select layers.

```python
from peft import LoraConfig, get_peft_model

target_modules = [
    'k_proj', 'gate_proj', 'v_proj', 'up_proj',
    'q_proj', 'o_proj', 'down_proj'
]

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules
)
```

### Technical Breakdown
- **`LoraConfig`**: Configures LoRA adapters.
- **`r=64` (rank)**: Size of the low‑rank bottleneck. Higher `r` → more capacity (and params) but higher overfitting risk.
  - PyTorch analogy: `A = nn.Linear(in, r)`, `B = nn.Linear(r, out)`.
  - Efficient forward: `W*X + A @ (B*X)` (never materialize ΔW = A@B).
- **Computation cost intuition** (example dims):
  - `W*X`: `[4096×4096] * [4096×1]` → expensive.
  - `B*X`: `[64×4096] * [4096×1]` → cheap.
  - `A @ (B*X)`: `[4096×64] * [64×1]` → cheap.
- **`lora_alpha=32`**: Forward‑pass scaling factor (applied as `alpha / r`). With `r=64`, scaling = `0.5`. Decouples adapter capacity from its influence.
  - Keep scaling consistent across `r` by adjusting `alpha` (common heuristic: `alpha = 2 * r`).
- **`lora_dropout=0.1`**: Regularizes adapter activations to reduce overfitting.
  - Applied between `B` and `A`: `h_final = W*X + (alpha/r) * (A @ dropout(B*X))`.
  - Active only during training; disabled during eval/inference.
  - Dropped activations receive zero gradient; remaining ones are scaled by `1/(1-p)`.
- **`bias="none"`**: Train only LoRA weights, not original layer biases.
- **`target_modules=[q,k,v,o,gate,up,down]_proj`**: Typical LLaMA components for attention and MLP; broad targeting yields strong results.
- **`task_type="CAUSAL_LM"` vs `"SEQ_2_SEQ_LM"`**:
  - *Causal LM*: decoder‑only (GPT‑style) next‑token prediction; PEFT injects adapters into decoder blocks accordingly.
  - *Seq2Seq*: encoder‑decoder (e.g., T5/BART); PEFT may target encoder and/or decoder stacks.

---

## Part 3: Training Configuration and Execution (SFTConfig and SFTTrainer)

**Objective:** Orchestrate optimizer, LR schedule, batching, evaluation, and logging.

```python
from trl import SFTConfig, SFTTrainer

output_dir = "output"
training_arguments = SFTConfig(
    # ... parameters ...
    optim="paged_adamw_32bit",
    # ... parameters ...
    chat_template_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)
```

### Technical Breakdown
1) **Core**  
- `output_dir="output"`: checkpoints, logs, final adapter weights.

2) **Batching & memory**  
- `per_device_train_batch_size=2`, `per_device_eval_batch_size=2`  
- `max_length=512`  
- `gradient_accumulation_steps=4` → effective batch size 8.

3) **Optimization**  
- `optim="paged_adamw_32bit"`: memory‑efficient AdamW via `bitsandbytes`.  
- `learning_rate=2e-4`: common for LoRA; cosine decay via `lr_scheduler_type="cosine"`.  
- `num_train_epochs=3`.

4) **Eval & logging**  
- `eval_strategy="steps"`, `eval_steps=25`.  
- `logging_steps=10`.

5) **Checkpointing**  
- `save_strategy="steps"`, `save_steps=50`, `save_total_limit=2`.  
- `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False`.

6) **Performance**  
- `fp16=True`, `gradient_checkpointing=True`.

7) **Task‑specific**  
- `chat_template_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0"`, `eos_token="</s>"`.

8) **Reporting**  
- `report_to="wandb"`, `run_name="tinyllama-qlora-with-eval"`.

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_arguments,
    processing_class=tokenizer,
    peft_config=peft_config,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
    ]
)
trainer.train()
```

**Notes:**  
`SFTTrainer` bundles model/datasets/args/tokenizer/PEFT config and handles training, evaluation, checkpointing, and logging. Early stopping monitors `eval_loss` and halts if no improvement ≥ 0.001 for 3 consecutive evals.

---

## Part 4: Interpretation of Training Results

### First iteration (1 epoch)

| Step | Training Loss | Validation Loss | Mean Token Accuracy |
|-----:|---------------:|----------------:|--------------------:|
| 50   | 1.419700       | 1.452835        | 0.649594            |
| 100  | 1.491700       | 1.428555        | 0.653111            |
| 150  | 1.432600       | 1.416978        | 0.655253            |
| 200  | 1.341800       | 1.409500        | 0.656532            |
| 250  | 1.386900       | 1.406413        | 0.657085            |
| 300  | 1.384500       | 1.405721        | 0.657289            |

**Analysis:**  
- **Training Loss** trends down overall, indicating learning; occasional spikes are normal.  
- **Validation Loss** decreases steadily (≈1.45 → 1.40), signaling generalization.  
- **Mean Token Accuracy** improves (≈64.9% → 65.7%), consistent with better next‑token prediction.

**Conclusion:** Successful training and generalization on unseen data.

### Second iteration (3 epochs)

| Step | Training Loss | Validation Loss | Mean Token Accuracy |
|-----:|---------------:|----------------:|--------------------:|
| 25   | 1.447500       | 1.480610        | 0.644358            |
| 50   | 1.448700       | 1.452686        | 0.649186            |
| 75   | 1.354300       | 1.435866        | 0.652099            |
| 100  | 1.499400       | 1.425786        | 0.653981            |
| 125  | 1.423200       | 1.420494        | 0.654265            |
| 150  | 1.392200       | 1.415990        | 0.655259            |
| 175  | 1.400600       | 1.411688        | 0.656099            |
| 200  | 1.435500       | 1.408333        | 0.656720            |
| 250  | 1.341000       | 1.401525        | 0.657751            |
| 275  | 1.311100       | 1.401694        | 0.657763            |
| 300  | 1.318300       | 1.407460        | 0.656587            |
| 325  | 1.333000       | 1.404904        | 0.656680            |

**Phase 1 — Healthy Learning (Steps 25–250):**  
Validation loss drops to its minimum at step 250; accuracy rises accordingly. Training loss trends downward.

**Phase 2 — Onset of Overfitting (Steps 275–325):**  
Training loss keeps improving, but validation loss worsens slightly (from the best at step 250), indicating mild overfitting.

**Early Stopping Behavior:**  
- Step 250: best model (patience reset).
- Step 275: worse than best (patience=1).
- Step 300: worse (patience=2).
- Step 325: worse (patience=3) → stop.

**Final Conclusion:** The run is successful and the configuration effective. With `load_best_model_at_end=True`, the final `trainer.model` loads the step‑250 checkpoint.

---

## Part 5: Inference

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
   "TinyLlama-1.1B-qlora-adapter",
   low_cpu_mem_usage=True,
   device_map="auto",
)

# Merge LoRA and base model
merged_model = model.merge_and_unload()
```

**Explanation:**  
- `AutoPeftModelForCausalLM.from_pretrained(...)` loads the base model and attaches trained LoRA adapter weights from `TinyLlama-1.1B-qlora-adapter`.
- `merge_and_unload()` fuses LoRA weights into the base layers, discards adapters, and returns a standard `transformers` model ready for efficient inference and deployment.

---

**End of document.**

