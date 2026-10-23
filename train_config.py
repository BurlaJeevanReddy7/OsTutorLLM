# LoRA Training Configuration for Qwen3-14B (PEFT)
from peft import LoraConfig, TaskType
from transformers import TrainingArguments

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none"
)

training_args = TrainingArguments(
    output_dir="./qwen_lora_os_tutor",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_ratio=0.03,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    num_train_epochs=3
)
print("LoRA Adapter Configuration ready for Qwen3-14B.")
