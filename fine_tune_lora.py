import json
import os

DATASET_FILE = "os_instruction_dataset.json"

def create_sample_instruction_dataset():
    dataset = [
        {
            "instruction": "Explain Peterson's Algorithm for process synchronization and list its constraints.",
            "input": "Module 1: Process Synchronization",
            "output": "Peterson's Algorithm is a classical software solution for two-process critical-section synchronization using two shared variables: 'boolean flag[2]' and 'int turn'. It guarantees: 1) Mutual Exclusion (only one process in critical section), 2) Progress (processes not in remainder section participate in selection), and 3) Bounded Waiting (a limit exists on process preemption before entry)."
        },
        {
            "instruction": "Compute the Turnaround and Waiting times given process completion times.",
            "input": "Module 2: CPU Scheduling",
            "output": "Turnaround Time (TAT) is calculated as: TAT = Completion Time (CT) - Arrival Time (AT). Waiting Time (WT) is calculated as: WT = Turnaround Time (TAT) - Burst Time (BT)."
        },
        {
            "instruction": "State the safety criteria for Banker's Algorithm deadlock avoidance.",
            "input": "Module 3: Deadlocks",
            "output": "A state is safe if there exists an execution sequence <P0, P1, ..., Pn-1> where for each process Pi, the remaining resource requests (Need_i) can be satisfied by the currently available resources plus resources held by prior processes: Need_i <= Work. When Pi completes, Work = Work + Allocation_i."
        },
        {
            "instruction": "What is Belady's Anomaly and which page replacement algorithm demonstrates it?",
            "input": "Module 4: Memory Management",
            "output": "Belady's Anomaly is the phenomenon where increasing the number of physical page frames results in an increase in the number of page faults for certain memory access patterns. It is typically observed in FIFO (First-In, First-Out) page replacement."
        }
    ]
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"[1/2] Generated instruction-tuning dataset: {DATASET_FILE} ({len(dataset)} pairs)")

def generate_lora_config():
    lora_config_code = '''# LoRA Training Configuration for Qwen3-14B (PEFT)
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
'''
    with open("train_config.py", "w", encoding="utf-8") as f:
        f.write(lora_config_code)
    print("[2/2] Generated LoRA training script: train_config.py")

if __name__ == "__main__":
    print("Initializing OSTutorLLM Phase 2 LoRA fine-tuning setup...")
    create_sample_instruction_dataset()
    generate_lora_config()
    print("Fine-tuning pipeline setup completed successfully!")