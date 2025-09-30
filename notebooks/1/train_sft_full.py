# Import required libraries for fine-tuning
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from datasets import load_dataset
import torch
# import trackio as wandb

import os

os.environ["TRACKIO_PROJECT_NAME"] = "smol-course-gemma3-270m-finetuning"
os.environ["TRACKIO_SPACE_ID"] = "pareshppp/smol-course"

# wandb.init(project="smol-course-smollm3-finetuning")

def load_model():
    ############### Load Pre-trained Model and Tokenizer ###############
    print("=== LOADING MODEL ===\n")
    # Load SmolLM3 base model for fine-tuning
    # model_name = "HuggingFaceTB/SmolLM3-3B"
    # model_name = "google/gemma-3-270m"
    model_name = "google/gemma-3-270m-it"
    # new_model_name = "SmolLM3-Custom-SFT"
    new_model_name = "Gemma3-270M-Custom-SFT"

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        # torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    tokenizer.padding_side = "right"  # Padding on the right for generation

    print(f"Model loaded! Parameters: {model.num_parameters():,}")
    return model, tokenizer, new_model_name


def apply_chat_template_to_dataset(dataset, tokenizer):
    """Apply chat template to dataset for training"""

    def format_messages(examples):
        formatted_texts = []

        for messages in examples["messages"]:
            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # We want the complete conversation
            )
            formatted_texts.append(formatted_text)

        return {"text": formatted_texts}

    return dataset.map(format_messages, batched=True)

def prepare_dataset(tokenizer):
    ############### Load and Prepare Dataset ###############
    print("=== PREPARING DATASET ===\n")

    # Option 1: Use SmolTalk2 (recommended for beginners)
    dataset = load_dataset("HuggingFaceTB/smoltalk2", "SFT")
    training_split = "smoltalk_everyday_convs_reasoning_Qwen3_32B_think"
    train_dataset = dataset[training_split].select(range(1000))  # type: ignore # Use subset for faster training

    train_dataset = apply_chat_template_to_dataset(train_dataset, tokenizer)

    # Option 2: Custom Dataset (uncomment to use)
    # from datasets import load_from_disk
    # dataset_path = "path/to/your/dataset"
    # dataset = load_from_disk(dataset_path)
    # train_dataset = dataset["train"]

    print(f"Training samples: {len(train_dataset)}")
    return train_dataset

def get_sft_config(new_model_name):
    ############### Configure SFT Training ###############
    # Configure training parameters
    training_config = SFTConfig(
        # Model and data
        output_dir=f"./{new_model_name}",
        dataset_text_field="text",
        max_length=2048,

        # Training hyperparameters
        per_device_train_batch_size=1,  # Adjust based on your GPU memory
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=1,  # Start with 1 epoch
        max_steps=500,  # Limit steps for demo

        # Optimization
        warmup_steps=50,
        weight_decay=0.01,
        optim="adamw_torch",

        # Logging and saving
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2, # Limit total saved checkpoints

        # Memory optimization
        dataloader_num_workers=0,
        group_by_length=True,  # Group similar length sequences

        # Hugging Face Hub integration
        push_to_hub=False,  # Set to True to upload to Hub
        hub_model_id=f"your-username/{new_model_name}",

        # Experiment tracking
        report_to=["trackio"],  # Use trackio for experiment tracking
        # report_to=["none"],  # Disable reporting for demo
        run_name=f"{new_model_name}-training",
    )

    print("Training configuration set!")
    print(f"Effective batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")
    return training_config

def run_training():
    ############### Initialize Trainer and Start Training ###############
    print("=== INITIALIZING TRAINER ===\n")

    model, tokenizer, new_model_name = load_model()
    train_dataset = prepare_dataset(tokenizer=tokenizer)
    training_config = get_sft_config(new_model_name)


    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Save the fine-tuned model
    trainer.save_model(training_config.output_dir)
    print(f"Model saved to {training_config.output_dir}")

if __name__ == "__main__":
    run_training()
    # wandb.finish()