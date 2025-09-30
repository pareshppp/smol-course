# Import required libraries for fine-tuning
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from peft import LoraConfig
from datasets import load_dataset
import torch
from dotenv import load_dotenv

import os

load_dotenv()  # Load environment variables from .env file

os.environ["TRACKIO_PROJECT_NAME"] = "smol-course-gemma3-270m-finetuning-lora"
os.environ["TRACKIO_SPACE_ID"] = "pareshppp/smol-course"


def get_chat_template_gemma3_270m():
    custom_chat_template_gemma = """
{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '

' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '

' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '
' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>
' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model
'}}
{%- endif -%}
"""
    return custom_chat_template_gemma


def apply_conversation_fix(dataset):
    def fix_conversation_format_smoltalk_everyday_convs_reasoning_Qwen3_32B(example):
        # conversation roles should be in the format
        # user/assistant/user/assistant/...
        # but some examples have assistant/assistant or user/user
        # so we need to fix that
        conversations = example["messages"]
        fixed_conversations = []
        for conversation in conversations:
            fixed_conversation = []
            last_role = None
            for turn in conversation:
                role = turn["role"]
                if role == "system" or role == "tool":
                    # skip system and tool messages
                    continue
                if role == "user" and last_role == "user":
                    turn["role"] = "assistant"
                elif role == "assistant" and last_role == "assistant":
                    turn["role"] = "user"
                fixed_conversation.append(turn)
                last_role = role
            fixed_conversations.append(fixed_conversation)
        example["messages"] = fixed_conversations
        return example

    return dataset.map(
        fix_conversation_format_smoltalk_everyday_convs_reasoning_Qwen3_32B,
        batched=True,
    )


def apply_chat_template_to_dataset(dataset, tokenizer):
    """Apply chat template to dataset for training"""

    if tokenizer.name_or_path == "google/gemma-3-270m":
        tokenizer.chat_template = get_chat_template_gemma3_270m()

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


def load_model(model_name):
    ############### Load Pre-trained Model and Tokenizer ###############
    print("=== LOADING MODEL ===\n")

    if model_name == "google/gemma-3-270m":
        attn_implementation = "eager"  # for gemma models
    else:
        attn_implementation = "sdpa"

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        # torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    tokenizer.padding_side = "right"  # Padding on the right for generation

    print(f"Model loaded! Parameters: {model.num_parameters():,}")
    return model, tokenizer


def prepare_dataset(dataset_name_path, tokenizer):
    ############### Load and Prepare Dataset ###############
    print("=== PREPARING DATASET ===\n")

    # Option 1: Use SmolTalk2 (recommended for beginners)
    dataset = load_dataset(dataset_name_path, "SFT")
    training_split = "smoltalk_everyday_convs_reasoning_Qwen3_32B_think"
    train_dataset = dataset[training_split].select(range(1000))  # type: ignore # Use subset for faster training
    eval_dataset = dataset[training_split].select(range(1000, 1500))  # type: ignore
    # test_dataset = dataset[training_split].select(range(1500,2000)) # type: ignore

    # Fix conversation format issues
    train_dataset = apply_conversation_fix(train_dataset)
    eval_dataset = apply_conversation_fix(eval_dataset)
    # Apply chat template
    train_dataset = apply_chat_template_to_dataset(train_dataset, tokenizer)
    eval_dataset = apply_chat_template_to_dataset(eval_dataset, tokenizer)

    # Option 2: Custom Dataset (uncomment to use)
    # from datasets import load_from_disk
    # dataset_path = "path/to/your/dataset"
    # dataset = load_from_disk(dataset_path)
    # train_dataset = dataset["train"]

    print(f"Training samples: {len(train_dataset)}")
    return train_dataset, eval_dataset


def get_sft_config_lora(
    run_name=None,
    model_output_dir=None,
    hub_model_id=None,
    hub_model_username=None,
    push_to_hub=False,
):
    ############### Configure SFT Training ###############

    # Configure training parameters
    training_config = SFTConfig(
        # Model and data
        output_dir=model_output_dir,
        dataset_text_field="text",
        max_length=2048,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        # Training hyperparameters
        per_device_train_batch_size=1,  # Adjust based on your GPU memory
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        # num_train_epochs=3,  # Start with 1 epoch
        max_steps=2000,  # Limit steps for demo
        # Optimization
        warmup_steps=50,
        weight_decay=0.01,
        optim="adamw_torch",
        # Logging and saving
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",  # steps/epoch/no
        eval_steps=50,
        eval_on_start=True,
        save_total_limit=2,  # Limit total saved checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # Memory optimization
        dataloader_num_workers=0,
        group_by_length=True,  # Group similar length sequences
        # Hugging Face Hub integration
        push_to_hub=push_to_hub,  # Set to True to upload to Hub
        hub_model_id=f"{hub_model_username}/{hub_model_id}",
        # Experiment tracking
        report_to=["trackio"],  # Use trackio for experiment tracking
        # report_to=["none"],  # Disable reporting for demo
        run_name=run_name,
    )

    print("Training configuration set!")
    print(
        f"Effective batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}"
    )
    return training_config


def get_lora_config():

    ############### Configure LoRA (if using) ###############
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return lora_config


def run_training(
    model_name: str = "HuggingFaceTB/SmolLM3-3B",
    dataset_name_path: str = "HuggingFaceTB/smoltalk2",
    model_output_dir: str | None = None,
    hub_model_username: str = "pareshppp",
    push_to_hub: bool = False,
):
    ############### Initialize Trainer and Start Training ###############
    print("=== INITIALIZING TRAINER ===\n")

    hub_model_id = (
        model_name.split("/")[-1] + "-SFT-LoRA" + dataset_name_path.split("/")[-1]
    )

    if model_output_dir:
        run_name = model_output_dir.split("/")[-1]  # type: ignore
        print(f"Resuming from checkpoint with run_name: {run_name}")
    else:
        # Create new run
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{hub_model_id}-{timestamp}"
        model_output_dir = f"./models/{run_name}"
        print(f"Starting new run: {run_name} at {model_output_dir}")

    model, tokenizer = load_model(model_name=model_name)

    train_dataset, eval_dataset = prepare_dataset(
        dataset_name_path=dataset_name_path, tokenizer=tokenizer
    )
    peft_config = get_lora_config()
    training_config = get_sft_config_lora(
        run_name=run_name,
        model_output_dir=model_output_dir,
        hub_model_id=hub_model_id,
        hub_model_username=hub_model_username,
        push_to_hub=push_to_hub,
    )

    # Initialize EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # Stop if no improvement in 3 evaluation steps
        early_stopping_threshold=0.01,  # Minimum change to be considered an improvement
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_config,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[early_stopping_callback],
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Training complete!")

    # Save the fine-tuned model
    trainer.save_model(training_config.output_dir)
    print(f"Model saved to {training_config.output_dir}")


if __name__ == "__main__":

    # model_name = "HuggingFaceTB/SmolLM3-3B"
    model_name = "google/gemma-3-270m"

    dataset_name_path = "HuggingFaceTB/smoltalk2"

    resume_from_checkpoint = False  # Set to True to resume from last checkpoint

    ############### Run Training ###############

    run_training(
        model_name=model_name,
        dataset_name_path=dataset_name_path,
        model_output_dir=None,  # Set to None to start a new run
        # model_output_dir="./models/gemma-3-270m-SFT-LoRA-20250930-061342",  # Example path to resume
        hub_model_username="pareshppp",
        push_to_hub=True,
    )
