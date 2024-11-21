#!/usr/bin/env python

import json
import s3fs

from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import os

max_seq_length = 4096
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

bucket_name = os.getenv("BUCKET_NAME")
assert bucket_name is not None

dataset_name = os.getenv("DATASET_NAME")
assert dataset_name is not None

model_name = os.getenv("MODEL_NAME")
assert model_name is not None

home_dir = os.getenv("HOME")
assert home_dir is not None

storage_options = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "endpoint_url": "https://fly.storage.tigris.dev"
}

assert storage_options["key"] is not None
assert storage_options["secret"] is not None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"{home_dir}/tigris/models/{model_name}",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Applying the Qwen-2.5 template to the tokenizer
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

fs = s3fs.S3FileSystem(**storage_options)

# Make a LoRA model stacked on top of the base model, this is what we train and
# save for later use.
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,                       # Supports any, but = 0 is optimized
    bias = "none",                          # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

def dataset_generator(bucket_name, model_name, dataset_name, storage_options, fs):
    # Load the info.json file on the first call
    info_path = f"s3://{bucket_name}/model-ready/{model_name}/{dataset_name}/info.json"
    with fs.open(info_path, 'r') as f:
        info = json.load(f)
        #print(info)
    
    # Get the value of "biggest", defaulting to -1
    biggest = info.get("count", -1)
    assert biggest != -1, "The 'count' key must not be -1"
    
    # Generate datasets from the standardized directory
    for i in range(biggest + 1): # Assuming the biggest value is inclusive
        dataset_path = f"s3://{bucket_name}/model-ready/{model_name}/{dataset_name}/{i}"
        dataset = load_from_disk(dataset_path, storage_options=storage_options)
        yield dataset

for dataset in dataset_generator(bucket_name, model_name, dataset_name, storage_options, fs):
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            save_strategy = "steps",
            save_steps = 100,
            report_to = "none", # Use this for WandB etc
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

model.save_pretrained(f"{home_dir}/tigris/done/{model_name}/{dataset_name}/lora_model")
tokenizer.save_pretrained(f"{home_dir}/tigris/done/{model_name}/{dataset_name}/lora_model")

# Save fused model for inference with vllm
model.save_pretrained_merged(f"{home_dir}/tigris/done/{model_name}/{dataset_name}/fused", tokenizer, save_method="merged_16bit")
