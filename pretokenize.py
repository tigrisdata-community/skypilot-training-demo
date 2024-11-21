#!/usr/bin/env python

import json
import os
import s3fs

from datasets import load_from_disk, Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

bucket_name = os.getenv("BUCKET_NAME")
assert bucket_name is not None

dataset_name = os.getenv("DATASET_NAME")
assert dataset_name is not None

model_name = os.getenv("MODEL_NAME")
assert model_name is not None

storage_options = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "endpoint_url": "https://fly.storage.tigris.dev"
}

assert storage_options["key"] is not None
assert storage_options["secret"] is not None

# maybe twiddle these?
max_seq_length = 4096
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

fs = s3fs.S3FileSystem(**storage_options)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Applying the Qwen-2.5 template to the tokenizer
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

if fs.exists(f"s3://{bucket_name}/model-ready/{model_name}/{dataset_name}/info.json"):
    print(f"Dataset {dataset_name} already exists and is model-ready")
    exit(0)

biggest = -1
with fs.open(f"s3://{bucket_name}/raw/{dataset_name}/info.json") as fin:
    data = json.load(fin)
    biggest = data["count"]

assert biggest != -1

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

for i in range(biggest+1):
    ds = load_from_disk(f"s3://{bucket_name}/standardized/{dataset_name}/{i}", storage_options=storage_options)
    
    ds = ds.map(formatting_prompts_func, batched=True,)

    ds.save_to_disk(f"s3://{bucket_name}/model-ready/{model_name}/{dataset_name}/{i}", storage_options=storage_options)

fs.write_text(f"/{bucket_name}/model-ready/{model_name}/{dataset_name}/info.json", json.dumps({"count": biggest}))