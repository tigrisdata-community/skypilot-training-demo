#!/usr/bin/env python

import json
import os
import s3fs

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

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

if os.path.exists(f"{home_dir}/tigris/models/{model_name}"):
    print(f"Model {model_name} already exists")
    exit(0)

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

model.save_pretrained(f"{home_dir}/tigris/models/{model_name}")
tokenizer.save_pretrained(f"{home_dir}/tigris/models/{model_name}")