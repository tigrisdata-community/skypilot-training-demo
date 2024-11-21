#!/usr/bin/env python

import json
import os
import s3fs

from datasets import load_dataset, Dataset
from unsloth import standardize_sharegpt

bucket_name = os.getenv("BUCKET_NAME")
assert bucket_name is not None

dataset_name = os.getenv("DATASET_NAME")
assert dataset_name is not None

storage_options = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "endpoint_url": "https://fly.storage.tigris.dev"
}

assert storage_options["key"] is not None
assert storage_options["secret"] is not None

fs = s3fs.S3FileSystem(**storage_options)

if fs.exists(f"/{bucket_name}/standardized/{dataset_name}"):
    print(f"Dataset {dataset_name} already exists and is standardized")
    exit(0)

dataset = load_dataset(dataset_name, split="train", streaming=True)

biggest = 0
for i, x in enumerate(dataset.iter(5_000_000)):
    if isinstance(x, dict):
        ds = Dataset.from_dict(x, features=dataset.features)
    else:
        ds = Dataset.from_generator(lambda: (yield from x), features=dataset.features)

    ds.save_to_disk(f"s3://{bucket_name}/raw/{dataset_name}/{i}", storage_options=storage_options)

    ds = standardize_sharegpt(ds)
    ds.save_to_disk(f"s3://{bucket_name}/standardized/{dataset_name}/{i}", storage_options=storage_options)

    biggest = i

fs.write_text(f"/{bucket_name}/raw/{dataset_name}/info.json", json.dumps({"count": biggest}))    