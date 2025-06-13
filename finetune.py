#!/usr/bin/env python3
# coding: utf-8

import os
import json
from os import path
from typing import Callable, Dict, Optional
import logging
from pandas.io.stata import max_len_string_array
import yaml

from librosa import example
from numpy import save
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from transformers.trainer_pt_utils import save_state
from datasets import load_dataset, Dataset, Audio, load_from_disk
from snac import SNAC
from peft import LoraConfig, get_peft_model
import torch
import torchaudio.transforms as T
from unsloth import FastModel, FastLanguageModel, is_bfloat16_supported

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_file = "config_finetuning.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config['encoded_dataset']

model_name = config["model_name"]
tokenizer_name = config["tokenizer_name"]
audio_encoder_name = config['audio_encoder_name']

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]
config_ratio = config["ratio"]

lora_rank = 32
lora_alpha = 64
lora_dropout = 0.0

def data_collator(features):
    # max_length = 2656 # set a crop based on vram - ideally you have stacked all sequences to the same length
    # from 3b on 8 h100s fsdp, at bf16, 8192 works well.

    # print(
    #     torch.Tensor(features[0]['input_ids']).size(),
    #     torch.Tensor(features[0]['attention_mask']).size(),
    #     torch.Tensor(features[0]['labels']).size()
    # )

    input_ids = [f["input_ids"] for f in features]

    if any("attention_mask" not in f for f in features):
        attention_mask = [[1]*len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]

    if any("labels" not in f for f in features):
        labels = input_ids
    else:
        labels = [f["labels"] for f in features]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x, dtype=torch.long) for x in input_ids],
        batch_first=True,
        padding_value=pad_token
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(m, dtype=torch.long) for m in attention_mask],
        batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(l, dtype=torch.long) for l in labels],
        batch_first=True, padding_value=-100
    )

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa")
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length= 2048, # Choose any for long context!
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    #token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj",  "o_proj", "gate_proj", "down_proj", "up_proj"],
    bias="none",
    modules_to_save=["lm_head", "embed_tokens"], # Optional to train the embeddings and lm head
    task_type="CAUSAL_LM",
    use_rslora=True,
)

model = get_peft_model(model, lora_config)

ds_train = load_from_disk(dsn)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    #num_train_epochs=epochs,
    max_steps=20,
    #per_device_train_batch_size=batch_size,
    per_device_train_batch_size=1,
    learning_rate=learning_rate,
    warmup_steps=50,
    logging_steps=10,
    #bf16=True,
    fp16 = not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    output_dir=f"./{base_repo_id}",
    report_to="tensorboard",
    save_steps=save_steps,
    save_strategy='steps',
    save_total_limit=3,
    # load_best_model_at_end=True,
    # metric_for_best_model="loss",
    # greater_is_better=False,
    remove_unused_columns=True,
    label_names=['labels'],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    data_collator=data_collator
    # eval_dataset=ds_eval
)

trainer.train()

merged_model = model.merge_and_unload()

merged_model.save_pretrained(f"./{base_repo_id}/merged")
tokenizer.save_pretrained(f"./{base_repo_id}/merged")

# model.save_pretrained(f"./{base_repo_id}/lora_model")  # Local saving
# tokenizer.save_pretrained("./{base_repo_id}/lora_model")

