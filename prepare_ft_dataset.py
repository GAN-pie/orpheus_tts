#!/usr/bin/env python3
# coding: utf-8

import logging
import yaml
import os

from datasets import load_dataset, Dataset, Audio
from snac import SNAC
import torch
import torchaudio.transforms as T
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_file = "config_finetuning.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config['TTS_dataset']

ds = load_dataset(dsn)
ds_sample_rate = ds['train'][0]["audio"]["sampling_rate"]

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
model = model.to("cuda")

resampler = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)

def tokenise_audio(waveform):
  waveform = torch.from_numpy(waveform).unsqueeze(0)
  waveform = waveform.to(dtype=torch.float32)
  waveform = resampler(waveform)

  waveform = waveform.unsqueeze(0).to("cuda")

  #generate the codes from snac
  with torch.inference_mode():
    codes = model.encode(waveform)

  all_codes = []
  for i in range(codes[0].shape[1]):
    all_codes.append(codes[0][0][i].item()+128266)
    all_codes.append(codes[1][0][2*i].item()+128266+4096)
    all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
    all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
    all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
    all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
    all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))

  return all_codes

def add_codes(example):
    # Always initialize codes_list to None
    codes_list = None

    try:
        answer_audio = example.get("audio")
        # If there's a valid audio array, tokenise it
        if answer_audio and "array" in answer_audio:
            audio_array = answer_audio["array"]
            codes_list = tokenise_audio(audio_array)
    except Exception as e:
        logger.info(f"Skipping row due to error: {e}")
        # Keep codes_list as None if we fail
    example["codes_list"] = codes_list

    return example

ds = ds.map(add_codes, remove_columns=["audio"])

tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai =  tokeniser_length + 6
pad_token = tokeniser_length + 7

audio_tokens_start = tokeniser_length + 10

tokenizer_name = config['tokenizer_name']

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
num_proc = os.cpu_count() - 8

ds = ds.filter(lambda x: x["codes_list"] is not None)
ds = ds.filter(lambda x: len(x["codes_list"]) > 0)

def remove_duplicate_frames(example):
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = vals[:7]

    removed_frames = 0

    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(vals[i:i+7])
        else:
            removed_frames += 1

    example["codes_list"] = result

    return example

ds = ds.map(remove_duplicate_frames, num_proc=num_proc)

# tok_info = '''*** HERE you can modify the text prompt
# i.e. if you wanted a multispeaker model like canopylabs/orpheus-3b-0.1-ft, you can pass:
# f"{example["source"]}:  {example["text"]}", as is passed.
# '''
# print(tok_info)

def create_input_ids(example):
    text_ids = tokenizer.encode(example["text"],  add_special_tokens=True)
    text_ids.append(end_of_text)
    example["text_tokens"] = text_ids
    input_ids = (
        [start_of_human]
        + example["text_tokens"]
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech]
        + [end_of_ai]
    )
    example["input_ids"] = input_ids
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)

    return example

ds = ds.map(create_input_ids, num_proc=num_proc, remove_columns=["text", "codes_list"])

columns_to_keep = ["input_ids", "labels", "attention_mask"]
columns_to_remove = [col for col in ds.column_names['train'] if col not in columns_to_keep]

ds = ds.remove_columns(columns_to_remove)

ds.save_to_disk('data/encoded')
