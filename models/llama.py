import torch
import numpy as np
from jax import grad,vmap
from tqdm.notebook import tqdm

from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer
)
from data.serialize import serialize_arr, SerializerSettings

DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def llama2_model_string(model_size, chat):
    chat = "chat-" if chat else ""
    return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

def get_tokenizer(model):
    name_parts = model.split("-")
    model_size = name_parts[0]
    chat = len(name_parts) > 1

    assert model_size in ["7b", "13b", "70b"]  
    
    tokenizer = LlamaTokenizer.from_pretrained(
    llama2_model_string(model_size, chat),
    use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer  

def get_model_and_tokenizer(model):
    name_parts = model.split("-")
    model_size = name_parts[0]
    chat = len(name_parts) > 1

    assert model_size in ["7b", "13b", "70b"]
    tokenizer = get_tokenizer(model)
    model = LlamaForCausalLM.from_pretrained(
        llama2_model_string(model_size, chat),
        device_map="auto",   
        torch_dtype=torch.float16,
    )

    model.eval()

    return model, tokenizer

def tokenize_fn(str, model):
    tokenizer = get_tokenizer(model)
    return tokenizer(str)
