from dataclasses import dataclass
import json
from safetensors import safe_open
import torch

qwen0_5_snapshot_path = "/home/bnp24202/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/"

@dataclass
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    head_dim: int

def load_config(model_snapshot_path):
    model_config_path = model_snapshot_path + 'config.json'
    with open(model_config_path, "r") as file:
        model_config_full = json.load(file)
    return ModelConfig(
        model_config_full['hidden_size'],
        model_config_full['num_hidden_layers'],
        model_config_full['num_attention_heads'],
        model_config_full['num_key_value_heads'],
        model_config_full['intermediate_size'],
        model_config_full['vocab_size'],
        model_config_full['rms_norm_eps'],
        model_config_full['rope_theta'],
        model_config_full['tie_word_embeddings'],
        model_config_full['hidden_size'] // model_config_full['num_attention_heads']
    )

def load_weights(model_snapshot_path):
    model_weights_path = model_snapshot_path + 'model.safetensors'
    model_weights = {}
    with safe_open(model_weights_path, framework="pt", device="cuda") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            model_weights[key] = tensor
    return model_weights

def rmsnorm(x, weight, eps):
    dtype = x.dtype
    x = x.float()
    x2 = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(x2 + eps)
    x = x.to(dtype)
    return x * weight

load_config(qwen0_5_snapshot_path)
load_weights(qwen0_5_snapshot_path)
