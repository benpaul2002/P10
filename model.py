from dataclasses import dataclass, field
import json
from safetensors import safe_open
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    max_position_embeddings: int
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
        model_config_full['max_position_embeddings'],
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

@dataclass
class KVCache:
    k: list[torch.Tensor] = field(default_factory=list)
    v: list[torch.Tensor] = field(default_factory=list)
    fill_pos: int = 0

    @classmethod
    def preallocate(cls, config, batch_size, device):
        cache = cls()
        for i in range(config.num_hidden_layers):
            cache.k.append(torch.empty(batch_size, config.num_key_value_heads, config.max_position_embeddings, config.head_dim, dtype=torch.bfloat16, device=device))
            cache.v.append(torch.empty(batch_size, config.num_key_value_heads, config.max_position_embeddings, config.head_dim, dtype=torch.bfloat16, device=device))
        cache.fill_pos = 0
        return cache

def rmsnorm(x, weight, eps):
    dtype = x.dtype
    x = x.float()
    x2 = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(x2 + eps)
    x = x.to(dtype)
    return x * weight

def rope(head_dim, max_seq_len, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x):
    first, second = torch.chunk(x, 2, dim=-1)
    return torch.cat([-second, first], dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    dtype = q.dtype
    q_out = (q*cos + rotate_half(q)*sin).to(dtype)
    k_out = (k*cos + rotate_half(k)*sin).to(dtype)
    return q_out, k_out

def gqa_attention(x, weights, layer_idx, config, cos, sin, kvcache, start_pos):
    q_w = weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
    q_b = weights[f"model.layers.{layer_idx}.self_attn.q_proj.bias"]
    k_w = weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
    k_b = weights[f"model.layers.{layer_idx}.self_attn.k_proj.bias"]
    v_w = weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]
    v_b = weights[f"model.layers.{layer_idx}.self_attn.v_proj.bias"]
    o_w = weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]

    q = F.linear(x, q_w, q_b)
    k = F.linear(x, k_w, k_b)
    v = F.linear(x, v_w, v_b)

    batch_size, seq_len, _ = x.shape
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    q_out, k_out = apply_rope(q, k, cos, sin)

    kvcache.k[layer_idx][:, :, start_pos : start_pos + seq_len] = k_out
    kvcache.v[layer_idx][:, :, start_pos : start_pos + seq_len] = v
    k_out = kvcache.k[layer_idx][:, :, : start_pos + seq_len]
    v = kvcache.v[layer_idx][:, :, : start_pos + seq_len]

    k_out = torch.repeat_interleave(k_out, n_heads//n_kv_heads, dim=1)
    v = torch.repeat_interleave(v, n_heads//n_kv_heads, dim=1)
    x = F.scaled_dot_product_attention(q_out, k_out, v, is_causal=start_pos==0)
    x = x.transpose(1, 2).reshape(batch_size, seq_len, config.hidden_size)
    return F.linear(x, o_w)

def swiglu(x, weights, layer_idx):
    g_w = weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
    u_w = weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
    d_w = weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
    return F.linear(F.silu(F.linear(x, g_w)) * F.linear(x, u_w), d_w)

def decoder_layer(x, weights, layer_idx, config, cos, sin, start_pos, seq_len, kvcache):
    input_layernorm_weight = weights[f"model.layers.{layer_idx}.input_layernorm.weight"]
    post_attention_layernorm_weight = weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]
    eps = config.rms_norm_eps
    sliced_cos = cos[start_pos : start_pos+seq_len]
    sliced_sin = sin[start_pos : start_pos+seq_len]
    h = x + gqa_attention(rmsnorm(x, input_layernorm_weight, eps), weights, layer_idx, config, sliced_cos, sliced_sin, kvcache, start_pos)
    out = h + swiglu(rmsnorm(h, post_attention_layernorm_weight, eps), weights, layer_idx)
    return out

@torch.inference_mode()
def forward(token_ids, weights, config, cos, sin, kvcache):
    batch_size, seq_len = token_ids.shape
    start_pos = kvcache.fill_pos
    x = F.embedding(token_ids, weights["model.embed_tokens.weight"])
    for i in range(config.num_hidden_layers):
        x = decoder_layer(x, weights, i, config, cos, sin, start_pos, seq_len, kvcache)
    x = rmsnorm(x, weights["model.norm.weight"], config.rms_norm_eps)
    kvcache.fill_pos += seq_len
    return F.linear(x, weights["model.embed_tokens.weight"])

@torch.inference_mode()
def prefill(token_ids, weights, config, cos, sin, kvcache):
    resp = forward(token_ids, weights, config, cos, sin, kvcache)
    return resp[:, -1]

@torch.inference_mode()
def decode(token_id, weights, config, cos, sin, kvcache):
    resp = forward(token_id, weights, config, cos, sin, kvcache)
    return resp[:, -1]

@torch.inference_mode()
def generate(prompt, max_new_tokens, tokenizer, weights, config, cos, sin, kvcache):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    generated = []
    logits = prefill(token_ids, weights, config, cos, sin, kvcache)
    last_token = logits.argmax(-1, keepdim=True)
    generated.append(last_token.item())

    for i in range(max_new_tokens-1):
        logits = decode(last_token, weights, config, cos, sin, kvcache)
        last_token = logits.argmax(-1, keepdim=True)
        generated.append(last_token.item())
        if last_token.item() == 151645:
            break

    return generated

def build_sincos_table(config, device):
    cos, sin = rope(config.head_dim, config.max_position_embeddings, config.rope_theta, device)
    return cos, sin

