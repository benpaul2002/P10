from dataclasses import dataclass, field
import torch
from math import ceil

@dataclass
class KVCache:
    k: list[torch.Tensor] = field(default_factory=list)
    v: list[torch.Tensor] = field(default_factory=list)
    free_blocks: list[int] = field(default_factory=list)
    refcounts: list[int] = field(default_factory=list)
    registry: dict[int, tuple] = field(default_factory=dict)
    block_hashes: list[int | None] = field(default_factory=list)
    block_size: int = 16
    num_blocks: int = 128

    @classmethod
    def preallocate(cls, config, device):
        cache = cls()
        for i in range(config.num_hidden_layers):
            cache.k.append(torch.empty(cache.num_blocks, cache.block_size, config.num_key_value_heads, config.head_dim, dtype=torch.bfloat16, device=device))
            cache.v.append(torch.empty(cache.num_blocks, cache.block_size, config.num_key_value_heads, config.head_dim, dtype=torch.bfloat16, device=device))
        cache.free_blocks = list(range(cache.num_blocks))
        cache.refcounts = [0 for i in range(cache.num_blocks)]
        cache.block_hashes = [None for i in range(cache.num_blocks)]
        return cache

    def allocate(self):
        if len(self.free_blocks)==0:
            raise RuntimeError("KV pool exhausted")
        block_id = self.free_blocks.pop(0)
        h = self.block_hashes[block_id]
        if h is not None:
            del self.registry[h]
            self.block_hashes[block_id] = None
        self.refcounts[block_id] += 1
        return block_id

    def free(self, block_id):
        if self.refcounts[block_id] == 0:
            raise RuntimeError("Can't call free on block with refcount == 0!")
        self.refcounts[block_id] -= 1
        if self.refcounts[block_id] == 0:
            self.free_blocks.append(block_id)

    def truncate(self, seq, new_length):
        keep = ceil(new_length / self.block_size)
        seq.num_registered = min(seq.num_registered, new_length // self.block_size)
        for block_id in seq.block_table[keep:]:
            self.free(block_id)
        seq.block_table = seq.block_table[:keep]
        seq.length = new_length
        seq.token_ids = seq.token_ids[:new_length]
        h = None
        for i in range(seq.num_registered):
            h = self.block_hash(tuple(seq.token_ids[i*self.block_size:(i+1)*self.block_size]), h)
        seq.prefix_hash = h

    def incref(self, block_id):
        if block_id in self.free_blocks:
            self.free_blocks.remove(block_id)
        self.refcounts[block_id] += 1

    def block_hash(self, token_ids, prev_hash):
        return hash((prev_hash, tuple(token_ids)))

    def register(self, seq):
        prev_hash = seq.prefix_hash
        for i in range(seq.num_registered, seq.length // self.block_size):
            start = i * self.block_size
            end = (i+1) * self.block_size
            chunk = tuple(seq.token_ids[start:end])
            h = self.block_hash(chunk, prev_hash)
            if h not in self.registry:
                self.registry[h] = (seq.block_table[i], chunk)
                self.block_hashes[seq.block_table[i]] = h
            prev_hash = h
        seq.num_registered = seq.length // self.block_size
        seq.prefix_hash = prev_hash

    def probe_prefix(self, seq):
        num_matched = 0
        num_to_revive = 0
        prev_hash = None
        for i in range((len(seq.token_ids)-1) // self.block_size):
            start = i * self.block_size
            end = (i+1) * self.block_size
            chunk = tuple(seq.token_ids[start:end])
            h = self.block_hash(chunk, prev_hash)
            if h not in self.registry or self.registry[h][1]!=chunk:
                break
            prev_hash = h
            num_matched += 1
            if self.refcounts[self.registry[h][0]] == 0:
                num_to_revive += 1
        return num_matched, num_to_revive

    def match_prefix(self, seq):
        matched = []
        prev_hash = None
        for i in range((len(seq.token_ids)-1) // self.block_size):
            start = i * self.block_size
            end = (i+1) * self.block_size
            chunk = tuple(seq.token_ids[start:end])
            h = self.block_hash(chunk, prev_hash)
            if h not in self.registry or self.registry[h][1]!=chunk:
                break
            matched.append(self.registry[h][0])
            self.incref(self.registry[h][0])
            prev_hash = h
        seq.block_table = matched
        seq.length = len(matched) * self.block_size
        seq.num_registered = len(matched)
        seq.prefix_hash = prev_hash
        return len(matched)

    def ensure_capacity(self, seq, n_new_tokens):
        block_table = seq.block_table
        blocks_needed = ceil((seq.length + n_new_tokens) / self.block_size)
        while len(block_table) < blocks_needed:
            block_id = self.allocate()
            block_table.append(block_id)

    def compute_blockId_offset(self, seq, start, end, device):
        positions = torch.arange(start, end, device=device)
        logical = positions // self.block_size
        offsets = positions % self.block_size
        table = torch.tensor(seq.block_table, device=device)
        block_ids = table[logical]
        return block_ids, offsets

    def scatter_prefill(self, seq, layer_idx, k_new, v_new):
        num_new_tokens = k_new.shape[2]
        device = k_new.device
        block_ids, offsets = self.compute_blockId_offset(seq, seq.length-num_new_tokens, seq.length, device)
        k_new_reshaped = k_new.squeeze(0).transpose(0, 1)
        v_new_reshaped = v_new.squeeze(0).transpose(0, 1)
        self.k[layer_idx][block_ids, offsets] = k_new_reshaped
        self.v[layer_idx][block_ids, offsets] = v_new_reshaped

    def scatter_decode(self, seq_list, layer_idx, k_new, v_new):
        num_new_tokens = k_new.shape[2]
        device = k_new.device
        positions = [seq.length-1 for seq in seq_list]
        logical = [position//self.block_size for position in positions]
        offsets = [position%self.block_size for position in positions]
        block_ids = [seq.block_table[l] for seq, l in zip(seq_list, logical)]
        k_new_reshaped = k_new.squeeze(2)
        v_new_reshaped = v_new.squeeze(2)
        self.k[layer_idx][block_ids, offsets] = k_new_reshaped
        self.v[layer_idx][block_ids, offsets] = v_new_reshaped

    def gather_prefill(self, seq, layer_idx):
        device = self.k[layer_idx].device
        block_ids, offsets = self.compute_blockId_offset(seq, 0, seq.length, device)
        k = self.k[layer_idx][block_ids, offsets]
        v = self.v[layer_idx][block_ids, offsets]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        return k, v

    def gather_decode(self, seq_list, layer_idx):
        device = self.k[layer_idx].device
        dtype = self.k[layer_idx].dtype
        max_len = max(seq.length for seq in seq_list)
        k_out = torch.zeros([len(seq_list), self.k[layer_idx].shape[2], max_len, self.k[layer_idx].shape[3]], dtype=dtype, device=device)
        v_out = torch.zeros([len(seq_list), self.k[layer_idx].shape[2], max_len, self.k[layer_idx].shape[3]], dtype=dtype, device=device)
        for i, seq in enumerate(seq_list):
            block_ids, offsets = self.compute_blockId_offset(seq, 0, seq.length, device)
            k_out[i, :, :seq.length] = self.k[layer_idx][block_ids, offsets].transpose(0, 1)
            v_out[i, :, :seq.length] = self.v[layer_idx][block_ids, offsets].transpose(0, 1)
        lengths = torch.tensor([seq.length for seq in seq_list], device=device)
        positions = torch.arange(max_len, device=device)
        mask = positions[None, :] < lengths[:, None]
        mask = mask[:, None, None, :]
        return k_out, v_out, mask

@dataclass
class Sequence:
    length: int = 0
    block_table: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    num_registered: int = 0
    prefix_hash: int | None = None
