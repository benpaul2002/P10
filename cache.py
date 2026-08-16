from dataclasses import dataclass, field
import torch
from math import ceil

@dataclass
class KVCache:
    k: list[torch.Tensor] = field(default_factory=list)
    v: list[torch.Tensor] = field(default_factory=list)
    free_blocks: list[int] = field(default_factory=list)
    block_size: int = 16
    num_blocks: int = 128

    @classmethod
    def preallocate(cls, config, device):
        cache = cls()
        for i in range(config.num_hidden_layers):
            cache.k.append(torch.empty(cache.num_blocks, cache.block_size, config.num_key_value_heads, config.head_dim, dtype=torch.bfloat16, device=device))
            cache.v.append(torch.empty(cache.num_blocks, cache.block_size, config.num_key_value_heads, config.head_dim, dtype=torch.bfloat16, device=device))
        cache.free_blocks = list(range(cache.num_blocks))
        return cache

    def allocate(self):
        if len(self.free_blocks)==0:
            raise RuntimeError("KV pool exhausted")
        return self.free_blocks.pop()

    def free(self, block_id):
        self.free_blocks.append(block_id)

    def ensure_capacity(self, seq, n_new_tokens):
        block_table = seq.block_table
        blocks_needed = ceil((seq.length + n_new_tokens) / self.block_size)
        while len(block_table) < blocks_needed:
            block_table.append(self.allocate())

    def compute_blockId_offset(self, seq, start, end, device):
        positions = torch.arange(start, end, device=device)
        logical = positions // self.block_size
        offsets = positions % self.block_size
        table = torch.tensor(seq.block_table, device=device)
        block_ids = table[logical]
        return block_ids, offsets

    def scatter(self, seq, layer_idx, k_new, v_new):
        num_new_tokens = k_new.shape[2]
        device = k_new.device
        block_ids, offsets = self.compute_blockId_offset(seq, seq.length-num_new_tokens, seq.length, device)
        k_new_reshaped = k_new.squeeze(0).transpose(0, 1)
        v_new_reshaped = v_new.squeeze(0).transpose(0, 1)
        self.k[layer_idx][block_ids, offsets] = k_new_reshaped
        self.v[layer_idx][block_ids, offsets] = v_new_reshaped

    def gather(self, seq, layer_idx):
        device = self.k[layer_idx].device
        block_ids, offsets = self.compute_blockId_offset(seq, 0, seq.length, device)
        k = self.k[layer_idx][block_ids, offsets]
        v = self.v[layer_idx][block_ids, offsets]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        return k, v

@dataclass
class Sequence:
    block_table: list[int] = field(default_factory=list)
    length: int = 0
        

if __name__ == "__main__":
    # Allocator invariants. Deliberately does not import model.py -- the pool
    # only needs three numbers off a config, and keeping this file free of the
    # model (and of transformers) means these checks stay instant.
    from types import SimpleNamespace

    # Qwen2.5-0.5B-Instruct's shape, so the footprint printed below is real.
    config = SimpleNamespace(
        num_hidden_layers=24,
        num_key_value_heads=2,
        head_dim=64,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = KVCache.preallocate(config, device)
    n = cache.num_blocks

    # Shape: block index leads, so a block is one contiguous slab of slots.
    assert len(cache.k) == len(cache.v) == config.num_hidden_layers
    expected = (n, cache.block_size, config.num_key_value_heads, config.head_dim)
    assert tuple(cache.k[0].shape) == expected, cache.k[0].shape
    assert cache.k[0].dtype == torch.bfloat16
    assert len(cache.free_blocks) == n

    per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * 2
    total = sum(t.numel() * t.element_size() for t in cache.k + cache.v)
    print(f"  pool          {n} blocks x {cache.block_size} tokens = {n * cache.block_size} tokens")
    print(f"  per token     {per_token / 1024:.1f} KB")
    print(f"  total         {total / 1e6:.1f} MB")

    # Drain the pool. Every id must be distinct -- the failure this catches is
    # an allocator that hands out the same block twice, which corrupts KV
    # silently rather than raising.
    ids = [cache.allocate() for _ in range(n)]
    assert len(set(ids)) == n, f"allocate returned duplicates: {n - len(set(ids))} repeats"
    assert set(ids) == set(range(n)), "allocated ids are not exactly the pool"
    assert not cache.free_blocks

    # Exhaustion raises rather than returning a sentinel: -1 is a *valid* torch
    # index (the last block), so a missed check would write over live KV.
    try:
        cache.allocate()
    except RuntimeError as e:
        print(f"  exhaustion    raised {e!r}")
    else:
        raise AssertionError("allocate() past capacity did not raise")

    # Freeing everything restores the pool exactly.
    for block_id in ids:
        cache.free(block_id)
    assert len(cache.free_blocks) == n, len(cache.free_blocks)
    assert set(cache.free_blocks) == set(range(n))

    # Interleaved alloc/free: freed blocks must become reusable, and the pool
    # size must not drift.
    held = [cache.allocate() for _ in range(10)]
    cache.free(held.pop())
    cache.free(held.pop())
    assert len(cache.free_blocks) == n - 8
    reused = [cache.allocate() for _ in range(2)]
    assert len(set(reused) & set(held)) == 0, "reissued a block that is still held"
    print(f"  interleaved   {len(cache.free_blocks)} free after 8 held")

    print("cache PASS")
