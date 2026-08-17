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

    # ------------------------------------------------------------------
    # Scatter/gather round-trip. A wrong index here does not crash -- it
    # silently returns another sequence's KV -- so the check is exact
    # equality against what went in. bf16 in, bf16 out, no arithmetic in
    # between, so torch.equal is the right comparison rather than allclose.
    # ------------------------------------------------------------------
    n_kv, head_dim = config.num_key_value_heads, config.head_dim
    layer = 0

    cache = KVCache.preallocate(config, device)

    # Fragment the pool first. Contiguous block tables would pass even if
    # gather ignored the table and read a flat slice -- exactly the bug
    # paging exists to make impossible.
    scratch = [cache.allocate() for _ in range(24)]
    for block_id in scratch[::2]:
        cache.free(block_id)

    # 5 spans one block, 40 spans three, 17 spans two with a 1-token tail.
    lengths = [5, 40, 17]
    seqs, ref_k, ref_v = [], [], []
    for length in lengths:
        seq = Sequence()
        cache.ensure_capacity(seq, length)
        k = torch.randn(1, n_kv, length, head_dim, device=device).bfloat16()
        v = torch.randn(1, n_kv, length, head_dim, device=device).bfloat16()
        seq.length = length
        cache.scatter_prefill(seq, layer, k, v)
        seqs.append(seq)
        ref_k.append(k)
        ref_v.append(v)

    assert any(
        any(b - a != 1 for a, b in zip(s.block_table, s.block_table[1:]))
        for s in seqs
    ), "block tables came out contiguous -- fragmentation did not take"
    print(f"  block tables  {[s.block_table for s in seqs]}")

    # Prefill round-trip, one sequence at a time.
    for seq, rk, rv in zip(seqs, ref_k, ref_v):
        gk, gv = cache.gather_prefill(seq, layer)
        assert gk.shape == rk.shape, (gk.shape, rk.shape)
        assert torch.equal(gk, rk), "gather_prefill returned different K than scattered"
        assert torch.equal(gv, rv), "gather_prefill returned different V than scattered"
    print(f"  prefill       round-trip exact for lengths {lengths}")

    # One decode step: every sequence appends exactly one token.
    k_dec = torch.randn(len(seqs), n_kv, 1, head_dim, device=device).bfloat16()
    v_dec = torch.randn(len(seqs), n_kv, 1, head_dim, device=device).bfloat16()
    for seq in seqs:
        cache.ensure_capacity(seq, 1)   # before the bump: capacity is for the new token
        seq.length += 1
    cache.scatter_decode(seqs, layer, k_dec, v_dec)

    gk, gv, mask = cache.gather_decode(seqs, layer)
    max_len = max(s.length for s in seqs)
    assert gk.shape == (len(seqs), n_kv, max_len, head_dim), gk.shape
    assert mask.shape == (len(seqs), 1, 1, max_len), mask.shape
    assert mask.dtype == torch.bool

    for i, seq in enumerate(seqs):
        # The decode token must land immediately after the prefilled ones.
        expected_k = torch.cat([ref_k[i], k_dec[i : i + 1]], dim=2)[0]
        expected_v = torch.cat([ref_v[i], v_dec[i : i + 1]], dim=2)[0]
        assert torch.equal(gk[i, :, : seq.length], expected_k), f"seq {i} K mismatch"
        assert torch.equal(gv[i, :, : seq.length], expected_v), f"seq {i} V mismatch"
        # Padding must stay zero. If it held stale KV, short sequences would
        # attend to garbage the moment the mask was wrong.
        assert gk[i, :, seq.length :].eq(0).all(), f"seq {i} padding is not zero"
        assert mask[i, 0, 0, : seq.length].all(), f"seq {i} real positions masked out"
        assert not mask[i, 0, 0, seq.length :].any(), f"seq {i} padding left unmasked"
    print(f"  decode        round-trip exact, padded to {max_len}, mask agrees")

    # Layer isolation: layer 0 was written, layer 1 never was. If the layer
    # index leaked into an index expression, this is where it shows up.
    gk1, _, _ = cache.gather_decode(seqs, 1)
    assert gk1.eq(0).all(), "writing layer 0 disturbed layer 1"
    print(f"  layers        independent")

    print("cache PASS")
