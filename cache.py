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
    reserved_blocks: int = 0
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
        for block_id in seq.block_table[keep:]:
            self.free(block_id)
        seq.block_table = seq.block_table[:keep]
        seq.length = new_length
        seq.token_ids = seq.token_ids[:new_length]

    def incref(self, block_id):
        if block_id in self.free_blocks:
            self.free_blocks.remove(block_id)
        self.refcounts[block_id] += 1

    def block_hash(self, token_ids, prev_hash):
        return hash((prev_hash, tuple(token_ids)))

    def register(self, seq):
        prev_hash = None
        for i in range(seq.length // self.block_size):
            start = i * self.block_size
            end = (i+1) * self.block_size
            chunk = tuple(seq.token_ids[start:end])
            h = self.block_hash(chunk, prev_hash)
            if h not in self.registry:
                self.registry[h] = (seq.block_table[i], chunk)
                self.block_hashes[seq.block_table[i]] = h
            prev_hash = h

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
        return len(matched)

    def ensure_capacity(self, seq, n_new_tokens):
        block_table = seq.block_table
        blocks_needed = ceil((seq.length + n_new_tokens) / self.block_size)
        while len(block_table) < blocks_needed:
            block_id = self.allocate()
            block_table.append(block_id)
            if seq.reserved_blocks>0:
                self.reserved_blocks -= 1
                seq.reserved_blocks -= 1

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
    reserved_blocks: int = 0
    block_table: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)

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
    # Refcounts. Every failure in this layer is silent by construction --
    # the counter is a plain int and no value looks illegal -- so rather
    # than checking outcomes these assert the invariant that ties the two
    # structures together: a block is in free_blocks IFF its refcount is 0.
    # Anything that breaks sharing breaks that.
    # ------------------------------------------------------------------
    def check_invariant(cache, note):
        free = set(cache.free_blocks)
        assert len(free) == len(cache.free_blocks), f"{note}: duplicate ids in free_blocks"
        for block_id in range(cache.num_blocks):
            held = cache.refcounts[block_id] > 0
            assert (block_id in free) != held, (
                f"{note}: block {block_id} has refcount {cache.refcounts[block_id]} "
                f"but is {'in' if block_id in free else 'not in'} free_blocks"
            )

    cache = KVCache.preallocate(config, device)
    check_invariant(cache, "fresh pool")
    assert all(rc == 0 for rc in cache.refcounts)

    block = cache.allocate()
    assert cache.refcounts[block] == 1, "allocate did not establish a reference"
    check_invariant(cache, "after allocate")

    # Three owners, released one at a time. The block must stay out of the
    # pool until the last one lets go -- returning it early is precisely
    # what hands a live shared prefix to the next request to be overwritten.
    cache.incref(block)
    cache.incref(block)
    assert cache.refcounts[block] == 3
    for remaining in (2, 1):
        cache.free(block)
        assert cache.refcounts[block] == remaining
        assert block not in cache.free_blocks, "shared block returned to the pool early"
        check_invariant(cache, "shared block partially released")
    cache.free(block)
    assert cache.refcounts[block] == 0 and block in cache.free_blocks
    check_invariant(cache, "last reference released")
    print(f"  refcounts     survives 3 owners, pooled once at zero")

    # Adopting a *cached* block: refcount 0, but its KV is still valid and
    # the registry may hand it out on a prefix hit. incref has to pull it
    # off the free list, or the allocator can reissue it to someone else
    # while its adopter is reading through it.
    before = len(cache.free_blocks)
    cache.incref(block)
    assert cache.refcounts[block] == 1
    assert block not in cache.free_blocks, "adopted a cached block but left it allocatable"
    assert len(cache.free_blocks) == before - 1
    check_invariant(cache, "adopted off the free list")
    handed_out = [cache.allocate() for _ in range(len(cache.free_blocks))]
    assert block not in handed_out, "allocator reissued an adopted block"
    print(f"  adopt         cached block pulled out of the free list")

    # free() on a block nobody holds means the caller is confused about
    # ownership. Absorbing it silently would leave that same caller's more
    # dangerous bug -- over-releasing a *shared* block, which looks like a
    # legal decrement -- with no symptom whatsoever.
    fresh = KVCache.preallocate(config, device)
    try:
        fresh.free(0)
    except RuntimeError as e:
        print(f"  unheld free   raised {e!r}")
    else:
        raise AssertionError("free() on an unheld block did not raise")

    # FIFO recycling. Under prefix caching a freed block still holds valid
    # KV and may be adopted later, so recycling the most-recently-freed
    # block first would evict exactly the entry most likely to be hit again.
    fresh = KVCache.preallocate(config, device)
    drained = [fresh.allocate() for _ in range(fresh.num_blocks)]
    order = [drained[5], drained[9], drained[2]]
    for block_id in order:
        fresh.free(block_id)
    assert [fresh.allocate() for _ in range(3)] == order, (
        "allocator is not FIFO -- freed blocks came back in the wrong order"
    )
    print(f"  fifo          oldest freed block recycled first")

    # ------------------------------------------------------------------
    # Prefix registry. The failure this layer produces is a *dangling
    # entry*: the registry still maps a hash to a block whose KV has since
    # been overwritten by different tokens. Whoever adopts it then attends
    # to an unrelated prefix -- wrong text, no error, not reproducible.
    # So these check the two structures against each other rather than
    # checking that lookups happen to work.
    # ------------------------------------------------------------------
    def check_registry(cache, note):
        for h, (block_id, chunk) in cache.registry.items():
            assert cache.block_hashes[block_id] == h, (
                f"{note}: registry[{h}] -> block {block_id}, but that block's "
                f"back-pointer is {cache.block_hashes[block_id]}"
            )
            assert len(chunk) == cache.block_size, f"{note}: registered a partial chunk"
        for block_id, h in enumerate(cache.block_hashes):
            if h is None:
                continue
            assert h in cache.registry, (
                f"{note}: block {block_id} claims hash {h}, which the registry has dropped"
            )
            assert cache.registry[h][0] == block_id, (
                f"{note}: block {block_id} claims hash {h}, but that entry points at "
                f"block {cache.registry[h][0]}"
            )

    def make_seq(cache, token_ids):
        """A sequence whose blocks are allocated and 'filled' with these tokens."""
        seq = Sequence(token_ids=list(token_ids))
        cache.ensure_capacity(seq, len(token_ids))
        seq.length = len(token_ids)
        return seq

    cache = KVCache.preallocate(config, device)
    check_registry(cache, "fresh pool")

    # 40 tokens = two full blocks plus an 8-token tail. Only the full ones
    # may be published: the tail block's remaining slots are about to be
    # written, so anyone adopting it would read uninitialised KV.
    prompt = list(range(1000, 1040))
    a = make_seq(cache, prompt)
    cache.register(a)
    assert len(a.block_table) == 3, a.block_table
    assert len(cache.registry) == 2, f"expected 2 full blocks registered, got {len(cache.registry)}"
    assert cache.block_hashes[a.block_table[2]] is None, "published the partial tail block"
    check_registry(cache, "after register")
    print(f"  register      {len(cache.registry)} full blocks published, tail withheld")

    # The chain must be position-sensitive. The same 16 tokens appearing as
    # block 0 of one prompt and block 1 of another are NOT interchangeable --
    # each token's KV depends on everything before it -- so their hashes
    # must differ. A per-chunk hash that ignored history would collide here
    # and hand out KV computed at the wrong positions.
    shifted = make_seq(cache, list(range(2000, 2016)) + prompt[:16])
    cache.register(shifted)
    assert len(cache.registry) == 4, "position-independent hash: chunk reused across positions"
    check_registry(cache, "after shifted register")
    print(f"  chained hash  same chunk at a different offset hashes differently")

    # Duplicate prefill: another sequence computes the same prefix into its
    # own blocks before anyone registered. The first entry must win, and the
    # loser's blocks must NOT claim the hash -- otherwise recycling them
    # would evict a live entry belonging to somebody else's block.
    owners = {h: b for h, (b, _) in cache.registry.items()}
    dup = make_seq(cache, prompt)
    cache.register(dup)
    assert {h: b for h, (b, _) in cache.registry.items()} == owners, "duplicate overwrote the entry"
    assert all(cache.block_hashes[b] is None for b in dup.block_table), (
        "losing sequence's blocks claimed a hash they do not own"
    )
    check_registry(cache, "after duplicate register")
    print(f"  duplicate     first registration wins, loser claims nothing")

    # A registered block outlives its last user: refcount 0, still in the
    # registry, still adoptable. That is the whole point of the design --
    # a request arriving after the first one finished should still hit.
    for block_id in a.block_table:
        cache.free(block_id)
    assert cache.refcounts[a.block_table[0]] == 0
    assert a.block_table[0] in cache.free_blocks
    assert cache.block_hashes[a.block_table[0]] is not None, "dropped the entry on free"
    check_invariant(cache, "registered block freed")
    check_registry(cache, "registered block freed")
    print(f"  cached free   entry outlives its last owner")

    # ...until memory pressure recycles it. Draining the pool must take every
    # registry entry with it. A block that comes back out of allocate() is
    # about to be overwritten, so any entry still pointing at it is a lie.
    #
    # Everything has to be released first: a block still held by a live
    # sequence is not in free_blocks, so allocate() can never reach it and
    # its entry is *supposed* to survive.
    for seq in (shifted, dup):
        for block_id in seq.block_table:
            cache.free(block_id)
    while cache.free_blocks:
        cache.allocate()
    assert cache.registry == {}, f"{len(cache.registry)} entries survived a full drain"
    assert all(h is None for h in cache.block_hashes)
    check_registry(cache, "after full drain")
    print(f"  eviction      recycling a block deletes its entry")

    # ------------------------------------------------------------------
    # match_prefix. Two things have to hold at once: the adopted ids must be
    # the *same physical blocks* (a copy would be correct but pointless), and
    # every adopted block must become unreachable to the allocator. The
    # second is the one that corrupts memory when it is missed, and
    # check_invariant cannot see it -- a refcount-0 block sitting in
    # free_blocks is self-consistent even while a block table points at it.
    # So check_ownership walks the live sequences instead.
    # ------------------------------------------------------------------
    def check_ownership(cache, seqs, note):
        for tag, seq in seqs:
            for block_id in seq.block_table:
                assert cache.refcounts[block_id] > 0, (
                    f"{note}: {tag} holds block {block_id} at refcount 0"
                )
                assert block_id not in cache.free_blocks, (
                    f"{note}: {tag} holds block {block_id}, but the allocator can still hand it out"
                )

    cache = KVCache.preallocate(config, device)
    system = list(range(500, 540))          # a 40-token "system prompt"

    cold = Sequence(token_ids=system + [1, 2, 3])
    assert cache.match_prefix(cold) == 0, "cold cache matched something"
    assert cold.block_table == [] and cold.length == 0

    # First request runs and publishes its full blocks.
    a = make_seq(cache, system + [1, 2, 3])
    cache.register(a)

    # Second request shares the 40-token prefix. It must adopt A's physical
    # blocks -- pointing at the same ids is the entire mechanism.
    b = Sequence(token_ids=system + [7, 8, 9, 10, 11])
    n_shared = cache.match_prefix(b)
    assert n_shared == 2, f"expected the 2 full shared blocks, matched {n_shared}"
    assert b.block_table == a.block_table[:2], "adopted copies rather than the same blocks"
    assert b.length == 32, b.length
    assert all(cache.refcounts[x] == 2 for x in b.block_table), "adopted without increfing"
    check_invariant(cache, "after match")
    check_registry(cache, "after match")
    check_ownership(cache, [("a", a), ("b", b)], "after match")

    cache.ensure_capacity(b, len(b.token_ids) - b.length)
    distinct = len(set(a.block_table) | set(b.block_table))
    assert distinct == len(a.block_table) + len(b.block_table) - n_shared
    print(f"  match         {n_shared} blocks shared, "
          f"{distinct} distinct vs {len(a.block_table) + len(b.block_table)} unshared")

    # Divergence: the walk stops at the first block that differs, and cannot
    # resume. Chaining makes that automatic -- every later hash is built on
    # the one that missed -- but it is the property sharing depends on.
    diverged = Sequence(token_ids=system[:16] + [999] * 24)
    assert cache.match_prefix(diverged) == 1, "match did not stop at the first differing block"
    check_ownership(cache, [("diverged", diverged)], "after divergent match")
    print(f"  divergence    stops at the first differing block")

    # An exact repeat of a block-aligned prompt must still leave a token to
    # prefill: the logits for sampling come off the last position, so a
    # 100% hit with nothing to compute cannot produce an output token.
    aligned = list(range(700, 732))         # exactly 2 blocks
    first = make_seq(cache, aligned)
    cache.register(first)
    repeat = Sequence(token_ids=list(aligned))
    matched = cache.match_prefix(repeat)
    assert repeat.length < len(repeat.token_ids), (
        f"matched all {repeat.length} tokens -- prefill would get an empty tensor"
    )
    print(f"  full repeat   matched {matched}/2 blocks, {len(aligned) - repeat.length} tokens left to prefill")

    # Hash collision. Forced by rewriting an entry's tokens behind its hash:
    # without the stored-chunk comparison this adopts a block holding
    # completely unrelated KV, and the output is wrong with no error anywhere.
    victim = Sequence(token_ids=system + [1, 2, 3])
    h0 = cache.block_hash(tuple(system[:16]), None)
    collided_block, _ = cache.registry[h0]
    cache.registry[h0] = (collided_block, tuple(range(9000, 9016)))
    assert cache.match_prefix(victim) == 0, "adopted a block whose tokens do not match the hash"
    print(f"  collision     mismatched tokens treated as a miss, not a hit")

    # ------------------------------------------------------------------
    # Scatter/gather round-trip. A wrong index here does not crash -- it
    # silently returns another sequence's KV -- so the check is exact
    # equality against what went in. bf16 in, bf16 out, no arithmetic in
    # between, so torch.equal is the right comparison rather than allclose.
    # ------------------------------------------------------------------
    n_kv, head_dim = config.num_key_value_heads, config.head_dim
    layer = 0

    cache = KVCache.preallocate(config, device)
    # preallocate uses torch.empty, so the "layer 1 was never written" check
    # at the bottom would otherwise be reading uninitialised memory and
    # passing on luck.
    for t in cache.k + cache.v:
        t.zero_()

    # Fragment the pool first. Contiguous block tables would pass even if
    # gather ignored the table and read a flat slice -- exactly the bug
    # paging exists to make impossible.
    #
    # Drain the pool completely before handing blocks back, so free_blocks
    # holds *only* scattered ids. Taking 24 and freeing every other one used
    # to work, but it leaned on allocate() popping from the end: once the
    # allocator went FIFO those holes sat behind a hundred untouched blocks
    # and the tables came out consecutive again.
    scratch = [cache.allocate() for _ in range(n)]
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
