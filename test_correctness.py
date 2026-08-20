"""Correctness gate for the engine. Run it after every change.

Checks run cheapest-relevant-first, so the most fundamental thing fails first.
`python test_correctness.py --record` re-freezes the Stage 2 reference
generations -- only when output is meant to change.

Tolerances. bf16 through 24 layers accumulates real error: HF's own bf16
logits differ from HF's fp32 by ~1.9 max / ~0.10 mean here, so a 1e-2 bound is
not reachable by anyone. The reference is HF in fp32, and we assert we land in
the same noise band HF's bf16 path does.

Most checks compare logits, not tokens. Token equality is only the right bar
at batch 1 on an unchanged code path -- batching and prefix hits change the
reduction order, which legitimately moves the low bits.

Argmax is checked but not demanded everywhere: near-tied top-2 logits flip
under bf16 noise for good reasons. We assert only that no *decisive* position
-- one where fp32 has a clear winner -- disagrees.
"""

import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from math import ceil

import model
import spec
from cache import Sequence
from scheduler import Request, RequestState, Scheduler

PROMPT = "Explain what a KV cache is in one sentence."

# Too small to hold the workload, so the pool is a real constraint. With all
# 128 blocks every request is admitted in step 1 and this is static batching.
# Admission is optimistic, so the pool goes further than it looks: 9 blocks is
# where the queue still has to wait and decode still has to preempt.
SCHEDULER_POOL_BLOCKS = 9

REFERENCE_PATH = Path(__file__).parent / "reference_generations.json"

# Recorded at Stage 2, replayed by every later stage. Varied lengths on
# purpose, including one that hits the token limit instead of EOS.
REFERENCE_PROMPTS = [
    ("What is 2+2?", 40),
    ("Explain what a KV cache is in one sentence.", 80),
    ("Write a haiku about GPUs.", 60),
    ("List three prime numbers.", 40),
    ("Describe the difference between a CPU and a GPU.", 32),
]

# Same arithmetic, different reduction order -- not bitwise identical, but
# much tighter than the HF comparison. One bf16 ulp here is ~0.25.
MAX_DECODE_ABS_DIFF = 0.5

# Measured against HF-fp32: ours ~0.10 mean, HF-bf16 ~0.10 mean. 0.25 leaves
# headroom for prompt variation without being vacuous.
MAX_MEAN_ABS_DIFF = 0.25

# The 1.5B runs 28 layers, so error has four more layers to build up in and
# the 0.5B bound does not transfer. Measured, not guessed:
#
#   0.5B draft    mean 0.095   max 2.40
#   1.5B target   mean 0.128   max 3.32
#
# Nothing asserts on max -- it is the noisier statistic.
MAX_TARGET_MEAN_ABS_DIFF = 0.25

# Lengths must differ, or gather_decode pads nothing and the mask goes
# untested. The spread matters too: the longest sequence has no padding, so a
# dropped mask leaves it bit-identical.
BATCH_PROMPTS = [
    "Hi.",
    "What is 2+2?",
    "Explain what a KV cache is in one sentence.",
    "Describe in detail the difference between a CPU and a GPU, "
    "covering memory hierarchy, parallelism, and typical workloads.",
]
BATCH_DECODE_STEPS = 12

# Padding to the batch max changes the reduction order again, so this is
# looser than the batch-1 bound. Calibrated by injecting the two bugs it
# exists to catch:
#
#   correct                    max abs diff  0.25 - 0.84
#   padding mask dropped                     5.50 - 14.09
#   RoPE positions off by one                8.97 - 15.89
MAX_BATCH_ABS_DIFF = 2.0

# A position is "decisive" if fp32's top-2 gap exceeds this. One bf16 ulp is
# ~0.25, so 1.0 is well outside what rounding could flip.
DECISIVE_MARGIN = 1.0


def tokenize(snapshot_path, prompt, device):
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def load_engine(snapshot_path, device):
    """Load once and share across checks -- 1 GB of weights per check adds up."""
    config = model.load_config(snapshot_path)
    weights = model.load_weights(snapshot_path)
    cos, sin = model.build_sincos_table(config, device)
    return config, weights, cos, sin


def fresh_pool(config, device):
    """A pool per check, so a leak in one cannot starve the next."""
    return model.KVCache.preallocate(config, device)


def our_logits(engine, token_ids):
    config, weights, cos, sin = engine
    # Fresh pool and empty sequence => length 0, so this is a full prefill.
    kvcache = fresh_pool(config, token_ids.device)
    seq = model.Sequence()
    logits = model.forward_prefill(token_ids, weights, config, cos, sin, kvcache, [seq])
    return logits.float()


def hf_logits(snapshot_path, token_ids, dtype=torch.float32):
    """Reference logits from HuggingFace. Frees the model before returning --
    8 GB does not hold both implementations.
    """
    hf = AutoModelForCausalLM.from_pretrained(
        snapshot_path, dtype=dtype, attn_implementation="sdpa"
    ).to(token_ids.device).eval()
    with torch.no_grad():
        logits = hf(token_ids).logits.float()
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    return logits


def compare(ours, reference):
    diff = (ours - reference).abs()
    agree = ours.argmax(-1) == reference.argmax(-1)

    # Positions where the reference has a clear winner, so rounding alone
    # should never change the argmax.
    top2 = reference.topk(2, dim=-1).values
    decisive = (top2[..., 0] - top2[..., 1]) > DECISIVE_MARGIN

    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "argmax_agreement": agree.float().mean().item(),
        "decisive_mismatches": int((decisive & ~agree).sum().item()),
        "n_decisive": int(decisive.sum().item()),
    }


def check_forward_pass(snapshot_path, engine, device="cuda"):
    token_ids = tokenize(snapshot_path, PROMPT, device)
    ours = our_logits(engine, token_ids)
    reference = hf_logits(snapshot_path, token_ids)

    assert ours.shape == reference.shape, f"shape {ours.shape} != {reference.shape}"

    stats = compare(ours, reference)
    print(f"  tokens                {token_ids.shape[1]}")
    print(f"  max abs diff          {stats['max_abs_diff']:.4f}")
    print(f"  mean abs diff         {stats['mean_abs_diff']:.5f}")
    print(f"  argmax agreement      {stats['argmax_agreement']:.4f}")
    print(f"  decisive mismatches   {stats['decisive_mismatches']} / {stats['n_decisive']}")

    assert stats["decisive_mismatches"] == 0, (
        f"{stats['decisive_mismatches']} decisive positions disagree with fp32 "
        "-- this is a structural bug, not bf16 noise"
    )
    assert stats["mean_abs_diff"] < MAX_MEAN_ABS_DIFF, (
        f"mean abs diff {stats['mean_abs_diff']:.5f} exceeds {MAX_MEAN_ABS_DIFF}"
    )
    return stats


def check_incremental_decode(snapshot_path, engine, device="cuda"):
    """Prefill n-1 tokens then decode 1, vs prefilling all n.

    Covers the decode path: RoPE at a non-zero offset, the mask branch, and
    reading K/V back out of the cache. Batch 1, so a wrong-rank positions
    tensor and a missing pad mask both still broadcast -- check_batched_decode
    covers those.
    """
    config, weights, cos, sin = engine
    token_ids = tokenize(snapshot_path, PROMPT, device)
    n = token_ids.shape[1]

    pool_full = fresh_pool(config, device)
    seq_full = model.Sequence()
    full = model.forward_prefill(
        token_ids, weights, config, cos, sin, pool_full, [seq_full]
    )[:, -1].float()

    pool_split = fresh_pool(config, device)
    seq_split = model.Sequence()
    model.forward_prefill(
        token_ids[:, :-1], weights, config, cos, sin, pool_split, [seq_split]
    )
    split = model.forward_decode(
        token_ids[:, -1:], weights, config, cos, sin, pool_split, [seq_split]
    )[:, -1].float()

    max_abs_diff = (full - split).abs().max().item()
    same_argmax = full.argmax(-1).item() == split.argmax(-1).item()

    # Same length AND same block count. Equal lengths with unequal blocks
    # means ensure_capacity over-allocates -- a leak the logits never show.
    blocks = (len(seq_full.block_table), len(seq_split.block_table))
    expected_blocks = -(-n // pool_full.block_size)

    print(f"  tokens                {n}")
    print(f"  length                {seq_full.length} / {seq_split.length}")
    print(f"  blocks                {blocks[0]} / {blocks[1]} (expect {expected_blocks})")
    print(f"  max abs diff          {max_abs_diff:.4f}")
    print(f"  argmax agrees         {same_argmax}")

    assert seq_full.length == n, f"single-pass sequence at {seq_full.length}, expected {n}"
    assert seq_split.length == n, (
        f"split sequence at {seq_split.length}, expected {n} -- "
        "length is not advancing by the number of tokens written"
    )
    assert blocks == (expected_blocks, expected_blocks), (
        f"block tables {blocks}, expected {expected_blocks} each -- "
        "ensure_capacity is allocating the wrong number of blocks"
    )
    assert same_argmax, (
        "prefill+decode predicts a different token than a single prefill -- "
        "the decode path is structurally wrong, not just noisy"
    )
    assert max_abs_diff < MAX_DECODE_ABS_DIFF, (
        f"max abs diff {max_abs_diff:.4f} exceeds {MAX_DECODE_ABS_DIFF}"
    )
    return {"max_abs_diff": max_abs_diff}


def check_batched_decode(snapshot_path, engine, device="cuda"):
    """Decode sequences of differing lengths together, vs one at a time.

    Where a missing pad mask or off-by-one RoPE positions show up. Both paths
    are fed the same tokens, so one divergence cannot cascade into twelve.

    Injecting those two bugs moves max abs diff to 5.5-15.9, against 0.25-0.84
    clean -- which is where MAX_BATCH_ABS_DIFF comes from.
    """
    config, weights, cos, sin = engine
    n = len(BATCH_PROMPTS)

    prompt_ids = [tokenize(snapshot_path, p, device) for p in BATCH_PROMPTS]
    prompt_lens = [ids.shape[1] for ids in prompt_ids]
    assert len(set(prompt_lens)) > 1, (
        f"batch prompt lengths {prompt_lens} are all equal -- gather_decode "
        "would pad nothing and the mask path would go untested"
    )

    # Reference: each sequence alone, in its own pool, at batch size 1.
    ref_tokens, ref_logits = [], []
    for ids in prompt_ids:
        pool = fresh_pool(config, device)
        seq = model.Sequence()
        logits = model.forward_prefill(
            ids, weights, config, cos, sin, pool, [seq]
        )[:, -1]
        tokens, steps = [], []
        for _ in range(BATCH_DECODE_STEPS):
            token = logits.argmax(-1, keepdim=True)
            tokens.append(int(token.item()))
            logits = model.forward_decode(
                token, weights, config, cos, sin, pool, [seq]
            )[:, -1]
            steps.append(logits.float())
        ref_tokens.append(tokens)
        ref_logits.append(steps)

    # Batched: one shared pool. Prefill stays one-at-a-time -- that is the
    # point of paging, a new sequence goes into free blocks without touching
    # the running batch's cache.
    pool = fresh_pool(config, device)
    free_before = len(pool.free_blocks)
    seqs = [model.Sequence() for _ in range(n)]
    for ids, seq in zip(prompt_ids, seqs):
        model.forward_prefill(ids, weights, config, cos, sin, pool, [seq])

    assert len(set(s.length for s in seqs)) > 1, (
        f"sequence lengths {[s.length for s in seqs]} converged after prefill"
    )

    batch_logits = []
    for step in range(BATCH_DECODE_STEPS):
        token_ids = torch.tensor(
            [[ref_tokens[i][step]] for i in range(n)], device=device
        )
        logits = model.forward_decode(
            token_ids, weights, config, cos, sin, pool, seqs
        )[:, -1]
        batch_logits.append(logits.float())

    ours = torch.stack([
        torch.stack([batch_logits[t][i] for t in range(BATCH_DECODE_STEPS)])
        for i in range(n)
    ])
    reference = torch.stack([
        torch.stack([ref_logits[i][t][0] for t in range(BATCH_DECODE_STEPS)])
        for i in range(n)
    ])

    print(f"  prompt lengths        {prompt_lens} (pad to {max(prompt_lens)})")
    print(f"  batch                 {n} sequences x {BATCH_DECODE_STEPS} steps")

    failures = []
    for i in range(n):
        stats = compare(ours[i], reference[i])
        ok = (
            stats["max_abs_diff"] <= MAX_BATCH_ABS_DIFF
            and stats["decisive_mismatches"] == 0
        )
        print(
            f"  seq {i} len {seqs[i].length:>3}         "
            f"max {stats['max_abs_diff']:.4f}  "
            f"argmax {stats['argmax_agreement']:.4f}  "
            f"decisive {stats['decisive_mismatches']}/{stats['n_decisive']}"
            f"{'' if ok else '   FAIL'}"
        )
        if not ok:
            failures.append(
                f"seq {i} ({BATCH_PROMPTS[i]!r}, prompt {prompt_lens[i]} tokens): "
                f"max abs diff {stats['max_abs_diff']:.4f} "
                f"(limit {MAX_BATCH_ABS_DIFF}), "
                f"{stats['decisive_mismatches']} decisive mismatches"
            )

    # Blocks must come back whole. Stage 4's eviction path depends on it, and
    # a leak here is invisible in the logits but fatal after enough requests.
    for seq in seqs:
        for block_id in seq.block_table:
            pool.free(block_id)
    free_after = len(pool.free_blocks)
    assert free_after == free_before, (
        f"pool leaked {free_before - free_after} blocks across the batch"
    )

    assert not failures, (
        "batched decode disagrees with single-sequence decode:\n  "
        + "\n  ".join(failures)
    )
    return {"prompt_lens": prompt_lens}


def _build_scheduler(engine, device, n_blocks=None):
    config, weights, cos, sin = engine
    kvcache = fresh_pool(config, device)
    if n_blocks is not None:
        # No size parameter on preallocate, so shrink the pool by withholding
        # free blocks. The tensors stay full size, which costs nothing.
        kvcache.free_blocks = kvcache.free_blocks[:n_blocks]
    return kvcache, Scheduler(kvcache, config, weights, cos, sin)


def _submit(scheduler, tokenizer, prompt, max_new_tokens, request_id, system=None):
    messages = [{"role": "user", "content": prompt}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    request = Request(
        Sequence(), max_new_tokens, request_id,
        prompt_ids=tokenizer(text).input_ids,
    )
    scheduler.add_request(request)
    return request


def check_scheduler_matches_generate(snapshot_path, engine, device="cuda"):
    """One request at a time through the scheduler must equal Stage 2 exactly.

    Batch 1 throughout, so tokens are the right bar here. Separates "the
    scheduler drives the engine correctly" from "batching moves the numerics".
    """
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    records = json.loads(REFERENCE_PATH.read_text())
    failures = []

    for i, record in enumerate(records):
        _, scheduler = _build_scheduler(engine, device)
        _submit(scheduler, tokenizer, record["prompt"], record["max_new_tokens"], i)
        scheduler.run()

        actual = scheduler.finished[0].output_ids
        if actual != record["token_ids"]:
            failures.append(
                f"{record['prompt']!r}: {len(actual)} tokens, "
                f"expected {len(record['token_ids'])}"
            )

    print(f"  batch 1               {len(records) - len(failures)}/{len(records)} match Stage 2")
    assert not failures, (
        "scheduler at batch 1 diverges from generate():\n  " + "\n  ".join(failures)
    )


# Long enough to span several blocks, so the shared region is more than the
# chat preamble. Matching only block 0 would look like it works and save
# almost nothing.
SHARED_SYSTEM = (
    "You are a precise and careful technical assistant. Answer in complete "
    "sentences. When a question is ambiguous, state the interpretation you "
    "are using before answering. Prefer concrete examples over abstractions, "
    "and never speculate beyond what you are certain of."
)

SHARED_QUESTIONS = [
    "What is a page table?",
    "Why does GQA reduce KV cache size?",
    "Name two causes of memory fragmentation.",
    "What is a page table?",          # exact repeat: must still leave a token to prefill
]


def check_prefix_caching(snapshot_path, engine, device="cuda"):
    """Stage 5 exit: same answers, fewer blocks.

    Compares logits, not tokens. A prefix hit swaps the fused causal kernel
    for an explicit mask, which moves the low bits.
    """
    config, weights, cos, sin = engine
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)

    prompts = []
    for question in SHARED_QUESTIONS:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SHARED_SYSTEM},
             {"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(tokenizer(text).input_ids)

    def run(share):
        """Prefill every prompt into one pool, with sharing on or off."""
        kvcache = fresh_pool(config, device)
        free_before = len(kvcache.free_blocks)
        logits, matched = [], []
        for token_ids in prompts:
            seq = model.Sequence(token_ids=list(token_ids))
            matched.append(kvcache.match_prefix(seq) if share else 0)
            tail = torch.tensor([token_ids[seq.length:]], device=device)
            out = model.forward_prefill(tail, weights, config, cos, sin, kvcache, [seq])
            if share:
                kvcache.register(seq)
            logits.append(out[:, -1].float())
            assert seq.length == len(token_ids), (
                f"sequence ended at {seq.length} tokens, prompt was {len(token_ids)}"
            )
        return logits, matched, free_before - len(kvcache.free_blocks)

    cold_logits, _, cold_blocks = run(share=False)
    warm_logits, matched, warm_blocks = run(share=True)

    shared_len = 0
    while all(len(p) > shared_len and p[shared_len] == prompts[0][shared_len]
              for p in prompts[:3]):
        shared_len += 1
    print(f"  prompts               {[len(p) for p in prompts]} tokens, "
          f"{shared_len} shared")
    print(f"  blocks allocated      {warm_blocks} with sharing, {cold_blocks} without")
    print(f"  blocks matched        {matched}")

    # The first request cannot hit -- nothing is registered yet. Every later
    # one must, or the registry is being populated but never read.
    assert matched[0] == 0, f"first request matched {matched[0]} blocks from an empty registry"
    assert all(m > 0 for m in matched[1:]), f"later requests got no cache hits: {matched}"

    # The saving is the point of the stage.
    assert warm_blocks < cold_blocks, (
        f"sharing allocated {warm_blocks} blocks, no better than {cold_blocks} without"
    )

    # A repeated prompt must still leave a token to prefill, or there are no
    # logits to sample the first output token from.
    assert matched[3] * model.KVCache.block_size < len(prompts[3]), (
        "an exact repeat consumed the whole prompt -- prefill would get an empty tensor"
    )

    # Answers must not move. A wrong mask on the warm path shows up here as
    # decisive mismatches: the new tokens would attend to a truncated prefix.
    for i, (cold, warm) in enumerate(zip(cold_logits, warm_logits)):
        stats = compare(warm, cold)
        print(f"  request {i} +{matched[i]} blocks   max {stats['max_abs_diff']:.4f}  "
              f"argmax {stats['argmax_agreement']:.4f}  "
              f"decisive {stats['decisive_mismatches']}/{stats['n_decisive']}")
        assert stats["decisive_mismatches"] == 0, (
            f"request {i}: {stats['decisive_mismatches']} decisive logit flips after a "
            f"prefix hit -- the warm-path attention mask is wrong"
        )
        assert stats["argmax_agreement"] == 1.0, (
            f"request {i}: argmax moved after a prefix hit"
        )


PREFIX_POOL_BLOCKS = 14


def check_prefix_caching_scheduler(snapshot_path, engine, device="cuda"):
    """Sharing under concurrency: several live sequences on the same blocks.

    The bug this exists for is retire() handing a block back while another
    running request is still reading through it. No error, just wrong text.
    """
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    budgets = [24, 40, 16, 30]

    def run(share):
        kvcache, scheduler = _build_scheduler(engine, device, PREFIX_POOL_BLOCKS)
        if not share:
            # Publish nothing, so match_prefix can never hit. Same code path
            # otherwise, which keeps the block-count comparison honest.
            kvcache.register = lambda seq: None
        free_before = len(kvcache.free_blocks)
        requests = [
            _submit(scheduler, tokenizer, question, budget, i, system=SHARED_SYSTEM)
            for i, (question, budget) in enumerate(zip(SHARED_QUESTIONS, budgets))
        ]

        peak_used, peak_refcount, batch_sizes = 0, 0, []
        steps = 0
        while scheduler.waiting_queue or scheduler.running:
            scheduler.schedule()
            steps += 1
            assert steps < 2000, "scheduler failed to drain -- admission is deadlocked"
            batch_sizes.append(len(scheduler.running))
            peak_used = max(peak_used, free_before - len(kvcache.free_blocks))
            live = [b for r in scheduler.running for b in r.seq.block_table]
            if live:
                peak_refcount = max(peak_refcount, max(kvcache.refcounts[b] for b in live))
            # Nothing a running request holds may be handed out again.
            assert not (set(live) & set(kvcache.free_blocks)), (
                "a block held by a running request is back in the free list"
            )

        assert len(kvcache.free_blocks) == free_before, (
            f"pool leaked {free_before - len(kvcache.free_blocks)} blocks "
            f"({'sharing' if share else 'no sharing'})"
        )
        return requests, peak_used, peak_refcount, max(batch_sizes)

    cold_reqs, cold_used, _, _ = run(share=False)
    warm_reqs, warm_used, peak_refcount, peak_batch = run(share=True)

    print(f"  pool                  {PREFIX_POOL_BLOCKS} blocks, peak batch {peak_batch}")
    print(f"  peak blocks in use    {warm_used} with sharing, {cold_used} without")
    print(f"  peak refcount         {peak_refcount}")
    print(f"  output lengths        {[len(r.output_ids) for r in warm_reqs]}")

    # A refcount above 1 is the only proof two *running* sequences shared a
    # block. Without it this is sequential reuse, already covered above.
    assert peak_refcount > 1, (
        "no block was ever held by two running requests at once -- sharing "
        "under concurrency went untested"
    )
    assert peak_batch > 1, "requests never overlapped"
    assert warm_used < cold_used, (
        f"sharing peaked at {warm_used} blocks, no better than {cold_used} without"
    )

    eos = engine[0].eos_token_id
    for r in warm_reqs:
        assert r.state is RequestState.DONE, f"request {r.request_id} never finished"
        assert len(r.output_ids) <= r.max_new_tokens
        assert r.output_ids[-1] == eos or len(r.output_ids) == r.max_new_tokens, (
            f"request {r.request_id} stopped early without EOS"
        )

    # Same workload, fewer blocks. Tokens are not compared (batching moves
    # the low bits), but corruption would show up as a request stopping for no
    # legitimate reason, which the loop above rules out.
    assert len(warm_reqs) == len(cold_reqs)


def check_scheduler_concurrent(snapshot_path, engine, device="cuda"):
    """Many requests, varied lengths, pool too small to hold them all.

    Asserts what holds regardless of numerics: everything terminates for a
    legitimate reason, nothing overruns its budget, every block comes back.

    Also asserts preemption fired. Admission is optimistic -- a request is let
    in on the blocks it needs now, not for its whole life -- so the pool can
    run dry mid-flight. If nothing was preempted the pool was never tight and
    this degrades into static batching.
    """
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    kvcache, scheduler = _build_scheduler(engine, device, SCHEDULER_POOL_BLOCKS)
    free_before = len(kvcache.free_blocks)

    requests = [
        _submit(scheduler, tokenizer, prompt, max_new_tokens, i)
        for i, (prompt, max_new_tokens) in enumerate(REFERENCE_PROMPTS)
    ]
    prompt_lens = [len(r.prompt_ids) for r in requests]

    # Drive the loop by hand so the batch size each step is observable --
    # the only evidence requests actually overlapped. Preemptions are counted
    # by wrapping the method, so this needs nothing extra from the engine.
    n_preempted = 0
    inner = scheduler.preempt

    def counting_preempt(req):
        nonlocal n_preempted
        n_preempted += 1
        return inner(req)

    scheduler.preempt = counting_preempt

    batch_sizes = []
    while scheduler.waiting_queue or scheduler.running:
        scheduler.schedule()
        batch_sizes.append(len(scheduler.running))

    eos = engine[0].eos_token_id
    lengths = [len(r.output_ids) for r in requests]

    print(f"  requests              {len(requests)}, prompts {prompt_lens}")
    print(f"  pool                  {SCHEDULER_POOL_BLOCKS} blocks, "
          f"{len(kvcache.free_blocks)}/{free_before} free at end")
    print(f"  peak batch            {max(batch_sizes)} over {len(batch_sizes)} steps")
    print(f"  preemptions           {n_preempted}")
    print(f"  output lengths        {lengths}")

    assert len(scheduler.finished) == len(requests), (
        f"{len(scheduler.finished)}/{len(requests)} requests finished"
    )
    assert all(r.state is RequestState.DONE for r in requests), (
        "a finished request was never marked DONE"
    )
    assert len(kvcache.free_blocks) == free_before, (
        f"pool leaked {free_before - len(kvcache.free_blocks)} blocks"
    )

    # Concurrency actually happened, and the pool actually constrained it.
    # Without both, this check silently degrades into the batch-1 case.
    assert max(batch_sizes) > 1, "no two requests were ever running together"
    assert batch_sizes[0] < len(requests), (
        f"all {len(requests)} requests were admitted in step 1 -- pool of "
        f"{SCHEDULER_POOL_BLOCKS} blocks was not a constraint, so "
        "admit-after-evict went untested"
    )
    assert n_preempted > 0, (
        f"no request was ever preempted with only {SCHEDULER_POOL_BLOCKS} "
        "blocks -- optimistic admission never overcommitted, so the "
        "preempt-and-recompute path went untested"
    )

    # Every request stopped for a reason, and none overran. A request that
    # neither hit EOS nor its budget means eviction fired on something else.
    for r in requests:
        assert len(r.output_ids) <= r.max_new_tokens, (
            f"request {r.request_id} produced {len(r.output_ids)} tokens, "
            f"budget was {r.max_new_tokens}"
        )
        assert r.output_ids[-1] == eos or len(r.output_ids) == r.max_new_tokens, (
            f"request {r.request_id} stopped at {len(r.output_ids)} tokens "
            f"without EOS and under its {r.max_new_tokens} budget"
        )

    # Varied output lengths are what exercise eviction at different steps --
    # CLAUDE.md's Stage 4 exit criterion calls this out specifically.
    assert len(set(lengths)) > 1, f"all outputs came out the same length: {lengths}"
    return {"batch_sizes": batch_sizes, "lengths": lengths}


# The rebuild takes a different path than the original write, so the two
# disagree in the low bits. Bounded relative to the values in that layer.
PREEMPT_KV_RTOL = 0.15


def check_preemption(snapshot_path, engine, device="cuda"):
    """A preempted request must come back as if nothing had happened.

    Three things can go wrong, and this separates them: the cache is rebuilt
    from the wrong tokens, a token is emitted twice, or blocks leak.

    The rebuilt K/V is compared with a tolerance, not for equality -- the
    original was written one token at a time by forward_decode, the rebuild in
    one forward_prefill. Different reduction order, so the low bits differ.
    Those differences can flip a near-tied argmax later, so only tokens
    generated *before* the preemption are required to match.
    """
    config = engine[0]
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    records = json.loads(REFERENCE_PATH.read_text())
    layers = (0, config.num_hidden_layers // 2, config.num_hidden_layers - 1)
    failures = []

    def run(record, request_id, n_calls, preempt_at=None):
        kvcache, scheduler = _build_scheduler(engine, device)
        free_before = len(kvcache.free_blocks)
        _submit(scheduler, tokenizer, record["prompt"], record["max_new_tokens"],
                request_id)
        n_preempted = 0
        for step in range(1, n_calls + 1):
            if not scheduler.waiting_queue and not scheduler.running:
                break
            scheduler.schedule()
            if step == preempt_at and scheduler.running:
                scheduler.preempt(scheduler.running[-1])
                n_preempted += 1
        return kvcache, scheduler, free_before, n_preempted

    for i, record in enumerate(records):
        expected = record["token_ids"]
        prompt = record["prompt"]

        for at_step in (1, 2, 5):
            # Both runs must still be mid-flight at the comparison point,
            # otherwise retire() has already cleared the state being compared.
            if at_step + 2 >= len(expected):
                continue

            cache_a, sched_a, _, _ = run(record, i, at_step + 1)
            cache_b, sched_b, _, n_pre = run(record, i, at_step + 1,
                                             preempt_at=at_step)
            if not sched_a.running or not sched_b.running:
                continue
            assert n_pre == 1, (
                f"{prompt!r}: preempt did not fire at step {at_step} -- "
                "this case proves nothing"
            )

            req_a, req_b = sched_a.running[0], sched_b.running[0]

            # The invariant the rebuild rests on: the cache holds every token
            # fed to the model, and the newest output token is not fed yet.
            want = list(req_b.prompt_ids) + list(req_b.output_ids[:-1])
            if req_b.seq.token_ids != want or req_b.seq.length != len(want):
                failures.append(
                    f"{prompt!r} @{at_step}: readmitted cache holds "
                    f"{len(req_b.seq.token_ids)} tokens (length "
                    f"{req_b.seq.length}), expected {len(want)}"
                )
                continue

            if req_a.output_ids != req_b.output_ids:
                n = min(len(req_a.output_ids), len(req_b.output_ids))
                at = next((j for j in range(n)
                           if req_a.output_ids[j] != req_b.output_ids[j]), n)
                failures.append(
                    f"{prompt!r} @{at_step}: output diverged at token {at} "
                    f"({len(req_a.output_ids)} vs {len(req_b.output_ids)} tokens)"
                )
                continue

            worst = 0.0
            for layer in layers:
                k_a, v_a = cache_a.gather_prefill(req_a.seq, layer)
                k_b, v_b = cache_b.gather_prefill(req_b.seq, layer)
                for x, y in ((k_a, k_b), (v_a, v_b)):
                    scale = x.float().abs().max().item()
                    diff = (x.float() - y.float()).abs().max().item()
                    worst = max(worst, diff / scale if scale else 0.0)
            if worst > PREEMPT_KV_RTOL:
                failures.append(
                    f"{prompt!r} @{at_step}: rebuilt cache differs by "
                    f"{worst:.1%} of value scale (limit {PREEMPT_KV_RTOL:.0%})"
                )
                continue

            print(f"  ok    preempt@{at_step}  {req_b.seq.length} tokens rebuilt  "
                  f"kv within {worst:.1%}  {prompt!r}")

        # Run to completion so retire() runs on a request that was preempted
        # earlier, and the pool has to come back whole.
        if len(expected) < 6:
            continue
        cache, sched, free_before, n_pre = run(record, i, 10_000, preempt_at=3)
        assert n_pre == 1, f"{prompt!r}: preempt did not fire on the full run"
        actual = sched.finished[0].output_ids

        if len(cache.free_blocks) != free_before:
            failures.append(
                f"{prompt!r}: pool leaked {free_before - len(cache.free_blocks)} "
                "blocks across a preemption"
            )
        # Tokens produced before the preemption cannot be affected by it.
        if actual[:4] != expected[:4]:
            failures.append(
                f"{prompt!r}: tokens before the preemption changed -- "
                f"{actual[:4]} vs {expected[:4]}"
            )

        tail = "matches Stage 2" if actual == expected else (
            "diverges at token "
            + str(next((j for j, (a, b) in enumerate(zip(actual, expected))
                        if a != b), min(len(actual), len(expected))))
            + " (tie-flip, allowed)"
        )
        print(f"  ok    full run       pool {len(cache.free_blocks)}/{free_before} "
              f"free, {len(actual)} tokens, {tail}")

    assert not failures, (
        "preemption does not preserve the request:\n  " + "\n  ".join(failures)
    )


def generate_reference(snapshot_path, engine, tokenizer, prompt, max_new_tokens,
                       device, kvcache=None):
    """One generation, asserting the pool comes back whole."""
    config, weights, cos, sin = engine
    if kvcache is None:
        kvcache = fresh_pool(config, device)
    free_before = len(kvcache.free_blocks)

    token_ids = model.generate(
        prompt, max_new_tokens, tokenizer, weights, config, cos, sin, kvcache
    )

    free_after = len(kvcache.free_blocks)
    assert free_after == free_before, (
        f"pool leaked {free_before - free_after} blocks generating {prompt!r} "
        "-- generate() is not freeing its sequence's block table"
    )
    return token_ids


def record_references(snapshot_path, engine, device="cuda"):
    """Freeze current output as the reference later stages replay."""
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    config = engine[0]
    kvcache = fresh_pool(config, device)  # shared, so any leak accumulates
    records = []
    for prompt, max_new_tokens in REFERENCE_PROMPTS:
        token_ids = generate_reference(
            snapshot_path, engine, tokenizer, prompt, max_new_tokens, device, kvcache
        )
        records.append({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "token_ids": token_ids,
            "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        })
        print(f"  recorded {len(token_ids):>3} tokens  {prompt!r}")

    REFERENCE_PATH.write_text(json.dumps(records, indent=2) + "\n")
    print(f"  wrote {REFERENCE_PATH.name}")
    return records


def check_reference_generations(snapshot_path, engine, device="cuda"):
    """Replay the recorded Stage 2 generations. Exact token equality.

    Batch 1 on an unchanged code path, so any divergence is a real behaviour
    change. Do not reuse this bar for batched output.
    """
    if not REFERENCE_PATH.exists():
        raise SystemExit(
            f"{REFERENCE_PATH.name} not found -- run "
            "`python test_correctness.py --record` to create it"
        )

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    config = engine[0]
    kvcache = fresh_pool(config, device)  # shared, so any leak accumulates
    records = json.loads(REFERENCE_PATH.read_text())
    failures = []

    for record in records:
        actual = generate_reference(
            snapshot_path, engine, tokenizer, record["prompt"],
            record["max_new_tokens"], device, kvcache,
        )
        expected = record["token_ids"]
        if actual == expected:
            print(f"  ok   {len(actual):>3} tokens  {record['prompt']!r}")
            continue

        # Report where it diverged -- far more useful than "not equal".
        first = next(
            (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
            min(len(actual), len(expected)),
        )
        failures.append(
            f"{record['prompt']!r}: diverged at token {first} "
            f"(expected {expected[first:first + 3]}, got {actual[first:first + 3]}); "
            f"expected {len(expected)} tokens, got {len(actual)}\n"
            f"      expected text: {record['text']!r}\n"
            f"      actual text:   {tokenizer.decode(actual, skip_special_tokens=True)!r}"
        )
        print(f"  FAIL      {record['prompt']!r}")

    assert not failures, "reference generations changed:\n  " + "\n  ".join(failures)
    return records


def check_truncate():
    """Rollback, over every (length, new_length) pair.

    Asserts against the pool rather than the output. A leaked block only shows
    up as the pool slowly running dry; a block freed while still in a block
    table shows up as another request's text. Neither is visible in logits.

    No model and no CUDA -- the allocator only needs three numbers off a
    config, so every pair is cheap enough to just enumerate.
    """
    from types import SimpleNamespace
    from cache import KVCache

    config = SimpleNamespace(num_hidden_layers=2, num_key_value_heads=2, head_dim=8)
    B = KVCache.block_size

    def fresh():
        return KVCache.preallocate(config, "cpu")

    checked = 0
    for length in range(0, 4 * B + 3):
        for new_length in range(0, length + 1):
            cache = fresh()
            seq = Sequence(token_ids=list(range(length)))
            cache.ensure_capacity(seq, length)
            seq.length = length
            before = len(cache.free_blocks)

            cache.truncate(seq, new_length)

            # The table must hold exactly the blocks the surviving tokens need.
            want_blocks = -(-new_length // B)
            assert len(seq.block_table) == want_blocks, (
                f"{length}->{new_length}: block_table has {len(seq.block_table)} "
                f"blocks, expected {want_blocks}"
            )
            assert seq.length == new_length
            assert seq.token_ids == list(range(new_length)), (
                f"{length}->{new_length}: token_ids not truncated with length"
            )

            # Blocks in the table must be held; dropped blocks must be back
            # in the pool. In the table AND allocatable is the corrupting
            # case; in neither is a leak.
            for block_id in seq.block_table:
                assert cache.refcounts[block_id] > 0, (
                    f"{length}->{new_length}: block {block_id} is in the table "
                    "at refcount 0"
                )
                assert block_id not in cache.free_blocks, (
                    f"{length}->{new_length}: block {block_id} is in the table "
                    "AND allocatable -- the allocator can hand it to another sequence"
                )
            freed = -(-length // B) - want_blocks
            assert len(cache.free_blocks) == before + freed, (
                f"{length}->{new_length}: pool has {len(cache.free_blocks)} free, "
                f"expected {before + freed} -- leaked or double-freed"
            )

            # Truncating to 0 must return everything.
            cache.truncate(seq, 0)
            assert seq.block_table == [] and seq.token_ids == []
            assert len(cache.free_blocks) == cache.num_blocks, (
                f"{length}->{new_length}->0: {cache.num_blocks - len(cache.free_blocks)} "
                "blocks never came back"
            )
            checked += 1

    # Shared blocks. Truncating a sequence that adopted a prefix must not
    # release blocks someone else still points at. Refcounts handle that, but
    # only if truncate goes through free().
    cache = fresh()
    prompt = list(range(1000, 1000 + 3 * B))
    a = Sequence(token_ids=list(prompt))
    cache.ensure_capacity(a, len(prompt))
    a.length = len(prompt)
    cache.register(a)

    b = Sequence(token_ids=prompt + [7, 8, 9])
    shared = cache.match_prefix(b)
    assert shared >= 2, f"expected a prefix hit to test against, matched {shared}"
    cache.ensure_capacity(b, len(b.token_ids) - b.length)
    b.length = len(b.token_ids)

    held = list(a.block_table)
    cache.truncate(b, 0)
    for block_id in held:
        assert cache.refcounts[block_id] > 0, (
            f"block {block_id} was released by truncating the *other* sequence"
        )
        assert block_id not in cache.free_blocks, (
            f"block {block_id} is still held by sequence a but is allocatable"
        )
    print(f"  pairs                 {checked} (length, new_length) combinations")
    print(f"  shared blocks         survive a sharer's truncation")


def check_acceptance_walk():
    """The accept/reject walk, with hand-built logits.

    Synthetic logits mean every cut point can be hit deliberately, including
    ones a real draft would produce rarely. The property asserted:

        emitted == the target's own predictions, up to and including the cut

    That holds for any proposal whatsoever, which is why the draft can never
    steer the output -- it is only ever a speedup.

    Run at temperature 0, where the stochastic rule collapses exactly onto
    greedy verification and the assertions can stay exact. The stochastic path
    proper is check_speculative_distribution.
    """
    VOCAB = 32

    def logits_for(preds):
        """Logits whose argmax at position i is preds[i]."""
        out = torch.full((1, len(preds), VOCAB), -1.0)
        for i, tok in enumerate(preds):
            out[0, i, tok] = 1.0
        return out

    def q_for(proposal):
        """A greedy draft's distributions: one-hot on each proposed token."""
        out = torch.zeros(len(proposal), VOCAB)
        for i, tok in enumerate(proposal):
            out[i, tok] = 1.0
        return out

    # Temperature 0 collapses the stochastic rule onto the greedy one, so every
    # assertion below is exact rather than distributional. See the docstring.
    T0 = torch.tensor([[0.0]])

    cases = 0
    for k in range(1, 7):
        preds = [(3 * i + 5) % VOCAB for i in range(k + 1)]
        logits = logits_for(preds)

        for cut in range(k + 1):
            # A proposal that agrees with the target up to `cut` and differs
            # there. cut == k means every proposal was right.
            proposal = list(preds[:cut])
            if cut < k:
                proposal.append((preds[cut] + 1) % VOCAB)
                # Tail beyond the cut is arbitrary -- it must not influence
                # anything, so make it deliberately wrong.
                proposal += [(preds[i] + 7) % VOCAB for i in range(cut + 1, k)]
            assert len(proposal) == k

            emitted, n_accepted = spec.accept_proposal(logits, proposal, q_for(proposal), T0, 1.0)

            assert n_accepted == cut, (
                f"k={k} cut={cut}: n_accepted {n_accepted}, expected {cut}"
            )
            assert emitted == preds[:cut + 1], (
                f"k={k} cut={cut}: emitted {emitted}, expected {preds[:cut + 1]} "
                "-- the output is not the target's own prediction prefix"
            )
            # Always one token past the accepted prefix -- which is why
            # progress is guaranteed. At zero acceptance a round still yields
            # one token, the non-speculative baseline.
            assert len(emitted) == n_accepted + 1, (
                f"k={k} cut={cut}: emitted {len(emitted)} tokens for "
                f"{n_accepted} accepted"
            )
            assert emitted[:n_accepted] == proposal[:n_accepted], (
                f"k={k} cut={cut}: accepted tokens are not the proposed ones"
            )
            assert all(isinstance(t, int) for t in emitted), (
                f"k={k} cut={cut}: emitted {[type(t).__name__ for t in emitted]} "
                "-- tensors leaking into a token list"
            )
            cases += 1

    # Random proposals, including ones that re-agree after diverging. Only
    # the FIRST mismatch may matter -- a walk that resumed after one would
    # pass every case above and fail here.
    rng = torch.Generator().manual_seed(0)
    k = 6
    preds = [(3 * i + 5) % VOCAB for i in range(k + 1)]
    logits = logits_for(preds)
    for _ in range(500):
        proposal = torch.randint(0, VOCAB, (k,), generator=rng).tolist()
        emitted, n_accepted = spec.accept_proposal(logits, proposal, q_for(proposal), T0, 1.0)

        expected_cut = k
        for i in range(k):
            if proposal[i] != preds[i]:
                expected_cut = i
                break
        assert n_accepted == expected_cut, (
            f"random proposal {proposal}: cut at {n_accepted}, expected "
            f"{expected_cut} -- the walk must stop at the first mismatch"
        )
        assert emitted == preds[:expected_cut + 1], (
            f"random proposal {proposal}: emitted {emitted}, "
            f"expected {preds[:expected_cut + 1]}"
        )
        cases += 1

    print(f"  cut points            {cases} cases, k=1..6, every position")
    print(f"  output                always the target's own prediction prefix")
    print(f"  progress              always n_accepted + 1 tokens")


def check_draft_loop(snapshot_path, engine, device="cuda"):
    """The draft loop alone must reproduce Stage 2 exactly.

    Same model, same prompt, batch 1, so it has no licence to differ. Run
    before any target is involved: a draft that mispredicts on its own looks
    exactly like a draft the target happens to disagree with.

    Also checks seq.token_ids records the token whose K/V was written, not the
    one predicted. Backwards, the tokens still come out right and the cache's
    idea of its own contents is shifted by one.
    """
    config, weights, cos, sin = engine
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    records = json.loads(REFERENCE_PATH.read_text())
    kvcache = fresh_pool(config, device)   # shared, so any leak accumulates
    failures = []

    for record in records:
        expected = record["token_ids"]
        # Prefill gives one token, so k proposals cover k+1 reference tokens.
        # Clamped so a short reference cannot push the draft past its EOS.
        k = min(8, len(expected) - 1)
        if k < 1:
            continue

        prompt_ids = tokenize(snapshot_path, record["prompt"], device)
        seq = Sequence(token_ids=prompt_ids[0].tolist())
        free_before = len(kvcache.free_blocks)

        seed = int(model.prefill(
            prompt_ids, weights, config, cos, sin, kvcache, [seq]
        ).argmax(-1))
        n_prompt = seq.length

        # Temperature 0 makes the draft greedy, so it must still reproduce the
        # recorded Stage 2 generations exactly.
        proposed, q = spec.draft_tokens(
            weights, config, cos, sin, kvcache, seq, seed, k,
            torch.tensor([[0.0]], device=device), 1.0
        )

        # One row per proposed token, not per requested k -- the loop breaks
        # early on EOS and accept_proposal indexes q alongside the proposal.
        assert q.shape == (len(proposed), config.vocab_size), (
            f"{record['prompt']!r}: q is {tuple(q.shape)}, expected "
            f"{(len(proposed), config.vocab_size)}"
        )
        assert torch.allclose(q.sum(-1), torch.ones(len(proposed), device=q.device)), (
            f"{record['prompt']!r}: draft distributions are not normalised"
        )

        # One token fed per iteration, starting with the seed -- so this is
        # what the cache should hold, and the last proposal is not in it.
        fed = ([seed] + proposed)[:len(proposed)]
        assert seq.length == n_prompt + len(proposed), (
            f"{record['prompt']!r}: seq.length {seq.length}, expected "
            f"{n_prompt + len(proposed)} after {len(proposed)} draft steps"
        )
        assert len(seq.token_ids) == seq.length, (
            f"{record['prompt']!r}: token_ids holds {len(seq.token_ids)} tokens "
            f"but seq.length is {seq.length} -- the cache and its token list "
            "disagree about how much is valid"
        )
        assert seq.token_ids[n_prompt:] == fed, (
            f"{record['prompt']!r}: cache holds {seq.token_ids[n_prompt:]}, "
            f"expected {fed} -- token_ids is recording the predicted token "
            "rather than the one whose K/V was written"
        )

        for block_id in seq.block_table:
            kvcache.free(block_id)
        assert len(kvcache.free_blocks) == free_before, (
            f"{record['prompt']!r}: pool leaked "
            f"{free_before - len(kvcache.free_blocks)} blocks"
        )

        actual = [seed] + proposed
        if actual == expected[:len(actual)]:
            print(f"  ok    k={k}  {len(actual)} tokens  {record['prompt']!r}")
            continue

        first = next(i for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
        failures.append(
            f"{record['prompt']!r}: diverged at token {first} "
            f"(expected {expected[first:first + 3]}, got {actual[first:first + 3]})"
        )
        print(f"  FAIL  k={k}  {record['prompt']!r}")

    assert not failures, "draft loop does not match Stage 2:\n  " + "\n  ".join(failures)


def check_verify_window(snapshot_path, engine, device="cuda"):
    """The target's forward pass over a speculative window, no draft involved.

    Feeds it a window that is already the greedy continuation, taken from the
    references, so the right answer is known: every position must predict the
    next reference token. That is the all-accepted case.

    Then corrupts one proposal and checks the damage stays at and after it.
    That confinement is what makes accepting the prefix before a mismatch
    sound; if the mask leaked, every acceptance decision would be unsound.
    """
    config, weights, cos, sin = engine
    records = json.loads(REFERENCE_PATH.read_text())
    kvcache = fresh_pool(config, device)   # shared, so any leak accumulates

    def run_window(prompt, seed, proposals):
        prompt_ids = tokenize(snapshot_path, prompt, device)
        seq = Sequence(token_ids=prompt_ids[0].tolist())
        free_before = len(kvcache.free_blocks)
        if seed is None:
            seed = int(model.prefill(
                prompt_ids, weights, config, cos, sin, kvcache, [seq]
            ).argmax(-1))
        else:
            model.prefill(prompt_ids, weights, config, cos, sin, kvcache, [seq])
        n_prompt = seq.length

        logits = spec.target_tokens(
            weights, config, cos, sin, kvcache, seq, seed, proposals
        )
        preds = logits.argmax(-1)[0].tolist()

        state = (seq.length, len(seq.token_ids), list(seq.token_ids[n_prompt:]))
        for block_id in seq.block_table:
            kvcache.free(block_id)
        assert len(kvcache.free_blocks) == free_before, (
            f"{prompt!r}: pool leaked {free_before - len(kvcache.free_blocks)} blocks"
        )
        return seed, preds, n_prompt, state

    for record in records:
        expected = record["token_ids"]
        # The window is k+1 wide and its last position predicts one token
        # beyond, so the reference has to be at least k+2 long.
        k = min(6, len(expected) - 2)
        if k < 1:
            continue
        proposals = expected[1:k + 1]

        seed, preds, n_prompt, (length, n_ids, written) = run_window(
            record["prompt"], None, proposals
        )
        assert seed == expected[0], (
            f"{record['prompt']!r}: prefill produced {seed}, reference starts {expected[0]}"
        )

        # Sequence state: k+1 tokens fed, so k+1 tokens of KV and k+1 ids.
        assert length == n_prompt + k + 1, (
            f"{record['prompt']!r}: seq.length advanced to {length}, expected "
            f"{n_prompt + k + 1} -- the window is not being appended as k+1 new tokens"
        )
        assert n_ids == length, (
            f"{record['prompt']!r}: token_ids holds {n_ids} but seq.length is {length}"
        )
        assert written == [seed] + proposals, (
            f"{record['prompt']!r}: cache holds {written}, expected {[seed] + proposals}"
        )

        # Position i predicts reference token i+1, including the bonus token
        # at the end that nothing proposed.
        want = expected[1:k + 2]
        if preds != want:
            first = next(i for i, (a, b) in enumerate(zip(preds, want)) if a != b)
            raise AssertionError(
                f"{record['prompt']!r}: window prediction diverges at position {first} "
                f"(expected {want[first:first + 3]}, got {preds[first:first + 3]}) -- "
                "one pass over k+1 positions disagrees with k+1 sequential decodes"
            )
        print(f"  ok    k={k}  {k + 1} positions  {record['prompt']!r}")

    # Causality. Corrupt one proposal and confirm the damage is confined to
    # positions at and after it.
    record = max(records, key=lambda r: len(r["token_ids"]))
    expected = record["token_ids"]
    k = min(6, len(expected) - 2)
    proposals = expected[1:k + 1]
    seed, clean, _, _ = run_window(record["prompt"], None, proposals)

    j = k // 2
    corrupt = list(proposals)
    corrupt[j] = (corrupt[j] + 1000) % config.vocab_size
    assert corrupt[j] != proposals[j]
    _, dirty, _, _ = run_window(record["prompt"], seed, corrupt)

    # proposals[j] sits at window position j+1, since the seed occupies 0.
    pos = j + 1
    assert clean[:pos] == dirty[:pos], (
        f"corrupting position {pos} changed predictions before it "
        f"(first differs at {next(i for i, (a, b) in enumerate(zip(clean, dirty)) if a != b)}) "
        "-- attention is not causal within the window, so accepting a prefix "
        "before a mismatch would be unsound"
    )
    assert clean[pos:] != dirty[pos:], (
        f"corrupting position {pos} changed nothing from there on -- the window "
        "tokens are not reaching attention at all"
    )
    print(f"  causal      corrupt at window pos {pos}: "
          f"positions 0-{pos - 1} identical, {pos}-{k} diverge")


def check_target_model(snapshot_path, device="cuda"):
    """Re-run the model-side gates on the 1.5B speculation target.

    Everything above only tested the 0.5B. The target changes head_dim
    (64 -> 128), which every attention reshape, the RoPE table and the cache's
    last axis are built around -- a number derived from the 0.5B rather than
    read from config surfaces here and nowhere else.

    This has to pass before any acceptance rate means anything. Ordering is
    forced by VRAM: our bf16 target is ~3 GB, HF's fp32 reference ~6 GB, and
    8 GB holds either but not both.
    """
    engine = load_engine(snapshot_path, device)
    config = engine[0]
    print(f"  layers                {config.num_hidden_layers}")
    print(f"  hidden / head_dim     {config.hidden_size} / {config.head_dim}")
    print(f"  kv heads              {config.num_key_value_heads} "
          f"(gqa {config.num_attention_heads // config.num_key_value_heads}:1)")

    # Draft ids are fed straight into the target, so the two must share a
    # vocabulary. Asserted rather than taken from the model card.
    draft_config = model.load_config(model.qwen0_5_snapshot_path)
    assert config.vocab_size == draft_config.vocab_size, (
        f"vocab {config.vocab_size} != draft's {draft_config.vocab_size} -- "
        "draft tokens cannot be verified by this target"
    )
    assert config.eos_token_id == draft_config.eos_token_id, (
        f"eos {config.eos_token_id} != draft's {draft_config.eos_token_id}"
    )
    print(f"  vocab / eos           {config.vocab_size} / {config.eos_token_id} "
          "(shared with draft)")

    print("  -- incremental decode")
    check_incremental_decode(snapshot_path, engine, device)
    print("  -- batched decode")
    check_batched_decode(snapshot_path, engine, device)

    print("  -- forward pass vs HuggingFace")
    token_ids = tokenize(snapshot_path, PROMPT, device)
    ours = our_logits(engine, token_ids).cpu()

    # Drop our weights before HF's fp32 copy lands, or this OOMs.
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    reference = hf_logits(snapshot_path, token_ids).cpu()
    assert ours.shape == reference.shape, f"shape {ours.shape} != {reference.shape}"

    stats = compare(ours, reference)
    print(f"  tokens                {token_ids.shape[1]}")
    print(f"  max abs diff          {stats['max_abs_diff']:.4f}")
    print(f"  mean abs diff         {stats['mean_abs_diff']:.5f}")
    print(f"  argmax agreement      {stats['argmax_agreement']:.4f}")
    print(f"  decisive mismatches   {stats['decisive_mismatches']} / {stats['n_decisive']}")

    assert stats["decisive_mismatches"] == 0, (
        f"{stats['decisive_mismatches']} decisive positions disagree with fp32 "
        "-- this is a structural bug, not bf16 noise"
    )
    assert stats["mean_abs_diff"] < MAX_TARGET_MEAN_ABS_DIFF, (
        f"mean abs diff {stats['mean_abs_diff']:.5f} exceeds {MAX_TARGET_MEAN_ABS_DIFF}"
    )

    del reference
    gc.collect()
    torch.cuda.empty_cache()
    return stats


def check_both_models_resident(device="cuda"):
    """Draft and target loaded at once, with a KV pool each.

    check_target_model frees one before loading the other, so it says nothing
    about whether they fit together -- and speculation needs both live for the
    whole run.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    draft = load_engine(model.qwen0_5_snapshot_path, device)
    after_draft = torch.cuda.memory_allocated()
    target = load_engine(model.qwen1_5_snapshot_path, device)
    after_target = torch.cuda.memory_allocated()

    draft_pool = fresh_pool(draft[0], device)
    target_pool = fresh_pool(target[0], device)
    after_pools = torch.cuda.memory_allocated()

    total = torch.cuda.get_device_properties(device).total_memory
    peak = torch.cuda.max_memory_allocated()

    def kv_per_token(config):
        return 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * 2

    mb = 1024 ** 2
    print(f"  draft weights         {(after_draft - base) / mb:.0f} MB")
    print(f"  target weights        {(after_target - after_draft) / mb:.0f} MB")
    print(f"  kv per token          {kv_per_token(draft[0]) / 1024:.0f} KB draft, "
          f"{kv_per_token(target[0]) / 1024:.0f} KB target")
    print(f"  pools                 {(after_pools - after_target) / mb:.0f} MB "
          f"({draft_pool.num_blocks} x {draft_pool.block_size} tokens each)")
    print(f"  resident              {after_pools / mb:.0f} MB of {total / mb:.0f} MB")
    print(f"  headroom              {(total - peak) / mb:.0f} MB for activations")

    # A tripwire, not a tight bound. If a bigger pool or a third model pushes
    # residency past this, speculation starts OOMing mid-run -- miserable to
    # debug through a rollback bug. Fail here instead.
    assert after_pools < 0.75 * total, (
        f"{after_pools / mb:.0f} MB resident of {total / mb:.0f} MB -- "
        "too little left for activations to run speculation comfortably"
    )

    del draft, target, draft_pool, target_pool
    gc.collect()
    torch.cuda.empty_cache()



def check_speculative_distribution():
    """Emitted tokens follow the target's own distribution, whatever the draft did.

    This is what makes speculation a pure latency win rather than a quality
    tradeoff, and it is the one thing check_acceptance_walk cannot see -- that
    runs at temperature 0, where the rule is deterministic.

    Synthetic p and q rather than real models: logits = log(p) makes the
    target's distribution exactly p, so there is a ground truth to compare
    frequencies against. The three pairs stress the rule where it does work --
    a draft that disagrees, one that agrees closely, and one that is
    confidently wrong (q peaked where p is thin).
    """
    V, N, TOL = 8, 40000, 0.01
    T1 = torch.tensor([[1.0]])
    g = torch.Generator().manual_seed(0)

    def norm(x):
        return x / x.sum()

    p_base = norm(torch.rand(V, generator=g))
    pairs = [
        ("draft disagrees", p_base, norm(torch.rand(V, generator=g))),
        ("draft agrees closely", p_base, norm(p_base + 0.02 * torch.rand(V, generator=g))),
        ("draft confidently wrong", p_base,
         norm(torch.full((V,), 0.01) + torch.eye(V)[int(p_base.argmin())])),
    ]

    for name, p, q in pairs:
        for k in (1, 4):
            # Every window position carries the same p, so the marginal at
            # position 0 has a known target no matter where the cut lands.
            logits = p.log().unsqueeze(0).repeat(k + 1, 1).unsqueeze(0)
            counts = torch.zeros(V)
            accepted = 0
            for _ in range(N):
                proposal = [int(model.sample_from_probs(q)) for _ in range(k)]
                emitted, n = spec.accept_proposal(
                    logits, proposal, q.unsqueeze(0).repeat(k, 1), T1, 1.0
                )
                counts[emitted[0]] += 1
                accepted += n
            err = (counts / N - p).abs().max().item()
            rate = accepted / (N * k)
            assert err < TOL, (
                f"{name} k={k}: max frequency error {err:.4f} exceeds {TOL} "
                "-- emitted tokens are not distributed as the target would "
                "have sampled them"
            )
            print(f"  {name:24s} k={k}  max err {err:.4f}  accept rate {rate:.2f}")

    # Negative control: the tempting wrong version skips the ratio test and
    # keeps whatever the draft proposed. That has to show up as skew, or the
    # check above cannot fail.
    p, q = pairs[0][1], pairs[0][2]
    counts = torch.zeros(V)
    for _ in range(N):
        counts[int(model.sample_from_probs(q))] += 1
    naive_err = (counts / N - p).abs().max().item()
    assert naive_err > TOL, (
        "negative control did not skew -- p and q are too similar for this "
        "test to demonstrate anything"
    )
    print(f"  {'always-accept (control)':24s}       max err {naive_err:.4f}  "
          f"-> rejected at {TOL}")



def check_speculative_end_to_end(device="cuda"):
    """speculative_generate vs the target decoding on its own.

    Every piece has been tested in isolation. This is what catches the ways
    they can be wrong together: a rollback that frees one block too few, a
    draft that fails to catch up, an off-by-one in the logical length.

    At temperature 0 speculation is not an approximation of greedy decoding,
    it *is* greedy decoding with fewer target passes -- so exact equality is
    the bar and any divergence is a bug.

    The second assertion is the Stage 6 exit criterion: strictly fewer target
    forwards than tokens. Without it the stage could pass with every proposal
    rejected -- correct output, no speedup.
    """
    tokenizer = AutoTokenizer.from_pretrained(model.qwen1_5_snapshot_path)
    # load_engine returns (config, weights, cos, sin); generate() and
    # speculative_generate() both take weights before config.
    d_config, d_weights, d_cos, d_sin = load_engine(model.qwen0_5_snapshot_path, device)
    t_config, t_weights, t_cos, t_sin = load_engine(model.qwen1_5_snapshot_path, device)

    prompts = [
        "What is 2+2?",
        "Explain what a KV cache is in one sentence.",
        "List three prime numbers.",
    ]
    k, max_new = 4, 48

    # Count target forwards by wrapping the one function that performs them.
    real_target_tokens = spec.target_tokens
    calls = {"n": 0}

    def counting_target_tokens(*args, **kwargs):
        calls["n"] += 1
        return real_target_tokens(*args, **kwargs)

    total_tokens = total_forwards = 0
    try:
        for prompt in prompts:
            baseline_pool = fresh_pool(t_config, device)
            baseline = model.generate(
                prompt, max_new, tokenizer,
                t_weights, t_config, t_cos, t_sin, baseline_pool,
            )
            assert len(baseline_pool.free_blocks) == baseline_pool.num_blocks, (
                f"{prompt!r}: baseline generate() leaked blocks"
            )

            draft_pool = fresh_pool(d_config, device)
            target_pool = fresh_pool(t_config, device)
            calls["n"] = 0
            spec.target_tokens = counting_target_tokens
            out = spec.speculative_generate(
                prompt, max_new, tokenizer,
                d_weights, d_config, d_cos, d_sin,
                t_weights, t_config, t_cos, t_sin,
                draft_pool, target_pool, k,
                temperature=0, top_p=1.0,
            )
            spec.target_tokens = real_target_tokens

            assert out == baseline, (
                f"{prompt!r}: speculative output diverges from the target's own "
                f"greedy decode\n  spec     {out}\n  baseline {baseline}\n"
                "  -- at temperature 0 these must be identical; a mismatch means "
                "the KV rollback or the draft catch-up is corrupting state"
            )
            assert all(isinstance(t, int) for t in out), (
                f"{prompt!r}: tensors leaking into the token list"
            )
            for pool, name in ((draft_pool, "draft"), (target_pool, "target")):
                assert len(pool.free_blocks) == pool.num_blocks, (
                    f"{prompt!r}: {name} pool leaked "
                    f"{pool.num_blocks - len(pool.free_blocks)} blocks"
                )

            # +1 for the prefill that produces the first token before any
            # speculation happens.
            forwards = calls["n"] + 1
            total_tokens += len(out)
            total_forwards += forwards
            print(f"  ok  {len(out):3d} tokens / {forwards:3d} target passes "
                  f"({len(out) / forwards:.2f}x)  {prompt!r}")
    finally:
        spec.target_tokens = real_target_tokens

    assert total_forwards < total_tokens, (
        f"{total_forwards} target forwards for {total_tokens} tokens -- "
        "speculation is not saving any target passes"
    )
    print(f"  overall               {total_tokens} tokens in {total_forwards} "
          f"target passes ({total_tokens / total_forwards:.2f}x), k={k}")



def check_prefix_caching_across_turns(snapshot_path, engine, device):
    """Blocks filled during decode must enter the registry too.

    register() at admission alone caches only the prompt, so a follow-up turn
    matches to the end of the first prompt and then re-prefills the whole
    reply it just produced. Multi-turn chat got no reuse past turn 1.

    Turn 2's ids are built by concatenating token ids rather than re-rendering
    the template: re-tokenising "prompt + reply" can give slightly different
    ids, and the resulting miss would be a tokenizer artefact, not a cache bug.
    """
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    kvcache, scheduler = _build_scheduler(engine, device)

    matches = []
    real_match_prefix = kvcache.match_prefix

    def counting_match_prefix(seq):
        n = real_match_prefix(seq)
        matches.append(n)
        return n

    kvcache.match_prefix = counting_match_prefix

    turn1 = _submit(scheduler, tokenizer, "Name three programming languages.", 48, 0)
    n_prompt1 = len(turn1.prompt_ids)
    scheduler.run()
    reply1 = scheduler.finished[0].output_ids

    followup = tokenizer(
        "<|im_start|>user\nWhich of those is oldest?<|im_end|>\n"
        "<|im_start|>assistant\n"
    ).input_ids
    turn2_ids = list(turn1.prompt_ids) + list(reply1) + list(followup)

    matches.clear()
    warm = Request(Sequence(), 32, 1, prompt_ids=turn2_ids)
    scheduler.add_request(warm)
    scheduler.run()
    matched_blocks = matches[0]
    matched_tokens = matched_blocks * kvcache.block_size
    warm_out = [r for r in scheduler.finished if r.request_id == 1][0].output_ids

    # The whole point: the match must run past where the first turn's prompt
    # ended, into blocks that were filled token-by-token during decode.
    assert matched_tokens > n_prompt1, (
        f"matched {matched_tokens} tokens but turn 1's prompt was {n_prompt1} "
        "-- the match stopped at the prompt boundary, so decode-filled blocks "
        "are not being registered"
    )

    # Control: the same request on an empty registry must match nothing and
    # produce the same tokens. Reuse is only legitimate if it is invisible.
    cold_cache, cold_scheduler = _build_scheduler(engine, device)
    cold_matches = []
    real_cold = cold_cache.match_prefix

    def counting_cold(seq):
        n = real_cold(seq)
        cold_matches.append(n)
        return n

    cold_cache.match_prefix = counting_cold
    cold_scheduler.add_request(Request(Sequence(), 32, 1, prompt_ids=turn2_ids))
    cold_scheduler.run()
    cold_out = cold_scheduler.finished[0].output_ids

    assert cold_matches[0] == 0, (
        f"fresh pool matched {cold_matches[0]} blocks -- the warm result proves "
        "nothing if a cold registry hits too"
    )
    assert warm_out == cold_out, (
        f"cached turn 2 diverges from uncached\n  warm {warm_out}\n"
        f"  cold {cold_out}\n  -- reused blocks hold the wrong K/V"
    )

    print(f"  turn 1                {n_prompt1} prompt + {len(reply1)} generated tokens")
    print(f"  turn 2 matched        {matched_blocks} blocks "
          f"({matched_tokens} tokens), {matched_tokens - n_prompt1} past the "
          "prompt boundary")
    print(f"  cold control          0 blocks matched, identical output")




def check_head_of_line_blocking(snapshot_path, engine, device):
    """Skipping past a request that does not fit must raise the batch size.

    The admission loop passes over a request whose blocks are unavailable and
    tries the ones behind it, so the gap it leaves gets filled.

    The workload has to keep the pool contended or this measures nothing. That
    needs prompts whose answers run long: with short answers every request
    retires within a few tokens, blocks free up immediately, and the blocked
    request gets in under either policy -- both then tie exactly.

    Tokens are not compared across the two policies. The batch composition
    differs, which changes the reduction order, which flips near-ties (section
    6). What is asserted instead is that every request finishes for a
    legitimate reason and none is starved.
    """
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)

    long_prompt = (
        "Consider the following list of hardware components and answer briefly. "
        "A GPU has streaming multiprocessors, registers, shared memory, an L2 "
        "cache, and high bandwidth memory attached over a wide bus. Which of "
        "these is largest in capacity? Explain the memory hierarchy from "
        "fastest to slowest and say which one the KV cache normally lives in "
        "and why."
    )
    # Long answers on purpose -- these must still be running when the long
    # prompt is considered, or there is no contention to schedule around.
    short_prompts = [
        "Describe the water cycle step by step.",
        "Explain what a cache is, in detail.",
        "List ten prime numbers with reasons.",
        "Explain recursion with an example.",
        "Describe how a hard drive works.",
    ]
    BUDGET = 96

    def run(max_skips, n_blocks):
        kvcache, scheduler = _build_scheduler(engine, device, n_blocks=n_blocks)
        scheduler.max_skips = max_skips
        # Order matters: the long prompt must arrive behind some short ones,
        # so the pool is already partly held when it is considered. First in
        # the queue it would take an empty pool and block nobody.
        _submit(scheduler, tokenizer, short_prompts[0], BUDGET, 1)
        _submit(scheduler, tokenizer, short_prompts[1], BUDGET, 2)
        _submit(scheduler, tokenizer, long_prompt, BUDGET, 0)
        for j, prompt in enumerate(short_prompts[2:]):
            _submit(scheduler, tokenizer, prompt, BUDGET, j + 3)
        scheduler.run()
        outputs = {r.request_id: list(r.output_ids) for r in scheduler.finished}
        order = [r.request_id for r in scheduler.finished]
        return kvcache, scheduler, outputs, order

    # 20 blocks: the leading requests hold enough that the long prompt cannot
    # be admitted, while the ones behind it still fit.
    n_blocks = 20
    fifo_kv, fifo, fifo_out, fifo_order = run(0, n_blocks)
    aged_kv, aged, aged_out, aged_order = run(3, n_blocks)

    n_requests = 1 + len(short_prompts)
    eos = engine[0].eos_token_id
    for st, name in ((fifo.stats, "fifo"), (aged.stats, "aged")):
        assert st.retired == n_requests, (
            f"{name}: {st.retired} of {n_requests} requests retired -- the long "
            "request was starved or the queue deadlocked"
        )

    # Reordering admission may not drop or truncate anybody. Tokens are not
    # compared -- see the docstring.
    for outs, name in ((fifo_out, "fifo"), (aged_out, "aged")):
        for rid, out in outs.items():
            assert len(out) <= BUDGET, f"{name}: request {rid} overran its budget"
            assert out[-1] == eos or len(out) == BUDGET, (
                f"{name}: request {rid} stopped at {len(out)} tokens without EOS"
            )

    # With max_skips=0 every skip breaks immediately, so the two counters must
    # coincide -- that is what "strict FIFO" means in terms of these counters.
    assert fifo.stats.skipped == fifo.stats.barriers, (
        f"max_skips=0 should make every skip a barrier, got "
        f"{fifo.stats.skipped} skips and {fifo.stats.barriers} barriers"
    )
    # And the aged run has to actually exercise the path, or this proves nothing.
    assert aged.stats.skipped > aged.stats.barriers, (
        f"aged run never scanned past a blocked request "
        f"({aged.stats.skipped} skips, {aged.stats.barriers} barriers) -- the "
        "pool is not tight enough for this check to mean anything"
    )
    # The point of the change: filling the gap the blocked request left behind
    # means larger batches and fewer steps for the same total work.
    assert aged.stats.mean_batch_size > fifo.stats.mean_batch_size, (
        f"aged mean batch {aged.stats.mean_batch_size:.2f} is not better than "
        f"fifo {fifo.stats.mean_batch_size:.2f} -- scanning past the blocked "
        "request bought nothing"
    )
    assert aged.stats.decode_steps < fifo.stats.decode_steps, (
        f"aged took {aged.stats.decode_steps} steps vs fifo "
        f"{fifo.stats.decode_steps} -- no throughput gain"
    )
    # The long request is the one being passed over, so it is the one at risk.
    # It has to still get in, and the barrier is what guarantees that.
    assert 0 in aged_out, "the long request never completed under aging"
    assert aged.stats.barriers > 0, (
        "the skip cap never fired, so this run does not exercise the "
        "starvation bound at all"
    )

    print(f"  pool                  {n_blocks} blocks, {n_requests} requests")
    print(f"  fifo   (max_skips=0)  {fifo.stats.decode_steps} steps, "
          f"mean batch {fifo.stats.mean_batch_size:.2f}, "
          f"order {fifo_order}")
    print(f"  aged   (max_skips=3)  {aged.stats.decode_steps} steps, "
          f"mean batch {aged.stats.mean_batch_size:.2f}, "
          f"order {aged_order}")
    print(f"  skips                 {fifo.stats.skipped} fifo "
          f"({fifo.stats.barriers} barriers), {aged.stats.skipped} aged "
          f"({aged.stats.barriers} barriers)")
    print(f"  output                all {n_requests} requests ran to EOS or budget")


def check_admission_accounts_for_prefix(snapshot_path, engine, device):
    """can_admit must count the blocks a request will actually allocate.

    The old check was ceil(prompt / block_size) <= free, computed before
    match_prefix ran, so a request that was mostly cache hit was held out as
    though it needed every block -- exactly when the pool was tight, which is
    when caching is worth most.

    A revived block -- matched from the registry while sitting on the free
    list at refcount 0 -- still takes a free-list slot, so it counts as a cost.
    """
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    # Long enough that the shared portion spans many blocks -- the whole point
    # is a request that is mostly cache hit and only slightly new.
    system = " ".join([
        "You are a careful and precise assistant.",
        "Answer briefly and concretely, and prefer examples over abstractions.",
        "Do not speculate about things you were not told.",
        "When you are unsure, say so plainly instead of guessing.",
        "Never invent citations, numbers, or names.",
        "If a question is ambiguous, state the reading you chose.",
        "Keep answers to a few sentences unless asked otherwise.",
    ])

    def build(n_blocks):
        kv, sch = _build_scheduler(engine, device, n_blocks=n_blocks)
        a = _submit(sch, tokenizer, "Name three programming languages.", 64, 0, system=system)
        sch.schedule()                      # admit + prefill A, one decode step
        assert a in sch.running, "request A was not admitted"
        b = _submit(sch, tokenizer, "Name three databases.", 64, 1, system=system)
        b.seq.token_ids = b.prompt_ids + b.output_ids[:-1]
        return kv, sch, a, b

    # Size the pool by measuring: A's blocks plus slack for B's uncached
    # tail. Hardcoded, it would stop testing anything the moment the prompts
    # or the tokenizer changed.
    probe_kv, _, probe_a, probe_b = build(None)
    a_blocks = len(probe_a.seq.block_table)
    total = ceil(len(probe_b.seq.token_ids) / probe_kv.block_size)
    m, _ = probe_kv.probe_prefix(probe_b.seq)
    slack = (total - m) + 1
    del probe_kv, probe_a, probe_b

    kvcache, scheduler, a, b = build(a_blocks + slack)

    total = ceil(len(b.seq.token_ids) / kvcache.block_size)
    m, r = kvcache.probe_prefix(b.seq)
    free = len(kvcache.free_blocks)

    assert m > 0, (
        "request B matched no blocks -- the two prompts do not actually share "
        "a prefix, so this test proves nothing"
    )
    # The scenario the fix exists for: the old check fails, the new one passes.
    assert total > free, (
        f"pool is not under pressure (total {total} <= free {free}) -- widen "
        "the system prompt or shrink the pool"
    )
    assert scheduler.can_admit(b), (
        f"B needs {total - m + r} blocks and {free} are free, but admission "
        "was refused -- can_admit is still counting the cached prefix"
    )

    # ...and it must genuinely fit. An over-permissive check would blow up here
    # rather than in the assertion above.
    scheduler.schedule()
    assert b in scheduler.running, "request B was admitted but did not survive prefill"
    assert all(c >= 0 for c in kvcache.refcounts), "negative refcount after admission"

    print(f"  pool                  {a_blocks + slack} blocks, {free} free when B arrived")
    print(f"  request B             {total} blocks total, {m} matched "
          f"({r} revived) -> {total - m + r} needed")
    print(f"  old check             would reject ({total} > {free})")


def check_probe_matches_match_prefix(snapshot_path, engine, device):
    """probe_prefix and match_prefix must agree, block for block.

    can_admit trusts probe_prefix to predict what match_prefix does a few
    lines later. If they drift, admission approves a request whose blocks are
    not there. The walk is deliberately duplicated; this test pays for that.
    """
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    kvcache, scheduler = _build_scheduler(engine, device)
    system = "You are a helpful assistant that answers in one short sentence."

    prompts = [
        "Name three programming languages.",
        "Name three databases.",
        "What is a KV cache?",
    ]
    for i, prompt in enumerate(prompts):
        _submit(scheduler, tokenizer, prompt, 24, i, system=system)
    scheduler.run()

    checked = revivals = 0
    # Include a prompt never seen, a prompt seen verbatim, and one sharing only
    # the system prefix -- match counts of 0, all, and partial.
    probes = prompts + ["Something entirely unrelated to the above."]
    for prompt in probes:
        for use_system in (True, False):
            seq = Sequence()
            text = tokenizer.apply_chat_template(
                ([{"role": "system", "content": system}] if use_system else [])
                + [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            seq.token_ids = tokenizer(text).input_ids

            m, r = kvcache.probe_prefix(seq)
            before = list(kvcache.refcounts)

            matched = kvcache.match_prefix(seq)
            assert matched == m, (
                f"{prompt!r} (system={use_system}): probe said {m} blocks, "
                f"match_prefix took {matched} -- the two walks have drifted"
            )
            # Every block match_prefix took that was at refcount 0 beforehand is
            # a revival, and that is exactly what probe_prefix counted.
            revived = sum(1 for b in seq.block_table if before[b] == 0)
            assert revived == r, (
                f"{prompt!r} (system={use_system}): probe said {r} revivals, "
                f"match_prefix revived {revived}"
            )
            checked += 1
            revivals += revived

            for b in seq.block_table:
                kvcache.free(b)

    # Those prompts all finished, so their blocks sit on the free list at
    # refcount 0 and matching them is a revival. At zero revivals the check
    # above compares 0 to 0 and can_admit's accounting is untested.
    assert revivals > 0, "no revivals exercised -- the r term is untested"

    print(f"  agreement             {checked} prompts, block counts and "
          f"revival counts identical ({revivals} revivals seen)")



def check_speculative_prefix_reuse(device="cuda"):
    """speculative_generate must reuse cached prefixes, in both caches.

    Draft and target are independent pools with independent registries, so a
    fix that wired up only one would leave half the work on the table and pass
    a test that looked at the other. Output must be identical to a cold run.
    """
    tokenizer = AutoTokenizer.from_pretrained(model.qwen1_5_snapshot_path)
    d_config, d_weights, d_cos, d_sin = load_engine(model.qwen0_5_snapshot_path, device)
    t_config, t_weights, t_cos, t_sin = load_engine(model.qwen1_5_snapshot_path, device)

    system = (
        "You are a careful assistant. Answer briefly and concretely. "
        "Prefer examples over abstractions, and never invent facts. "
        "If a question is ambiguous, state the reading you chose."
    )
    prompts = [system + " Name three programming languages.",
               system + " Name three databases."]
    k, max_new = 4, 32

    def run(draft_pool, target_pool, prompt):
        return spec.speculative_generate(
            prompt, max_new, tokenizer,
            d_weights, d_config, d_cos, d_sin,
            t_weights, t_config, t_cos, t_sin,
            draft_pool, target_pool, k, temperature=0, top_p=1.0,
        )

    def instrument(pool, log):
        real = pool.match_prefix

        def counting(seq):
            n = real(seq)
            log.append(n)
            return n

        pool.match_prefix = counting

    # Warm: one pool pair shared across both prompts.
    draft_pool = fresh_pool(d_config, device)
    target_pool = fresh_pool(t_config, device)
    d_log, t_log = [], []
    instrument(draft_pool, d_log)
    instrument(target_pool, t_log)

    warm_outputs = [run(draft_pool, target_pool, p) for p in prompts]

    # First call sees an empty registry, second must hit the shared prefix.
    assert d_log[0] == 0 and t_log[0] == 0, (
        f"first call matched blocks against an empty registry "
        f"(draft {d_log[0]}, target {t_log[0]})"
    )
    assert d_log[1] > 0, (
        "second call matched no blocks in the draft cache -- the draft path "
        "is still prefilling the shared prefix from scratch"
    )
    assert t_log[1] > 0, (
        "second call matched no blocks in the target cache -- the target path "
        "is still prefilling the shared prefix from scratch"
    )

    # Cold: a fresh pool pair per prompt, so nothing is reused.
    cold_outputs = []
    for prompt in prompts:
        cold_outputs.append(run(fresh_pool(d_config, device),
                                fresh_pool(t_config, device), prompt))

    for prompt, warm, cold in zip(prompts, warm_outputs, cold_outputs):
        assert warm == cold, (
            f"{prompt[-40:]!r}: reusing cached blocks changed the output\n"
            f"  warm {warm}\n  cold {cold}"
        )

    for pool, name in ((draft_pool, "draft"), (target_pool, "target")):
        assert len(pool.free_blocks) == pool.num_blocks, (
            f"{name} pool leaked {pool.num_blocks - len(pool.free_blocks)} blocks"
        )

    print(f"  draft cache           {d_log[0]} then {d_log[1]} blocks matched")
    print(f"  target cache          {t_log[0]} then {t_log[1]} blocks matched")
    print(f"  output                identical to a cold run, both prompts")



def check_instrumentation(snapshot_path, engine, device):
    """The counters must agree with independently derived truths.

    Instrumentation that drifts is worse than none: the numbers look
    authoritative and nothing downstream can tell. So each counter is checked
    against something derived another way -- a wrapped call site, a
    conservation law over the pool, or the returned tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)

    # ---- scheduler ----
    kvcache, scheduler = _build_scheduler(engine, device, n_blocks=10)
    # _build_scheduler constrains the pool by withholding blocks from the free
    # list, so num_blocks is not the baseline -- the starting free count is.
    blocks_available = len(kvcache.free_blocks)
    system = "You are a helpful assistant that answers in one short sentence."
    n_requests = 5
    for i in range(n_requests):
        _submit(scheduler, tokenizer,
                ["What is 2+2?", "Name a colour.", "Name three primes.",
                 "What is a GPU?", "Say hello."][i], 8 + 4 * i, i, system=system)
    scheduler.run()
    st = scheduler.stats

    assert st.retired == len(scheduler.finished) == n_requests, (
        f"retired {st.retired}, finished {len(scheduler.finished)}, "
        f"submitted {n_requests}"
    )
    assert st.admitted >= n_requests, (
        f"admitted {st.admitted} < {n_requests} submitted"
    )
    assert st.admitted == n_requests + st.preempted, (
        f"admitted {st.admitted} != {n_requests} submitted + {st.preempted} "
        "readmissions -- admissions and preemptions disagree"
    )
    # Every request gets its first token from prefill and the rest from decode
    # steps, so this ties tokens_decoded to the actual output.
    produced = sum(len(r.output_ids) for r in scheduler.finished)
    assert produced == n_requests + st.tokens_decoded, (
        f"{produced} tokens produced but {st.tokens_decoded} decoded + "
        f"{n_requests} prefill tokens"
    )

    # Conservation: in use == allocated + revived - freed. This only involves
    # the free list, which the pool cannot fake.
    cs = kvcache.stats
    in_use = blocks_available - len(kvcache.free_blocks)
    assert cs.blocks_allocated + cs.blocks_revived - cs.blocks_freed == in_use, (
        f"pool accounting: {cs.blocks_allocated} allocated + "
        f"{cs.blocks_revived} revived - {cs.blocks_freed} freed != "
        f"{in_use} in use"
    )
    # The real bound, and one the pool cannot fake: it never held more blocks
    # than the free list ever offered.
    assert cs.peak_blocks_in_use <= blocks_available, (
        f"peak {cs.peak_blocks_in_use} exceeds the {blocks_available} blocks "
        "the pool was given"
    )
    assert in_use == 0, f"{in_use} blocks still held after every request retired"
    assert cs.blocks_revived <= cs.blocks_matched, (
        f"{cs.blocks_revived} revived exceeds {cs.blocks_matched} matched"
    )
    print("  scheduler")
    print(_indent(st.report()))
    print("  pool")
    print(_indent(cs.report(kvcache.block_size)))




def check_spec_instrumentation(device="cuda"):
    """SpecStats against a wrapped call site.

    Split out because the 0.5B engine is freed before the target sections run,
    and three sets of weights do not fit on an 8 GB card.
    """
    spec_tokenizer = AutoTokenizer.from_pretrained(model.qwen1_5_snapshot_path)
    d_config, d_weights, d_cos, d_sin = load_engine(model.qwen0_5_snapshot_path, device)
    t_config, t_weights, t_cos, t_sin = load_engine(model.qwen1_5_snapshot_path, device)

    real_target_tokens = spec.target_tokens
    seen = {"n": 0}

    def counting(*args, **kwargs):
        seen["n"] += 1
        return real_target_tokens(*args, **kwargs)

    sp = spec.SpecStats()
    try:
        spec.target_tokens = counting
        out = spec.speculative_generate(
            "Explain what a KV cache is in one sentence.", 40, spec_tokenizer,
            d_weights, d_config, d_cos, d_sin,
            t_weights, t_config, t_cos, t_sin,
            fresh_pool(d_config, device), fresh_pool(t_config, device), 4,
            temperature=0, top_p=1.0, stats=sp,
        )
    finally:
        spec.target_tokens = real_target_tokens

    # +1 for the prefill, which is a target forward pass but not a round.
    assert sp.target_passes == seen["n"] + 1, (
        f"stats says {sp.target_passes} target passes, wrapping the call site "
        f"counted {seen['n']} + 1 prefill"
    )
    assert sp.rounds == seen["n"], (
        f"{sp.rounds} rounds but {seen['n']} target_tokens calls"
    )
    assert sp.tokens_emitted == len(out), (
        f"stats says {sp.tokens_emitted} tokens, caller got {len(out)}"
    )
    assert sp.tokens_accepted <= sp.tokens_proposed, (
        f"accepted {sp.tokens_accepted} > proposed {sp.tokens_proposed}"
    )
    assert 0.0 <= sp.acceptance_rate <= 1.0
    # Speculation is only worth reporting if it beat one-token-per-pass.
    assert sp.tokens_emitted > sp.target_passes, (
        f"{sp.tokens_emitted} tokens from {sp.target_passes} target passes -- "
        "no better than plain decoding"
    )
    print("  speculative decoding")
    print(_indent(sp.report()))



def _indent(text):
    return "\n".join("  " + line for line in text.splitlines())


if __name__ == "__main__":
    snapshot_path = model.qwen0_5_snapshot_path
    device = "cuda"
    engine = load_engine(snapshot_path, device)

    if "--record" in sys.argv:
        print("Recording Stage 2 reference generations")
        record_references(snapshot_path, engine, device)
        raise SystemExit(0)

    print("Stage 1 -- forward pass vs HuggingFace")
    check_forward_pass(snapshot_path, engine, device)

    print("Stage 2 -- incremental decode vs single-pass prefill")
    check_incremental_decode(snapshot_path, engine, device)

    print("Stage 2 -- recorded reference generations")
    check_reference_generations(snapshot_path, engine, device)

    print("Stage 4 -- batched decode vs one sequence at a time")
    check_batched_decode(snapshot_path, engine, device)

    print("Stage 4 -- scheduler at batch 1 vs generate()")
    check_scheduler_matches_generate(snapshot_path, engine, device)

    print("Stage 4 -- concurrent requests, constrained pool")
    check_scheduler_concurrent(snapshot_path, engine, device)

    print("Stage 5 -- prefix caching, shared system prompt")
    check_prefix_caching(snapshot_path, engine, device)

    print("Stage 5 -- prefix caching through the scheduler")
    check_prefix_caching_scheduler(snapshot_path, engine, device)

    print("Stage 5 -- prefix caching across conversation turns")
    check_prefix_caching_across_turns(snapshot_path, engine, device)

    print("Stage 4 -- head-of-line blocking in the admission loop")
    check_head_of_line_blocking(snapshot_path, engine, device)

    print("Stage 4/5 -- admission accounts for cached prefix")
    check_admission_accounts_for_prefix(snapshot_path, engine, device)

    print("Stage 5 -- probe_prefix agrees with match_prefix")
    check_probe_matches_match_prefix(snapshot_path, engine, device)

    print("Stage 4 -- preemption and readmission")
    check_preemption(snapshot_path, engine, device)

    print("Stage 6 -- rollback truncation")
    check_truncate()

    print("Stage 6 -- acceptance walk")
    check_acceptance_walk()

    print("Stage 6 -- stochastic acceptance preserves the target distribution")
    check_speculative_distribution()

    print("Stage 7 -- scheduler and pool counters agree with reality")
    check_instrumentation(snapshot_path, engine, device)

    print("Stage 6 -- draft loop vs Stage 2 references")
    check_draft_loop(snapshot_path, engine, device)

    print("Stage 6 -- target forward pass over a speculative window")
    check_verify_window(snapshot_path, engine, device)

    # The draft's weights are dead from here on, and the target section needs
    # every byte it can get.
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    print("Stage 6 -- target model (1.5B) re-gate")
    check_target_model(model.qwen1_5_snapshot_path, device)

    print("Stage 6 -- speculative decoding end to end vs target-only greedy")
    check_speculative_end_to_end(device)

    print("Stage 5/6 -- speculative decoding reuses cached prefixes")
    check_speculative_prefix_reuse(device)

    print("Stage 7 -- speculation counters agree with reality")
    check_spec_instrumentation(device)

    print("Stage 6 -- draft and target resident together")
    check_both_models_resident(device)

    print("PASS")
