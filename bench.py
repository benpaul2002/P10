"""Wall-clock benchmarks. Every timed region is bracketed by cuda.synchronize()
because CUDA is async -- without it we would be timing kernel launches."""
import gc
import statistics
import time
import torch
from transformers import AutoTokenizer

import model
import spec
from cache import Sequence
from scheduler import Request, Scheduler

DEV = "cuda"


def sync():
    torch.cuda.synchronize()


def timed(fn, n=1):
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    sync()
    return (time.perf_counter() - t0) / n, out


def load(path):
    cfg = model.load_config(path)
    w = model.load_weights(path)
    cos, sin = model.build_sincos_table(cfg, DEV)
    return cfg, w, cos, sin


def pool(cfg):
    return model.KVCache.preallocate(cfg, DEV)


def reset(kv):
    """Return the pool to a pristine state without reallocating its tensors,
    so pool allocation stays outside every timed region."""
    kv.free_blocks = list(range(kv.num_blocks))
    kv.refcounts = [0] * kv.num_blocks
    kv.registry = {}
    kv.block_hashes = [None] * kv.num_blocks


def fake_ids(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (n,), generator=g).tolist()


def warmup(eng):
    cfg, w, cos, sin = eng
    for _ in range(3):
        kv = pool(cfg)
        seq = Sequence()
        model.forward_prefill(torch.tensor([fake_ids(64)], device=DEV), w, cfg, cos, sin, kv, [seq])
        model.forward_decode(torch.tensor([[100]], device=DEV), w, cfg, cos, sin, kv, [seq])
    sync()


# ---------------------------------------------------------------- 1. TTFT
def bench_ttft(eng, lengths):
    cfg, w, cos, sin = eng
    kv = pool(cfg)
    out = {}
    for n in lengths:
        ids = fake_ids(n, seed=n)
        samples = []
        for _ in range(9):
            reset(kv)
            seq = Sequence(token_ids=list(ids))
            sync()
            t0 = time.perf_counter()
            logits = model.prefill(torch.tensor([ids], device=DEV), w, cfg, cos, sin, kv, [seq])
            model.sample(logits, torch.tensor([[0.0]], device=DEV),
                         torch.tensor([[1.0]], device=DEV))
            sync()
            samples.append((time.perf_counter() - t0) * 1000)
        out[n] = statistics.median(samples)
    return out


# ------------------------------------------------- 2. prefix cache on TTFT
def bench_prefix_ttft(eng, shared_len, tail_len):
    """Cold vs warm on the same pool object, so only the prefill work differs."""
    cfg, w, cos, sin = eng
    kv = pool(cfg)
    shared = fake_ids(shared_len, seed=7)

    def time_one(ids):
        seq = Sequence(token_ids=list(ids))
        sync()
        t0 = time.perf_counter()
        m = kv.match_prefix(seq)
        model.prefill(torch.tensor([ids[seq.length:]], device=DEV),
                      w, cfg, cos, sin, kv, [seq])
        sync()
        dt = (time.perf_counter() - t0) * 1000
        return dt, m

    cold, warm, matched = [], [], 0
    for i in range(9):
        # cold: empty registry, nothing to match
        reset(kv)
        dt, _ = time_one(shared + fake_ids(tail_len, seed=100 + i))
        cold.append(dt)

        # warm: prime the registry with the shared prefix, then time a
        # different request that shares it
        reset(kv)
        seq = Sequence(token_ids=shared + fake_ids(tail_len, seed=11))
        model.prefill(torch.tensor([seq.token_ids], device=DEV), w, cfg, cos, sin, kv, [seq])
        kv.register(seq)
        dt, m = time_one(shared + fake_ids(tail_len, seed=200 + i))
        warm.append(dt)
        matched = m
    return statistics.median(cold), statistics.median(warm), matched


# ------------------------------------------------------------------ 3. ITL
def bench_itl(eng, batch_sizes, steps=32):
    cfg, w, cos, sin = eng
    out = {}
    for b in batch_sizes:
        kv = pool(cfg)
        reset(kv)
        seqs = []
        for i in range(b):
            s = Sequence()
            model.forward_prefill(torch.tensor([fake_ids(64, seed=i)], device=DEV),
                                  w, cfg, cos, sin, kv, [s])
            seqs.append(s)
        tok = torch.full((b, 1), 100, device=DEV)
        # discard the first few steps -- allocator warm-up
        for _ in range(3):
            model.forward_decode(tok, w, cfg, cos, sin, kv, seqs)
        sync()
        per_step = []
        for _ in range(steps):
            t0 = time.perf_counter()
            model.forward_decode(tok, w, cfg, cos, sin, kv, seqs)
            sync()
            per_step.append((time.perf_counter() - t0) * 1000)
        out[b] = (statistics.median(per_step), statistics.median(per_step) / b)
    return out


# ----------------------------------------------------- 4. throughput vs load
def bench_throughput(eng, tokenizer, concurrencies, max_new=48):
    cfg, w, cos, sin = eng
    prompts = ["What is 2+2?", "Name three colours.", "Explain a KV cache.",
               "List three primes.", "What is a GPU?", "Say hello.",
               "Name a river.", "Define latency."]
    out = {}
    for n in concurrencies:
        kv = pool(cfg)
        sch = Scheduler(kv, cfg, w, cos, sin)
        for i in range(n):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompts[i % len(prompts)]}],
                tokenize=False, add_generation_prompt=True)
            sch.add_request(Request(Sequence(), max_new, i,
                                    prompt_ids=tokenizer(text).input_ids))
        sync()
        t0 = time.perf_counter()
        sch.run()
        sync()
        dt = time.perf_counter() - t0
        toks = sum(len(r.output_ids) for r in sch.finished)
        out[n] = (toks, dt, toks / dt, sch.stats.mean_batch_size)
    return out


# ------------------------------------------------- 5. speculation wall clock
def bench_spec(k=4, max_new=48):
    tok = AutoTokenizer.from_pretrained(model.qwen1_5_snapshot_path)
    d = load(model.qwen0_5_snapshot_path)
    t = load(model.qwen1_5_snapshot_path)
    warmup(t)
    warmup(d)
    prompts = ["What is 2+2?",
               "Explain what a KV cache is in one sentence.",
               "List three prime numbers."]
    rows = []
    for p in prompts:
        def baseline():
            return model.generate(p, max_new, tok, t[1], t[0], t[2], t[3], pool(t[0]))

        def speculative():
            return spec.speculative_generate(
                p, max_new, tok, d[1], d[0], d[2], d[3], t[1], t[0], t[2], t[3],
                pool(d[0]), pool(t[0]), k, temperature=0, top_p=1.0)

        dt_b, out_b = timed(baseline, n=3)
        dt_s, out_s = timed(speculative, n=3)
        rows.append((p, len(out_b), dt_b * 1000, len(out_s), dt_s * 1000,
                     out_b == out_s))
    del d, t
    gc.collect(); torch.cuda.empty_cache()
    return rows


def bench_admission(eng, tokenizer, n_blocks=9):
    """What optimistic admission buys, measured against reserving worst case.

    Conservative admission reserves prompt + max_new_tokens up front, so it can
    never overcommit and never preempts. Optimistic admission reserves only what
    a request needs now and preempts when the pool runs dry. Same requests, same
    pool -- the difference is how many fit at once.
    """
    from math import ceil
    from scheduler import Request, Scheduler
    cfg, w, cos, sin = eng
    prompts = [("What is 2+2?", 40),
               ("Explain what a KV cache is in one sentence.", 80),
               ("Write a haiku about GPUs.", 60),
               ("List three prime numbers.", 40),
               ("Describe the difference between a CPU and a GPU.", 32)]

    def run(conservative):
        kv = pool(cfg)
        kv.free_blocks = kv.free_blocks[:n_blocks]
        sch = Scheduler(kv, cfg, w, cos, sin)
        for i, (p, budget) in enumerate(prompts):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True)
            sch.add_request(Request(Sequence(), budget, i,
                                    prompt_ids=tokenizer(text).input_ids))
        if conservative:
            def can_admit(req):
                need = ceil((len(req.prompt_ids) + req.max_new_tokens) / kv.block_size)
                return need <= len(kv.free_blocks)
            sch.can_admit = can_admit

        peak, steps = 0, 0
        while sch.waiting_queue or sch.running:
            sch.schedule()
            steps += 1
            peak = max(peak, len(sch.running))
            if steps > 5000:
                raise RuntimeError("scheduler stalled")
        return peak, steps, sch.stats.preempted, sch.stats.mean_batch_size

    return {"conservative": run(True), "optimistic": run(False)}


if __name__ == "__main__":
    for name, path in (("0.5B", model.qwen0_5_snapshot_path),
                       ("1.5B", model.qwen1_5_snapshot_path)):
        eng = load(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        warmup(eng)
        print(f"\n===== {name} =====")

        print("-- TTFT (prompt -> first token)")
        for n, ms in bench_ttft(eng, [32, 128, 512]).items():
            print(f"   {n:4d} tok prompt   {ms:7.2f} ms")

        c, wm, matched = bench_prefix_ttft(eng, shared_len=480, tail_len=32)
        print("-- TTFT with prefix cache (480 shared + 32 new)")
        print(f"   cold             {c:7.2f} ms")
        print(f"   warm             {wm:7.2f} ms  ({matched} blocks matched, "
              f"{100*(1-wm/c):.0f}% faster)")

        print("-- inter-token latency (decode step)")
        for b, (step, per) in bench_itl(eng, [1, 2, 4, 8]).items():
            print(f"   batch {b:2d}         {step:6.2f} ms/step   {per:5.2f} ms/token")

        print("-- throughput vs concurrency")
        for n, (toks, dt, tps, mb) in bench_throughput(eng, tokenizer, [1, 2, 4, 8]).items():
            print(f"   {n:2d} requests      {tps:6.1f} tok/s   ({toks} tokens in "
                  f"{dt:.2f} s, mean batch {mb:.2f})")

        print("-- admission policy (5 requests, 9-block pool)")
        for label, (peak, steps, pre, mb) in bench_admission(eng, tokenizer).items():
            print(f"   {label:13s}  peak batch {peak}  {steps} steps  "
                  f"mean batch {mb:.2f}  {pre} preemptions")

        del eng
        gc.collect(); torch.cuda.empty_cache()

    print("\n===== speculative decoding, wall clock =====")
    for p, nb, db, ns, ds, same in bench_spec():
        print(f"   baseline {nb:3d} tok {db:8.1f} ms | spec {ns:3d} tok {ds:8.1f} ms "
              f"| {db/ds:.2f}x | identical={same}  {p!r}")


# ------------------------------------- 6. optimistic vs conservative admission
