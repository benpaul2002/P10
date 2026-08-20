# llm-engine

An inference engine for Qwen2.5, written in PyTorch.
Implements a paged KV cache with prefix caching, continuous batching, and speculative decoding. 
Pretrained weights and tokenizer from HuggingFace.

Hardware - RTX 4060 GPU, 8GB VRAM.

Models used: Qwen2.5-0.5B-Instruct (draft), Qwen2.5-1.5B-Instruct (target)

```
model.py      forward pass: RMSNorm, RoPE, GQA attention, SwiGLU, tied head
cache.py      paged KV cache: block pool, block tables, refcounts, prefix registry
scheduler.py  continuous batching: admission, preemption, eviction
spec.py       speculative decoding
bench.py      wall-clock benchmarks (TTFT, inter-token latency, throughput)
```

## Numbers

**KV cache** 1

6-token blocks, 128-block pool. GQA makes it far smaller than MHA would: the 1.5B shares 12 query heads across 2 KV heads, so 6x less cache. 28 KB/token on the 1.5B, 12 KB/token on the 0.5B.

**Prefix caching** 

*Concurrent requests sharing a system prompt* - Ran with four requests, each with a 54-token system prompt. 20 blocks allocated in naive implementation, 10 with prefix caching. Both resulted in the same outputs.

*A follow-up turn in one conversation* - Ran a 34-token prompt, then a follow-up re-sending the whole history. Turn 2 is 67 tokens, 53 of them already computed. 48 matched (3 full blocks), so only 19 needed prefilling.

*Latency* - Ran a 512-token prompt with its first 480 tokens already cached. TTFT drops from 84 ms to 23 ms, 73% faster.

**Scheduling** 

Admission scans past a request that does not fit, capped at 3 skips so a long prompt cannot be starved. Ran 6 requests, one long prompt queued behind two others, 20-block pool, all generating 96 tokens:

| | decode steps | mean batch |
|---|---|---|
| strict FIFO | 213 | 2.68 |
| scan past | 184 | 3.10 |

decode_steps: number of decode steps to finish all 6 requests. 

mean_batch: how many requests each decode step served. A step costs nearly the same at batch 1 or 8, so a fuller batch is close to free throughput.

**Preemption** 

Admission is optimistic (assume most requests won't hit the max sequence length limit). This can cause the pool to exhaust, which is handled by preempting the newest running request and rebuilding its KV on readmission. Ran 5 requests with a 9-block pool, against worst case admission (reserves prompt + max_new_tokens up front):

| | peak batch | decode steps | preemptions |
|---|---|---|---|
| worst case | 2 | 69 | 0 |
| optimistic | 4 | 49 | 4 |

Twice the concurrency and 29% fewer steps out of the same memory. Both led to same outputs.

**Latency** 

Time to first token, and per-token cost as the batch grows:

| | TTFT 512 tok | ms/step @1 | ms/step @8 |
|---|---|---|---|
| 0.5B | 32.0 | 11.5 | 16.5 |
| 1.5B | 85.8 | 20.7 | 27.1 |

A batch-8 decode step costs only 1.4x a batch-1 step but serves 8 sequences, so per-token cost drops 6x. Throughput follows: 47 tok/s at 1 request, 156 tok/s at 8 (1.5B).

**Speculative decoding** 

Used the smaller (0.5B) model as a draft model for the target (1.5B) model. Draft model produces k=4 proposal tokens, target model goes through all k at once and greedily picks all that match. Ran three prompts - 56 tokens from 19 target passes (2.95x fewer than one pass per token). Acceptance rate of 0.52. Output is token-identical to the target decoding alone.
However, it's slower - 0.81x, 0.80x, 1.20x on those three prompts. This is because at batch 1, cost is dominated by launch overhead rather than FLOPs. A larger target would show a speedup, however this isn't possible here due to memory constraints.
