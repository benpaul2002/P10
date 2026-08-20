# llm-engine

An inference engine for Qwen2.5, written in PyTorch.
Implements a paged KV cache with prefix caching, continuous batching, and speculative decoding. 
Pretrained weights, tokenizer, and `safetensors` from HuggingFace.

```
model.py      forward pass: RMSNorm, RoPE, GQA attention, SwiGLU, tied head
cache.py      paged KV cache: block pool, block tables, refcounts, prefix registry
scheduler.py  continuous batching: admission, preemption, eviction
spec.py       speculative decoding
```

## Numbers

**Forward pass** vs HuggingFace in fp32, 40-token prompt

| | mean abs diff | max abs diff | decisive mismatches |
|---|---|---|---|
| 0.5B draft | 0.095 | 2.40 | 0 / 19 |
| 1.5B target | 0.128 | 3.32 | 0 / 27 |

HF's own bf16 logits differ from its fp32 by about the same margin, so the absolute diffs say little on their own. 
Decisive mismatches: A position is *decisive* if fp32's top two logits are more than 1.0 apart - a single bf16 rounding step is ~0.25, so rounding alone cannot flip the winner there. A decisive mismatch is such a position where our argmax disagrees with fp32's: near-tied positions may legitimately flip, decisive ones may not.

**KV cache.** 16-token blocks, 128-block pool. GQA (14 query heads, 2 KV heads) makes it 7x smaller than MHA: 12 KB/token on the 0.5B, 28 KB/token on the 1.5B.

**Prefix caching**, four requests sharing a 54-token system prompt: 20 blocks allocated without sharing, 10 with. Across a conversation turn, 80 tokens matched - 46 of them past the first prompt's boundary, in blocks filled during decode. In one scheduler run, 43% of prefill tokens were served from cache.

**Scheduling.** Admission scans past a request that does not fit, capped at 3 skips so a long prompt cannot be starved. 5 requests, 10-block pool, identical output either way:

| | decode steps | mean batch |
|---|---|---|
| strict FIFO | 34 | 1.65 |
| scan past | 25 | 2.24 |

**Speculative decoding**, k=4, greedy, 0.5B drafting for the 1.5B: 56 tokens in 19 target passes (**2.95x fewer target forward passes**, not wall clock - each round also costs 4 draft passes), acceptance rate 0.52. Output is token-identical to the target decoding alone. Both models plus both pools fit in 4056 MB of 7808 MB.
