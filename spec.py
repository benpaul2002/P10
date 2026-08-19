import torch
from cache import KVCache, Sequence
from model import ModelConfig, load_config, load_weights, forward_prefill, prefill, decode, probs_from_logits, sample_from_probs, sample

def draft_tokens(weights, config, cos, sin, kvcache, seq, last_token, k, temperature, top_p):
    proposed = []
    last_token = torch.tensor([[last_token]], device=cos.device)
    q_rows = []
    for i in range(k):
        seq.token_ids.append(last_token.item())
        logits = decode(last_token, weights, config, cos, sin, kvcache, [seq])
        probs = probs_from_logits(logits, temperature, top_p)
        q_rows.append(probs)
        last_token = sample_from_probs(probs).view(1, 1)
        proposed.append(last_token.item())
        if last_token.item() == config.eos_token_id:
            break
    return proposed, torch.cat(q_rows, dim=0)

def target_tokens(weights, config, cos, sin, kvcache, seq, last_token, proposed):
    window = [last_token] + proposed
    token_ids = torch.tensor([window], device=cos.device)
    seq.token_ids.extend(window)
    return forward_prefill(token_ids, weights, config, cos, sin, kvcache, [seq])

def accept_proposal(logits, proposal, q, temperature, top_p):
    # preds = logits.argmax(-1)[0].tolist()
    p = probs_from_logits(logits, temperature, top_p).squeeze(0)
    for i in range(len(proposal)):
        # if preds[i] != proposal[i]:
        #     return proposal[:i] + [preds[i]], i
        t = proposal[i]
        r = torch.rand(())
        if r >= (p[i, t]/q[i, t]):
            residual = (p[i] - q[i]).clamp(min=0)
            residual = residual / residual.sum().clamp(min=1e-10)
            sampled = int(sample_from_probs(residual))
            return proposal[:i] + [sampled], i
    sampled = int(sample_from_probs(p[len(proposal)]))
    return proposal + [sampled], len(proposal)

def speculative_generate(prompt, max_new_tokens, tokenizer, draft_weights, draft_config, draft_cos, draft_sin, target_weights, target_config, target_cos, target_sin, draft_kvcache, target_kvcache, k, temperature=0, top_p=1.0):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer(text, return_tensors="pt").input_ids.to(target_cos.device)

    prompt_list = token_ids[0].tolist()
    draft_seq = Sequence(token_ids=list(prompt_list))
    target_seq = Sequence(token_ids=list(prompt_list))

    generated = []

    draft_logits = prefill(token_ids, draft_weights, draft_config, draft_cos, draft_sin, draft_kvcache, [draft_seq])
    target_logits = prefill(token_ids, target_weights, target_config, target_cos, target_sin, target_kvcache, [target_seq])

    temperature_t = torch.tensor([[temperature]], dtype=torch.float32, device=target_cos.device)
    top_p_t = torch.tensor([[top_p]], dtype=torch.float32, device=target_cos.device)
    
    last_token = int(sample(target_logits, temperature_t, top_p_t))
    generated.append(last_token)
    if last_token == target_config.eos_token_id:
        for block_id in draft_seq.block_table:
            draft_kvcache.free(block_id)
        for block_id in target_seq.block_table:
            target_kvcache.free(block_id)
        return generated

    while len(generated) < max_new_tokens:
        proposed, q = draft_tokens(draft_weights, draft_config, draft_cos, draft_sin, draft_kvcache, draft_seq, last_token, k, temperature_t, top_p_t)
        logits = target_tokens(target_weights, target_config, target_cos, target_sin, target_kvcache, target_seq, last_token, proposed)
        emitted, num_accepted = accept_proposal(logits, proposed, q, temperature_t, top_p_t)
        L = target_seq.length - (len(proposed) - num_accepted)
        target_kvcache.truncate(target_seq, L)
        if draft_seq.length > L:
            draft_kvcache.truncate(draft_seq, L)
        if draft_seq.length < L:
            gap = target_seq.token_ids[draft_seq.length:L]
            draft_seq.token_ids.extend(gap)
            forward_prefill(torch.tensor([gap], device=draft_cos.device), draft_weights, draft_config, draft_cos, draft_sin, draft_kvcache, [draft_seq])

        last_token = emitted[-1]
        generated.extend(emitted)
        if target_config.eos_token_id in generated:
            del generated[generated.index(target_config.eos_token_id) + 1:]
            break

    for block_id in draft_seq.block_table:
        draft_kvcache.free(block_id)
    for block_id in target_seq.block_table:
        target_kvcache.free(block_id)

    del generated[max_new_tokens:]

    return generated
