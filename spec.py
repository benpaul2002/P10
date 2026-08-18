import torch
from cache import KVCache, Sequence
from model import ModelConfig, load_config, load_weights, forward_prefill, prefill, decode

def draft_tokens(weights, config, cos, sin, kvcache, seq, last_token, k):
    proposed = []
    last_token = torch.tensor([[last_token]], device=cos.device)
    for i in range(k):
        seq.token_ids.append(last_token.item())
        logits = decode(last_token, weights, config, cos, sin, kvcache, [seq])
        last_token = logits.argmax(-1, keepdim=True)
        proposed.append(last_token.item())
        if last_token.item() == config.eos_token_id:
            break
    return proposed

def target_tokens(weights, config, cos, sin, kvcache, seq, last_token, proposed):
    window = [last_token] + proposed
    token_ids = torch.tensor([window], device=cos.device)
    seq.token_ids.extend(window)
    return forward_prefill(token_ids, weights, config, cos, sin, kvcache, [seq])

def accept_proposal(logits, proposal):
    preds = logits.argmax(-1)[0].tolist()
    for i in range(len(proposal)):
        if preds[i] != proposal[i]:
            return proposal[:i] + [preds[i]], i
    return proposal + [preds[len(proposal)]], len(proposal)

def speculative_generate(prompt, max_new_tokens, tokenizer, draft_weights, draft_config, draft_cos, draft_sin, target_weights, target_config, target_cos, target_sin, draft_kvcache, target_kvcache, k):
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
    
    last_token = int(target_logits.argmax(-1))
    generated.append(last_token)
    if last_token == target_config.eos_token_id:
        for block_id in draft_seq.block_table:
            draft_kvcache.free(block_id)
        for block_id in target_seq.block_table:
            target_kvcache.free(block_id)
        return generated

    while len(generated) < max_new_tokens:
        proposed = draft_tokens(draft_weights, draft_config, draft_cos, draft_sin, draft_kvcache, draft_seq, last_token, k)
        logits = target_tokens(target_weights, target_config, target_cos, target_sin, target_kvcache, target_seq, last_token, proposed)
        emitted, num_accepted = accept_proposal(logits, proposed)
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
