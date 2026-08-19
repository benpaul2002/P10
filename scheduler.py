from dataclasses import dataclass, field 
import torch
from enum import Enum
from cache import KVCache, Sequence
from model import ModelConfig, load_config, load_weights, prefill, decode, sample
from math import ceil

class RequestState(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    DONE = "done"

@dataclass
class Request:
    seq: Sequence
    max_new_tokens: int
    request_id: int
    state: RequestState = RequestState.WAITING
    prompt_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    temperature: int = 0
    top_p: float = 1.0

    def next_token(self):
        return self.output_ids[-1]

    def is_finished(self, eos_token_id):
        return (self.output_ids[-1] == eos_token_id or len(self.output_ids) >= self.max_new_tokens)

@dataclass
class Scheduler:
    kvcache: KVCache
    config: ModelConfig
    weights: dict[str, torch.Tensor]
    cos: torch.Tensor
    sin: torch.Tensor
    waiting_queue: list[Request] = field(default_factory=list)
    running: list[Request] = field(default_factory=list)
    finished: list[Request] = field(default_factory=list)

    def can_admit(self, request):
        blocks_needed = ceil(len(request.prompt_ids + request.output_ids[:-1]) / self.kvcache.block_size)
        return blocks_needed <= len(self.kvcache.free_blocks)

    def get_num_new_blocks_needed_decode(self):
        num_new_blocks_needed = 0
        for req in self.running:
            if req.seq.length % self.kvcache.block_size == 0:
                num_new_blocks_needed += 1
        return num_new_blocks_needed

    def add_request(self, request):
        blocks_needed = ceil((len(request.prompt_ids) + request.max_new_tokens) / self.kvcache.block_size)
        if blocks_needed > self.kvcache.num_blocks:
            raise RuntimeError("Request prompt too big, cannot accomodate!")
        request.seq.token_ids = list(request.prompt_ids)
        self.waiting_queue.append(request)

    def retire(self, req):
        for block_id in req.seq.block_table:
            self.kvcache.free(block_id)
        req.seq.block_table.clear()
        req.seq.length = 0
        req.state = RequestState.DONE
        self.finished.append(req)

    def preempt(self, req):
        for block_id in req.seq.block_table:
            self.kvcache.free(block_id)
        req.seq.block_table.clear()
        req.seq.length = 0
        req.state = RequestState.WAITING
        self.running.remove(req)
        self.waiting_queue.insert(0, req)

    def schedule(self):
        for req in self.running:
            if req.is_finished(self.config.eos_token_id):
                self.retire(req)
        self.running = [req for req in self.running if not req.is_finished(self.config.eos_token_id)]

        for req in list(self.waiting_queue):
            if self.can_admit(req):
                self.waiting_queue.remove(req)
                req.seq.token_ids = req.prompt_ids + req.output_ids[:-1]
                num_matched_blocks = self.kvcache.match_prefix(req.seq)
                logits = prefill(torch.tensor([req.seq.token_ids[req.seq.length:]], device=self.cos.device), self.weights, self.config, self.cos, self.sin, self.kvcache, [req.seq])
                self.kvcache.register(req.seq)
                token = int(sample(logits, torch.tensor([[req.temperature]], dtype=torch.float32, device=self.cos.device), torch.tensor([[req.top_p]], dtype=torch.float32, device=self.cos.device)))
                if len(req.output_ids) == 0:
                    req.output_ids.append(token)
                if req.is_finished(self.config.eos_token_id):
                    self.retire(req)  
                else:
                    self.running.append(req)
                    req.state = RequestState.RUNNING
            else:
                break

        num_new_blocks_needed = self.get_num_new_blocks_needed_decode()
        while self.running and num_new_blocks_needed > len(self.kvcache.free_blocks):
            self.preempt(self.running[-1])
            num_new_blocks_needed = self.get_num_new_blocks_needed_decode()
        if len(self.running)>0:
            seq_list = [req.seq for req in self.running]
            token_ids = torch.tensor([[req.next_token()] for req in self.running], device=self.cos.device)
            # out = decode(token_ids, self.weights, self.config, self.cos, self.sin, self.kvcache, seq_list).argmax(-1).tolist()
            temperatures = torch.tensor([[req.temperature] for req in self.running], dtype=torch.float32, device=self.cos.device)
            top_ps = torch.tensor([[req.top_p] for req in self.running], dtype=torch.float32, device=self.cos.device)
            out = decode(token_ids, self.weights, self.config, self.cos, self.sin, self.kvcache, seq_list)
            sampled = sample(out, temperatures, top_ps).tolist()
            for req, token in zip(self.running, sampled):
                req.seq.token_ids.append(req.next_token())
                req.output_ids.append(token)
                self.kvcache.register(req.seq)
            
    def run(self):
        while len(self.waiting_queue)>0 or len(self.running)>0:
            self.schedule()
