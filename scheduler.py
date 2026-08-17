from dataclasses import dataclass, field 
import torch
from cache import Sequence
from enum import Enum
from cache import KVCache
from model import ModelConfig, load_config, load_weights, prefill, decode
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
        blocks_needed = ceil((len(request.prompt_ids) + request.max_new_tokens) / self.kvcache.block_size)
        return blocks_needed <= len(self.kvcache.free_blocks)

    def add_request(self, request):
        blocks_needed = ceil((len(request.prompt_ids) + request.max_new_tokens) / self.kvcache.block_size)
        if blocks_needed > self.kvcache.num_blocks:
            raise RuntimeError("Request prompt too big, cannot accomodate!")
        self.waiting_queue.append(request)

    def retire(self, req):
        for block_id in req.seq.block_table:
            self.kvcache.free(block_id)
        req.seq.block_table.clear()
        req.seq.length = 0
        req.state = RequestState.DONE
        self.finished.append(req)

    def schedule(self):
        for req in self.running:
            if req.is_finished(self.config.eos_token_id):
                self.retire(req)
        self.running = [req for req in self.running if not req.is_finished(self.config.eos_token_id)]

        for req in list(self.waiting_queue):
            if self.can_admit(req):
                self.waiting_queue.remove(req)
                logits = prefill(torch.tensor([req.prompt_ids], device=self.cos.device), self.weights, self.config, self.cos, self.sin, self.kvcache, [req.seq])
                token = int(logits.argmax(-1))
                req.output_ids.append(token)
                if req.is_finished(self.config.eos_token_id):
                    self.retire(req)  
                else:
                    self.running.append(req)
                    req.state = RequestState.RUNNING
            else:
                break

        if len(self.running)>0:
            seq_list = [req.seq for req in self.running]
            token_ids = torch.tensor([[req.next_token()] for req in self.running], device=self.cos.device)
            out = decode(token_ids, self.weights, self.config, self.cos, self.sin, self.kvcache, seq_list).argmax(-1).tolist()
            for req, token in zip(self.running, out):
                req.output_ids.append(token)
            
    def run(self, ):
        while len(self.waiting_queue)>0 or len(self.running)>0:
            self.schedule()
