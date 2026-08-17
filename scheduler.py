from dataclasses import dataclass, field 
import torch
from cache import Sequence
from enum import Enum

class State(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    DONE = "done"

@dataclass
class Request:
    seq: Sequence
    max_new_tokens: int
    request_id: int
    state: State = State.WAITING
    prompt_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)

    def next_token(self):
        return output_ids[-1]
