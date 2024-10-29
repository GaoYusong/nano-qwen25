from dataclasses import dataclass
import json
import os
from pathlib import Path
import fire
from safetensors import safe_open
from torch import nn

from tokenizer import Tokenizer

@dataclass
class ModelArgs:
    vocab_size: int = -1
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

class Qwen:
    @staticmethod
    def build(model_path: str) -> "Qwen":
        # Read the code from refs/main to find the snapshot
        with open(Path(model_path) / "refs" / "main", "r") as f:
            snapshot_code = f.read().strip()
        
        # Construct full snapshot path
        snapshot_path = Path(model_path) / "snapshots" / snapshot_code
        
        # Load the checkpoint
        checkpoint = {}
        with safe_open(Path(snapshot_path) / "model.safetensors", framework="pt", device=0) as f:
            for k in f.keys():
                checkpoint[k] = f.get_tensor(k)
                
        # Load the model parameters
        with open(Path(snapshot_path) / "config.json", "r") as f:
            params = json.loads(f.read())
        model_args = ModelArgs(**params)
        
        # Load the tokenizer
        tokenizer = Tokenizer(131072)
        
        # Initialize the model
        model = Transformer(model_args)
        model.load_state_dict(checkpoint)
        
        return Qwen(model=model, tokenizer=tokenizer)
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

def main(
    model_path: str = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"),
):
    qwen = Qwen.build(
        model_path=model_path
    )
    
    model = qwen.model
    model.train()

    

if __name__ == "__main__":
    fire.Fire(main)