from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
import fire
from safetensors import safe_open
from torch import nn
import torch

from tokenizer import Tokenizer
import utils

# -----------------------------------------------------------------------------
# ModelArgs

@dataclass
class ModelArgs:
    hidden_size: int = -1
    vocab_size: int = -1
    num_hidden_layers: int = -1
    num_attention_heads: int = -1
    num_key_value_heads: int = -1
    rms_norm_eps: float = -1.0
    rope_theta: float = -1.0
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
                
# -----------------------------------------------------------------------------
# Transformer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x, freqs_cis):
    pass

class Attention(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()


class FeedForward(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

class TransformerBlock(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.attention = Attention(params)
        self.feed_forward = FeedForward(params)
        self.attention_norm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        self.ffn_norm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.hidden_size)
        self.layers = nn.ModuleList(
            TransformerBlock(params) for _ in range(params.num_hidden_layers)
        )
        self.norm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        self.output = nn.Linear(params.hidden_size, params.vocab_size, bias=False)
        
        self.freqs_cis = precompute_freqs_cis(
            params.hidden_size // params.num_attention_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

        
    def forward_inference(self, tokens: torch.Tensor, start_pos: int):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        mask = None
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        
        h = self.norm(h)
        output = self.output(h).float()
        return output

# -----------------------------------------------------------------------------
# Qwen

class Qwen:
    @staticmethod
    def build(model_path: str) -> "Qwen":
        snapshot_path = utils.newest_snapshot(model_path)
        
        # Load the checkpoint
        checkpoint = {}
        with safe_open(Path(snapshot_path) / "model.safetensors", framework="pt", device=0) as f:
            for k in f.keys():
                checkpoint[k] = f.get_tensor(k)
                
        # Load the model parameters
        with open(Path(snapshot_path) / "config.json", "r") as f:
            params = json.loads(f.read())
        model_args = ModelArgs(**params)
        
        # Print the model parameters
        print(model_args)
        
        # Load the tokenizer
        tokenizer = Tokenizer(model_path)
        
        # Initialize the model
        model = Transformer(model_args)
        model.load_state_dict(checkpoint)
        
        return Qwen(model=model, tokenizer=tokenizer)
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    @torch.inference_mode()
    def generate(
        self, 
        prompt_tokens: List[List[int]],
        sample_rng: torch.Generator,
        max_gen_len: int,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)
        
        # TODO: install KV cache
        
        out_tokens = []
        return out_tokens
        
    def text_completion(
        self, 
        prompts: List[str],
        sample_rng: torch.Generator,
        max_gen_len: Optional[int] = None,
    ):
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        
        prompt_tokens = [self.tokenizer.encode(p) for p in prompts]
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            sample_rng=sample_rng,
            max_gen_len=max_gen_len,
        )
        completions = [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]
        return completions


def main(
    model_path: str = utils.MODEL_PATH,
):
    qwen = Qwen.build(
        model_path=model_path
    )
    
    model = qwen.model
    model.train()

    

if __name__ == "__main__":
    fire.Fire(main)