

from typing import List


class Tokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
