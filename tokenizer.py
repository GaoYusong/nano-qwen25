
from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer
from typing import List

import utils

class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = HFTokenizer.from_file(str(Path(utils.newest_snapshot(model_path)) / "tokenizer.json"))
        
    def encode(self, text: str)-> List[int]:
        return self.tokenizer.encode(text).ids
    
    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

if __name__ == "__main__":
    tokenizer = Tokenizer(utils.MODEL_PATH)
    encoded = tokenizer.encode("Hello, 你好! How are you 😁 ?")
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)
