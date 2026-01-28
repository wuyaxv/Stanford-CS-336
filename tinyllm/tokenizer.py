from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Union, Iterator
from collections import defaultdict

class Tokenizer(ABC):
    """Tokenizer Abstract Classs
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        special_tokens: List[str]|None = None,
        **kwargs
    ):

        self.vocab_size: int = vocab_size   # 词典大小
        self.special_tokens: List[str] = special_tokens if special_tokens else []
        self.vocab: Dict[int, bytes] = {}     # 词典，约定使用str来表示每个索引
        self.inverse_vocab: Dict[bytes, int] = {}
    
    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass

    @abstractmethod
    def train(
        self,
        corpus: Dict[str, int]
    ):
        pass

    @property
    def get_vocab_size(self):
        return len(self.vocab)
