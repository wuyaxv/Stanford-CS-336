from dataclasses import dataclass
from sortedcontainers import SortedKeyList, SortedDict
from typing import Generator
from collections import defaultdict

from pre_tokenizer import GPT2PreTokenizer

import io
import os

@dataclass
class MergedPair:
    count: int # Merged pair count in frequency_table
    in_bytes: bytes # Byte format of the MergedPair
    index: list[tuple[int, ...], ...] # Stores the mapping to the frequency_table

    def _key(self):
        return (self.count, self.in_bytes)

class BPETokenizer:

    def __init__(self, \
            source: str | bytes | os.PathLike | io.TextIOBase | io.BufferedIOBase, \
            special_tokens: list[str | bytes], \
            chunk_size: int = 4096):
        """Handle the necessary operations

        遍历frequency_table时，确保每次将对应的映射关系存放到merged_pair中
        """
        self.source = source
        self.special_tokens = special_tokens

        self.vocab: dict[id, bytes] = self._init_vocab()# f: id -> bytes
        self.inverse_vocab = {k:v for v,k in self.vocab.items()}# f: bytes -> id
        self.frequency_table: dict[tuple[int, ...], int] \
                = defaultdict(int)
        self.merged_pairs = ... # TODO: Implement a sorted pair incase I need to update the list
        self.pre_tokenizer: GPT2PreTokenizer
        self.chunk_size = chunk_size

    def encode(self):
        ...

    def decode(self):
        ...

    def train(self):
        ...

    def _train(self):
        """Train a BPE Tokenizer

        Corpuse are from @property text
        """

        for t in self.text: # Go through the corpus and generate the frequency table
            print(type(t))
            self._update_frequency_table(t)

    def _merge(self):
        ...

    @property
    def text(self):
        return self._read_from_source(chunk_size=self.chunk_size)

    def _read_from_source(self, \
            start: int | None = None, \
            end: int | None = None, \
            chunk_size: int = 4096):
        """Read from source in chunked bytes (default in 4096 bytes)

        If the start and end is designated, then read from start to end
        int bytes
        """

        if isinstance(self.source, os.PathLike): # Read from a file
            length_of_file: int

            # Get then length of file
            with open(self.source, 'rb') as f:
                length_of_file = f.seek(0, os.SEEK_END)

            if not start: start = 0
            if not end: end = length_of_file
            assert start <= end
            total_read = end - start

            with open(self.source, 'rb') as f: # Read in `chunk_size` bytes a chunk
                for i in range(total_read // chunk_size):
                    yield f.read(chunk_size)
                yield f.read(total_read % chunk_size)
                
        elif isinstance(self.source, io.TextIOBase) or isinstance(self.source, io.BufferedIOBase):
            current_pos = self.source.seek(0, os.SEEK_CUR)
            end_of_file = self.source.seek(0, os.SEEK_END)
            self.source.seek(current_pos, os.SEEK_SET)
            if not start: start = current_pos
            if not end: end = end_of_file
            print(start, end)
            
            assert current_pos <= start
            assert end <= end_of_file
            assert start <= end
            total_read = end - start

            for i in range(total_read // chunk_size):
                yield self.source.read(chunk_size).encode('utf-8') if isinstance(self.source, io.TextIOBase) else self.source.read(chunk_size)
            yield self.source.read(total_read % chunk_size).encode('utf-8') if isinstance(self.source, io.TextIOBase) else self.source.read(total_read % chunk_size)

            self.source.seek(current_pos, os.SEEK_SET) # set io stream to the original state
        elif isinstance(self.source, str):
            i = 0
            for i in range(len(self.source)//chunk_size):
                yield self.source[i*chunk_size:(i+1)*chunk_size].encode('utf-8')
            yield self.source[i*chunk_size:].encode('utf-8')
        elif isinstance(self.source, bytes):
            i = 0
            for i in range(len(self.source)//chunk_size):
                yield self.source[i*chunk_size:(i+1)*chunk_size]
            yield self.source[i*chunk_size:]
        else:
            raise ValueError

    def _lookup(self, i:int) -> bytes:
        """Lookup the vocabulary table by ID
        """
        assert i < len(self.vocab)

        return self.vocab[i]

    def _inverse_lookup(self, w:bytes) -> id:
        """Reverse the vocabulary table by bytes and returns the respective ID
        """
        assert w in self.inverse_vocab

        return self.inverse_vocab[w]

    def _assemble_bytes(self, pair: tuple[int, int]) -> bytes:
        """Take a pair of IDs and return their assembled bytes

        Args:
            pair (tuple[int, int])

        Returns:
            Assembled bytes (bytes)
        """
        return self._lookup(pair[0])+self._lookup(pair[1])

    @property
    def vocab_size(self):
        """Get the vocabulary size
        """
        return len(self.vocab_size)

    def _update_vocab(self, i: int, pair: tuple[int, int]):
        """Update the vocabulary based on integer pairs,
            those integers in the pair must be in the vocab already
        """
        
        new_pair_bytes = self._assemble_bytes(pair)
        new_id = self.vocab_size # New id is the end of queue of current vocab
        self.vocab[new_id] = new_pair_bytes
        self.inverse_vocab[new_pair_bytes] = new_id

    def _is_in_vocab(self, k: int | bytes):
        if isinstance(k, int):
            if k in self.vocab: return True
            else: return False
        else:
            if k in self.inverse_vocab: return True
            else: return False

    def _update_frequency_table(self, \
            text: str | os.PathLike | io.TextIOBase | io.BufferedIOBase):
        gpt2_pt = GPT2PreTokenizer(text, self.special_tokens)

        for t in gpt2_pt.pre_tokenizer(): # 获取的token首先将其按字节拆分
            assert isinstance(t, bytes)

            new_k = tuple([t[i:i+1] for i in range(len(t))])
            if new_k in self.frequency_table:
                self.frequency_table[new_k] += 1
            else:
                self.frequency_table[new_k] = 1 # Add new entry

    def _init_vocab(self):
        """Initialize vocabulary using special tokens and bytes from 0 to 255
        """
        vocab = {}
        for st in self.special_tokens:
            assert isinstance(st, bytes)
            vocab[len(vocab)] = st
    
        for i in range(256):
            vocab[len(vocab)] = bytes([i])
        return vocab

if __name__ == '__main__':
    import pathlib
    p = pathlib.Path('Sampletext')
    special_tokens=[b'<|endoftext|>']
    with open(p, 'rb') as f:
        bpe = BPETokenizer(f, special_tokens)
        bpe._train()
    print(bpe.frequency_table)
