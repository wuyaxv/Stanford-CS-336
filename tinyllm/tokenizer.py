from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Iterator
from abc import ABC

import regex as re

class BaseTokenizer(ABC):

    def __init__(self):
        ...

    def encode(self):
        """encoder 

        """
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class BPETokenizer(BaseTokenizer):
    """This is a implementation of Byte-Pair Encoding (BPE) Tokenizer

    TODO: 
        - represent arbitary Unicode strings as a sequence of bytes and train our BPE tokeinzer on this byte sequence.
        - Use trained tokenizer to encode text into tokens.

    DONE:

    UPDATEs:
        - Update (2025-12-08): Created the class

    """

    def __init__(self):
        self.vocab_size: int # vocabulary size
        self.special_tokens: list[str] # Special tokens which are used to encode metadata. They are always be treated as a single token.
        self.vocab: dict[int, bytes]

        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def _pre_tokenization(self, text: str) -> Iterator:
        '''Pre tokenization

        The original BPE implementation [Sennrich et al., 2016]
            uses method that simply splitting on whitespaces. In this 
            implementation we use a regex-based pre-tokenizer used
            by GPT-2 [Radford et al., 2019].

        Args:
            text (str): corpus (sentences) to be processed.
        '''

        return re.finditer(self.pat, text)


    def train(self):    
        ...

    def _train(self, text: str, vocab_size: int, special_tokens: list[str]):
        """Train the BPE tokenizer

        Step: Preprocessing. split corpus into words.
        Step: Initialize vocabulary with 256 characters and special_tokens.

        """
        
        assert vocab_size > 255 # This is must

        #base_vocab = [(i,bytes([b])) for i, b in enumerate(range(256))]
        base_vocab = [bytes([i]) for i in range(256)]
        for s in special_tokens:
            base_vocab.append(s.encode('utf-8'))
        # self.vocab = dict(base_vocab)

        pretokenized_words: Iterator = self._pre_tokenization(text)
        frequency_table: dict[tuple[byte, byte], int] = defaultdict(int)

        for word in pretokenized_words:
            _word = tuple(word[0].encode('utf-8'))
            if _word in frequency_table.keys():
                frequency_table[_word] += 1
            else:
                frequency_table[_word] = 1

        _merges: dict[tuple(bytes, bytes), int] = defaultdict(int) # Store current maximum frequence subwords
        for i in range(vocab_size - 255):
            for to_merge in frequency_table:
                for _ in zip(to_merge, to_merge[1:]): # Count frequency
                    _merges[_] = 1 if (_ not in _merges.keys()) else _merges[_]+1

            # update the frequency table
            _max_counts_k = max(_merges, key=_merges.get)
            base_vocab.append(bytes(_max_counts_k)) # append the frequency table

        return base_vocab

if __name__ == '__main__':
    text = '''
    low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
'''

    from pprint import pprint
    bpe = BPETokenizer()
    pprint(bpe._train(text, 289, ['<|endoftext|>']))




