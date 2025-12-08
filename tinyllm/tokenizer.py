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
        self.merge: list[dict[bytes, bytes]] = list()

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
        # Let's just define pre tokenization as words spliting on space for now.
        #return iter(text.split(' '))
    
    def _merge(self, t: tuple[bytes, bytes], new_idx: int, frequency_table: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
        """Merge freqeuency table
        
        Combines the bytes which determined as the most freqeuent subword token.
        """

        new_frequency_table: list[tuple[tuple[bytes], int]] = list()
        i: int = 0

        for sub, count in frequency_table.items():
            _ = []
            i = 0
            while i < len(sub)-1:
                if sub[i] == t[0] and sub[i+1] == t[1]:
                    _.append(new_idx) # merge the most freqeuent subset
                    i+=2
                else:
                    _.append(sub[i])
                    i+=1
            if i == len(sub)-1: _.append(sub[i])
            new_frequency_table.append((tuple(_), count))

        return defaultdict(int, new_frequency_table)

    def _max(self, base_vocab, d: dict[tuple[bytes, bytes], int]) -> bytes:
        __max = 0
        r = []

        for k, v in d.items():
            if v > __max:
                r.clear()
                r.append((k, base_vocab[k[0]]+base_vocab[k[1]], v))
                __max = v
            elif v == __max:
                r.append((k, base_vocab[k[0]]+base_vocab[k[1]], v))

        return sorted(r, key=lambda k: k[1])[-1][0]


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
        frequency_table: dict[tuple[bytes], int] = defaultdict(int)

        for word in pretokenized_words:
            _word = tuple(word[0].encode('utf-8'))
            if _word in frequency_table.keys():
                frequency_table[_word] += 1
            else:
                frequency_table[_word] = 1

        for i in range(vocab_size - 255):
            _merges: dict[tuple[bytes, bytes], int] = defaultdict(int) # Store current maximum frequence subwords
            for to_merge, v in frequency_table.items():
                for _ in zip(to_merge, to_merge[1:]): # Count frequency
                    _merges[_] = 1*v if (_ not in _merges.keys()) else _merges[_]+1*v
            # update the frequency table
            try:
                _max_counts_k = self._max(base_vocab, _merges)
            except ValueError:
                break

            base_vocab.append(base_vocab[_max_counts_k[0]]+base_vocab[_max_counts_k[1]]) # append the frequency table

            # so here we have to implement a merge function.
            frequency_table = self._merge(_max_counts_k, len(base_vocab)-1, frequency_table)
            self.merge.append(_max_counts_k)

        return frequency_table, base_vocab, self.merge

if __name__ == '__main__':
    text = '''low low low low low lower lower widest widest widest newest newest newest newest newest newest'''

    from pprint import pprint
    bpe = BPETokenizer()
    with open("tests/fixtures/TinyStoriesV2-GPT4-train.txt", 'r') as f:
        text = f.read()
        pprint(bpe._train(text, 500, ['<|endoftext|>']))




