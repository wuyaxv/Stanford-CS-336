from abc import ABC
from dataclasses import dataclass
from collections import defaultdict

import os

@dataclass(frozen=True)
class TokenizerParams:
    vocab: ...
    merges: ...

class Tokenizer(ABC):
    """Abstract interface for a tokenizer.

    There are 3 primary functions of a Tokenizer:
        1) Train the tokenizer vocabulary and merges on a given text,
        2) encode from text to tokens,
        3) decode from tokens to text.
    """
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

    def train(self, input_path, vocab_size: int, special_tokens: list) -> TokenizerParams:
        raise NotImplementedError


@dataclass(frozen=True) # frozen=True => assigning to fields will generate an exception.
class BPETokenizerParams(TokenizerParams):
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index

class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))
    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))

class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")  # @inspect string_bytes
        indices = list(map(int, string_bytes))  # @inspect indices
        return indices
    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)  # @inspect string_bytes
        string = string_bytes.decode("utf-8")  # @inspect string
        return string

class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams | None = None):
        self.params = params if params else BPETokenizerParams(dict(), dict())

    def encode(self, string: str) -> list[int]:
        """encode
        :param string
        """
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string

    def train(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str] = ["<|endoftext|>"]) -> BPETokenizerParams:
        """Train a Byte-Pair Embedding (BPE) tokenizer

        The BPE algorithm was orignally described in 1994: "A New Algorithm for Data Compression" by Philip Gage.

        BPE algorithm outline:
            1. Identify frequent pairs. In each iteration, scan the text for the most commonly occurring pair of bytes.
            2. Replace and record. Replace that pair with a new placeholder ID (> 255), the first placeholder would be 256,
                cause the bytes are encoded between 0 to 255 already; Then record this mapping in a lookup table; The size of
                the lookup table is a hyperparameter, also called vocabulary size (vocab_size).
            3. Repeat the process to continually merging the most frequent pairs; Stop when no further compression is possible
            4. (For decoding) to restore the original text, reverse the process by substituting each ID with its corresponding pair, using
                the lookup table.

        Assignment 1 Requirements:
            1. encode() only loop over merges that matter.
            2. Detect and preserve special tokens <|endoftext|>
            3. Use pre-tokenization (e.g., the GPT-2 tokenizer regex)
            4. Try to make the implementation as fast as possible

        Args:
            input_path (str | os.PathLike): Path to the BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary, including special tokens.
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be kept as a single token.

        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
                vocab:
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes). vocab is used for speedy lookup and transform token into IDs.
                merges:
                    BPE merges. Each list item is a tuple of bytes (<token1>, <token2>). merges are used for dissecting texts into tokens
        """

        raise NotImplementedError

    def _train(self, sentence: str, vocab_size: int, special_tokens: list[str] = ["<|endoftext|>"]):

        assert vocab_size > 255

        # GPT-2 pre-tokenization: Replace space with "Ġ"
        # E.g., "Hello World" might be tokenized as ["Hello", "ĠWorld"]


        raise NotImplementedError

if __name__ == '__main__':

    ...


