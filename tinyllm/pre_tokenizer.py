import regex as re
from abc import ABC, abstractmethod

from typing import Union, Optional, TextIO, BinaryIO, Any
import io # io.TextIOBase, io.BufferedIOBase
import os # os.PathLike


class PreTokenizer(ABC):

    @abstractmethod
    def pre_tokenizer(self):
        raise NotImplementedError

class GPT2PreTokenizer(PreTokenizer):

    PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    def __init__(self, \
            source: str | os.PathLike | TextIO | BinaryIO, \
            special_tokens: list[str|bytes] | None = None, \
            ):
        '''GPT2 pre-tokenizer

        初始化处理器，支持多种输入类型
        '''

        self.source = source
        self.pat = re.compile(self.PATTERN.encode('utf-8'))
        self.special_tokens = special_tokens

    @property
    def text(self):
        """This is a getter method that acts as a generator
        and yields a new bytearray.
        """

        if isinstance(self.source, str):
            yield self.source.encode('utf-8').rstrip()
        elif isinstance(self.source, bytes):
            yield self.source.rstrip()
        elif isinstance(self.source, os.PathLike):
            with open(self.source, 'rb') as f:
                for line in f:
                    yield line.rstrip()
        elif isinstance(self.source, io.TextIOBase):
            for line in self.source:
                yield line.encode('utf-8').rstrip()
        elif isinstance(self.source, io.BufferedIOBase):
            for line in self.source:
                yield line.rstrip()
        else:
            raise ValueError("支持的类型是字符串、目录或者文件io")

    def pre_tokenizer(self):

        for t in self.text:
            t = self._remove_special_tokens(t)
            tokens = re.findall(self.pat, t)
            for token in tokens:
                yield token

    def _remove_special_tokens(self, text):
        """Remove the special tokens inside the text
        
        Args:
            text (bytes): text to be processed.
        """
        assert isinstance(text, bytes) # Text must be bytes

        stripped: bytes = text
        if self.special_tokens:
            for st in self.special_tokens:
                if isinstance(st, bytes):
                    stripped = stripped.replace(st, b'')
                elif isinstance(st, str):
                    stripped = stripped.replace(st.encode('utf-8'), b'')
                else:
                    raise ValueError('Special Tokens对象错误')

        return stripped


if __name__ == '__main__':

    import pathlib

    text: str
    special_tokens = ["<|", "|>"]
    p = pathlib.Path('Sampletext')

    gpt2_pt = GPT2PreTokenizer(p, special_tokens)
    for t in gpt2_pt.pre_tokenizer():
        print(t)




        
        

        


    
