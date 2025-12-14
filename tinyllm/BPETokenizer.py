from typing import Optional, Any
from typing import List, Dict, Tuple, TextIO, BinaryIO
from typing import Generator
from typing import TextIO, BinaryIO

from dataclasses import dataclass
from functools import total_ordering
from collections import defaultdict
from abc import ABC, abstractmethod

from pre_tokenizer import GPT2PreTokenizer

import io
import os
import heapq
from pprint import pprint

class Tokenizer(ABC):
    @abstractmethod
    def train(self, input_path, vocab_size, special_tokens): ...
    
    @abstractmethod
    def encode(self): ...

    @abstractmethod
    def decode(self): ...

@dataclass
@total_ordering
class BPEHeapElement:
    pair: Tuple[int, int]
    count: int
    in_bytes: int

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        else:
            return self.in_bytes > other.in_bytes

    def __eq__(self, other):
        return self.count == other.count \
                and self.in_bytes == other.in_bytes

    def __repr__(self):
        return 'BPEHeapElement({}, {})'.format(
                    self.count,
                    self.in_bytes)

class BPETokenizer(Tokenizer):
    def __init__(self, vocab_size: int = 5000, special_tokens: Optional[List[str]] = None):

        self.target_vocab_size = vocab_size
        self.special_tokens: List[str] = special_tokens or []
        self.vocab: Dict[int, bytes]
        self.inverse_vocab: Dict[bytes, int]
        self._source: str

        self.merges: List[Tuple[int, int]] = []

        self.word_freq: Dict[bytes, int]
        self.word_decoded: Dict[bytes, List[int]]
        self.pair_freq: Dict[Tuple[int, int], int]
        self.pair_freq_heap: List[BPEHeapElement]
        self.pair_to_words: Dict[Tuple[int, int], List[bytes]]

        # Initialization of the vocabulary and the inverse vocabulary
        # self.vocab = self._init_vocab()
        # self.inverse_vocab = {b:i for i,b in self.vocab.items()}
        self.word_freq = defaultdict(int)
        self.word_decoded = defaultdict(list)
        self.pair_freq = defaultdict(int)
        self.pair_to_words = defaultdict(list)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, path: str|os.PathLike):
        self._source = path

    # Updates
    def _update_vocab(self, pair: Tuple[int, int]) -> int:
        new_idx = self.vocab_size
        new_pair_bytes = self._assemble_bytes(pair)
        self.vocab[new_idx] = new_pair_bytes
        self.inverse_vocab[new_pair_bytes] = new_idx

        return new_idx

    def _update_merges(self, new_pair: Tuple[int, int]):
        self.merges.append(new_pair)

    def _update_word_freq(self, token: bytes, delta: int = 1):
        self.word_freq[token] += delta

    def _add_word_decoded(self, token: bytes):
        if token not in self.word_decoded:
            self.word_decoded[token] = []
            for b in token:
                self.word_decoded[token].append(self._inverse_lookup(bytes([b])))
    
    def _init_pair_freq(self, pair: Tuple[int, int], delta: int):
        assert delta > 0
        if pair not in self.pair_freq:
            self.pair_freq[pair] = delta
        else:
            self.pair_freq[pair] += delta

    def _init_pair_freq_heap(self):

        self.pair_freq_heap = []
        for pair in self.pair_freq:
            new_heap_item = BPEHeapElement(
                    pair = pair,
                    count = self.pair_freq[pair],
                    in_bytes = self._assemble_bytes(pair)
                    )
            heapq.heappush(self.pair_freq_heap, new_heap_item)

    def _update_pair_freq(self, pair: Tuple[int, int], delta: int):
        """Update the pair frequency, if not exsits then create the index"""
        if pair not in self.pair_freq and delta > 0: # Creating a new entry for the new pair
            self.pair_freq[pair] = delta
        elif delta < 0 and pair not in self.pair_freq:
            raise ValueError("pair不存在")
        else:
            self.pair_freq[pair] += delta
            if self.pair_freq[pair] < 0: raise ValueError("pair频率小于0，出现异常")
            else:
                if self.pair_freq[pair] == 0: # 说明此时pair已经完全被合并了，没有再可能继续被合并了
                    del self.pair_freq[pair]
        
        # 更新pair_freq的最大堆排序
        if pair in self.pair_freq and self.pair_freq[pair] > 0: # 只要其发生变动并且还存在与列表中，就需要重新排序
            new_heap_item = BPEHeapElement(pair=pair, count=self.pair_freq[pair], in_bytes=self._assemble_bytes(pair))
            heapq.heappush(self.pair_freq_heap, new_heap_item)

    def _update_pair_to_words(self, pair, word):
        if pair in self.pair_to_words:
            if word not in self.pair_to_words[pair]: self.pair_to_words[pair].append(word) # Add a new word to the pair to word index # 这里我只添加了新的，没有删除旧的
        else:
            self.pair_to_words[pair] = [word] # Create a new index for the pair if not exist

    def _pop_the_largest_pair(self) -> Tuple[int, int]:
        """将heap中的最大的pair弹出，如果发现弹出的内容和pair_freq中的结果
        中的计数不匹配，说明数据过时，跳过重新弹出
        """
        while True:
            poped_value = heapq.heappop(self.pair_freq_heap)
            p = poped_value.pair
            if p in self.pair_freq and self.pair_freq[p] == poped_value.count: # 必须确保弹出的pair和当前的pair_freq匹配，否则说明数据过时
                return p

    def _merge_pair_and_get_adjacent_pairs(self, new_idx: int, pair_to_merge: Tuple[int, int], token: bytes) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int]:
        """Search and find the adjacent pairs in word_decoded table while merge the largest pair that's newly been found.

        Return:
            old_adjacent_pairs (list) adjacent pairs whose frequency to be updated(decreased) and new adjacent pairs whose frequency to be updated(increased)
            new_adjacent_pairs (list)
            matched (int)
        """

        old_adjacent_pairs = []
        new_adjacent_pairs = []
        new_id_list = []
        id_list = self.word_decoded[token]
        id_list_len = len(id_list)
        i: int
        matched: int = 0

        if id_list_len < 2: raise ValueError("长度小于2（0或1）的token ID序列不可能存在需要合并的字节")
        if id_list_len == 2:
            if pair_to_merge == tuple(id_list):
                matched += 1
                new_id_list = [new_idx]
        else:
            if tuple(id_list[:2]) == pair_to_merge: 
                matched += 1
                new_id_list.append(new_idx)
                if tuple(id_list[2:4]) != pair_to_merge:
                    new_adjacent_pairs.append((new_idx, id_list[2]))
                    old_adjacent_pairs.append(tuple(id_list[1:3]))
                i = 2
            else:
                new_id_list.append(id_list[0])
                i = 1

            while i < id_list_len-2:
                if tuple(id_list[i:i+2]) == pair_to_merge:
                    matched += 1
                    new_id_list.append(new_idx)
                    old_adjacent_pairs.append(tuple(id_list[i-1:i+1])) # Add the adjacent pairs of two
                    new_adjacent_pairs.append((new_id_list[-2], new_idx))
                    if tuple(id_list[i+2:i+4]) != pair_to_merge:
                        new_adjacent_pairs.append((new_idx, id_list[i+2]))
                        old_adjacent_pairs.append(tuple(id_list[i+1:i+3]))
                    i += 2
                else:
                    new_id_list.append(id_list[i])
                    i += 1
                
            if i == id_list_len-1: # 说明最后一组被归并了，且此时右边的相邻的pair已经被添加
                new_id_list.append(id_list[-1])
            else: # 否则最后一组没有归并，此时需要将最后一组进行检测
                if tuple(id_list[-2:]) == pair_to_merge: # 最后一组可以归并
                    matched += 1
                    new_id_list.append(new_idx)
                    old_adjacent_pairs.append(tuple(id_list[-3:-1]))
                    new_adjacent_pairs.append((new_id_list[-2], new_idx))
                else:
                    new_id_list.append(id_list[-2])
                    new_id_list.append(id_list[-1])

        assert token in self.word_decoded
        self.word_decoded[token] = new_id_list      # Update the word_decoded list
        del self.pair_to_words[pair_to_merge][self.pair_to_words[pair_to_merge].index(id_list)]
        # 这里我还要更新self.pair_to_word，否则的话pair_to_word记录的仍然是旧的word_record

        return old_adjacent_pairs, new_adjacent_pairs, matched
    
    def encode(self):
        ...

    def decode(self):
        ...

    def train(self, input_path: str, vocab_size: int, special_tokens: List[str]|None = None) -> Tuple[Dict[int, bytes], List[tuple[int, int]]]:
        self._train(input_path, vocab_size, special_tokens)

        # replace space with 
        return self.vocab, self.merges

    def _train(self, input_path: str, vocab_size: int, special_tokens: List[str]):
        self.target_vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = self._init_vocab()
        self.inverse_vocab = {b:i for i,b in self.vocab.items()}
        
        for chunk in self._get_chunk(input_path):

            gpt2_pt = GPT2PreTokenizer(chunk, self.special_tokens)
            token_generator = gpt2_pt.pre_tokenizer()
            
            # Initial build of all the tables:
            # i.e. word_freq, word_decoded, pair_freq, pair_to_words
            for token in token_generator:
                self._update_word_freq(token, 1)
                self._add_word_decoded(token)
                new_pair_bytes = zip(token, token[1:]) # generate new pairs
                for npb in new_pair_bytes:
                    np = (self._inverse_lookup(bytes([npb[0]])), self._inverse_lookup(bytes([npb[1]])))
                    self._init_pair_freq(np, 1)         # Update the pair_freq table
                    self._update_pair_to_words(np, token) # Update the pair_to_word lookup table

            self._init_pair_freq_heap() # 初始化优先队列

            # Generate vocab based on pair frequency
            while self.vocab_size <= self.target_vocab_size and len(self.pair_freq) > 0:
                largest_pair: Tuple[int, int]
                old_adjacent_pairs: List[Tuple[int, int]]
                new_adjacent_pairs: List[Tuple[int, int]]
                count: int
                new_id: int

                largest_pair = self._pop_the_largest_pair() # pop the largest pair
                count = self.pair_freq[largest_pair]        # frequency of the largest pair
                accumulate = 0 # for debug purpose the pair_freq should equal the sum of the word counts

                self._update_merges(largest_pair)                  # Track the merged pairs history
                new_id = self._update_vocab(largest_pair)          # Add new pair to the vocabulary and get the new id

                word_l = self.pair_to_words[largest_pair]           # Get the word list of the updated pair for updating the frequency of adjacent pairs

                # Now its time to update the word table
                for w in word_l:
                    word_count = self.word_freq[w] # Get the word count to calc the delta for updating the pair_freq of adjacent pairs.
                    old_adjacent_pairs, new_adjacent_pairs, matched = \
                            self._merge_pair_and_get_adjacent_pairs(new_id,
                                                                    largest_pair, # pair_to_merge
                                                                    w,            # token to be searched in the word_decomposed
                                                                    )
                    self._update_pair_freq(largest_pair, -matched * word_count) # Decrease the largest pair's frequency
                    accumulate += word_count*matched # Debug
                    for o_adj_p in old_adjacent_pairs:
                        self._update_pair_freq(o_adj_p, -word_count)
                    for n_adj_p in new_adjacent_pairs:
                        self._update_pair_freq(n_adj_p, word_count)
                        self._update_pair_to_words(n_adj_p, w)                  # update hte pair to words for the newly created adjacent words

                assert accumulate == count

    def _init_vocab(self):
        """Initialize vocabulary using special tokens and bytes from 0 to 255
        """
        vocab = {}
        count = 0
        for st in self.special_tokens:
            vocab[count] = st.encode('utf-8')
            count += 1
        for i in range(256):
            vocab[count] = bytes([i])
            count += 1
        return vocab
    
    def _get_chunk(self, input_path: str|os.PathLike, chunk_size=4096) -> Generator[bytes, None, None]:
        self.source = input_path
        return self._read_from_source(chunk_size=chunk_size)

    def _lookup(self, i:int) -> str:
        """Lookup the vocabulary table by ID
        """

        return self.vocab[i]

    def _inverse_lookup(self, w:str) -> int:
        """Reverse the vocabulary table by bytes and returns the respective ID
        """

        return self.inverse_vocab[w]

    def _assemble_bytes(self, pair: Tuple[int, int]) -> str:
        """Take a pair of IDs and return their assembled bytes

        Args:
            pair (tuple[int, int])

        Returns:
            Assembled bytes (bytes)
        """
        return self._lookup(pair[0])+self._lookup(pair[1])

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

if __name__ == '__main__':
    import pathlib
    input_path = pathlib.Path('Sampletext')
    special_tokens = ['<|endoftext|>']
    vocab_size = 50000

    bpe = BPETokenizer()
    print(bpe.train(input_path, vocab_size, special_tokens))

