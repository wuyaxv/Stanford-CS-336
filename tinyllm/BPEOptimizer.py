from dataclasses import dataclass
from sortedcontainers import SortedKeyList, SortedDict
from typing import Generator, TextIO, BinaryIO
from collections import defaultdict

from pre_tokenizer import GPT2PreTokenizer

import io
import os
from pprint import pprint

class NoMaximumPairError(Exception):
    pass

@dataclass
class MergedPair:
    count: int # Merged pair count in frequency_table
    in_bytes: bytes # Byte format of the MergedPair
    indices: list[tuple[int, ...]] # Stores the mapping to the frequency_table

    def _key(self):
        return (self.count, self.in_bytes)

class BPETokenizer:

    def __init__(self, \
            source: str | bytes | os.PathLike | io.TextIOBase | io.BufferedIOBase, \
            special_tokens: list[str] | list[bytes] | None = None, \
            target_vocab_size: int = 5000, \
            chunk_size: int = 4096):
        """Handle the necessary operations

        遍历frequency_table时，确保每次将对应的映射关系存放到merged_pair中
        """
        self.source = source
        self.special_tokens = special_tokens if special_tokens else []

        self.vocab: dict[int, bytes] = self._init_vocab()# f: id -> bytes
        self.inverse_vocab = {k:v for v,k in self.vocab.items()}# f: bytes -> id
        self.frequency_table: dict[tuple[int, ...], int] = defaultdict(int) # The frequency table. It's global and has to be maintained during the whole time.
        self.merged_pairs: dict[tuple[int, int], MergedPair] = dict()# e.g. {(120, 135): [8, [(...), (...), ...]], ...}
        self.maximum_merged_pairs: list = [0, []] # e.g. [8, [(113, 125), (114, 178)]
        self.sorted_merged_pairs: dict[int, list[tuple[int, int]]] = dict()
        self.chunk_size = chunk_size
        self._target_vocab_size = target_vocab_size

    def encode(self):
        ...

    def decode(self):
        ...

    def train(self):
        self._train()

    def debug_iter(self, infomation:str):
        states = self._get_stat()
        print('-'*20+infomation+'-'*20)
        for k,v in states.items():
            print(k+":",end='')
            pprint(v)
        print('-'*20+infomation+'-'*20)

    def _get_stat(self):
        return dict(
        # vocab=self.vocab, 
        freq_t=self.frequency_table,
        merg_p=self.merged_pairs, 
        sorted_merge_list=self.sorted_merged_pairs, 
        maxmum_pair=self._get_the_maximum_pair())

    def _train(self):
        """Train a BPE Tokenizer

        Corpuse are from @property text
        """

        for t in self.text: # Go through the corpus and generate the frequency table
            self._init_frequency_table(t) # Generate frequency_table, merged_pairs, maximum_merged_pairs
        self._update_frequency_table()

    @property
    def target_vocab_size(self):
        return self._target_vocab_size

    @target_vocab_size.setter
    def target_vocab_size(self, size: int):

        assert size >= len(self.special_tokens)+256

        self._target_vocab_size = size

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

    def _inverse_lookup(self, w:bytes) -> int:
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
        return len(self.vocab)

    def _update_vocab(self, pair: tuple[int, int]):
        """Update the vocabulary based on integer pairs,
            those integers in the pair must be in the vocab already
        """
        
        new_pair_bytes = self._assemble_bytes(pair)
        new_id = self.vocab_size # New id is the end of queue of current vocab
        self.vocab[new_id] = new_pair_bytes
        self.inverse_vocab[new_pair_bytes] = new_id # Update the inverse vocab

    def _is_in_vocab(self, k: int | bytes):
        if isinstance(k, int):
            if k in self.vocab: return True
            else: return False
        else:
            if k in self.inverse_vocab: return True
            else: return False

    def _init_frequency_table(self, text: str | bytes | os.PathLike | TextIO | BinaryIO):
        """
        初始化 frequency table
        """
        gpt2_pt = GPT2PreTokenizer(text, self.special_tokens)

        for t in gpt2_pt.pre_tokenizer(): # 获取的token首先将其按字节拆分
            assert isinstance(t, bytes)

            new_k:tuple[int, ...] = tuple([self._inverse_lookup(t[i:i+1]) for i in range(len(t))]) # e.g. new_k = (112, 145, 187, ...), the indices of freq table are tuples of integers
            if new_k in self.frequency_table:
                self.frequency_table[new_k] += 1
            else:
                self.frequency_table[new_k] = 1 # Add new entry

            self._update_merged_table_during_init_frequency_table(new_k) # Update merged table during initialization of the frequency table

    def _update_merged_table_during_init_frequency_table(self, dissected_bytes: tuple[int, ...]):
        """Using newly dissected bytes and build the merged table
        """

        for pair in zip(dissected_bytes, dissected_bytes[1:]):
            # Update merged_pairs with newly zipped pairs
            if pair not in self.merged_pairs:
                self.merged_pairs[pair] = self._create_a_new_merged_pair(pair, dissected_bytes)
            else:
                self._update_a_merged_pair(pair, dissected_bytes)

    def _create_a_new_merged_pair(self, pair: tuple[int, int], index: tuple[int, ...], count: int=1) -> MergedPair:
        """创建一个新的二元元组，用于作为merged_pairs的键来统计词对频次

        Args:
            pair (tuple)。新merged_pair的键，主要用于生成MergedPair的in_bytes表示形式
            index (tuple)，新创建的二元元组在frequency_table对应的索引
            count (int) = 1
        """
        
        if count not in self.sorted_merged_pairs: self.sorted_merged_pairs[count] = [] # init the counting dict
        self.sorted_merged_pairs[count].append(pair) # Add new entry to the counting dict

        in_bytes = self._lookup(pair[0])+self._lookup(pair[1])
        return MergedPair(count=count, in_bytes=in_bytes, indices=[index])

    def _update_a_merged_pair(self, pair: tuple[int, int], index: tuple[int, ...], delta: int=1):
        """This function updates self.merged_pairs and dynamically updates the maximum merged pairs list
        By default, this function adds one to the self.merged_pairs count

        Args:
            pair (tuple): current merged pair for updating the merged_pairs dictionary
            index (tuple): index to the frequency_table
            delta (int): changes to the merged_pairs count, might below zero cause updating adjacent pairs
        """
        assert pair in self.merged_pairs
        assert index
        assert delta != 0

        before_update:int = self.merged_pairs[pair].count

        if delta > 0: # Add a new index to the merged_pair
            self.merged_pairs[pair].count += delta
            if index not in self.merged_pairs[pair].indices: self.merged_pairs[pair].indices.append(index)
            self._update_sorted_merged_pairs(before_update, self.merged_pairs[pair].count, pair) # update the sorting table
        else: # Delete a index from merged pair. delta < 0, rm the index and decrease the count
            assert self.merged_pairs[pair].count + delta >= 0
            self.merged_pairs[pair].count += delta
            del self.merged_pairs[pair].indices[
                self.merged_pairs[pair].indices.index(index)
            ] # Remove the current index from the merged_pairs
            if len(self.merged_pairs[pair].indices) == 0: # if the last one is deleted, clean it up.
                assert self.merged_pairs[pair].count == 0
                del self.merged_pairs[pair]
                self._update_sorted_merged_pairs(before_update, 0, pair)
            else: # update the sorting table
                assert self.merged_pairs[pair].count > 0
                self._update_sorted_merged_pairs(before_update, self.merged_pairs[pair].count, pair)

    def _update_sorted_merged_pairs(self, old_idx: int, new_idx: int, pair: tuple[int, int]):
        """Update the sorted list that tracks the order of merged_pairs
        """

        # self.debug_iter('_update_sorted_merged_pairs')
        assert old_idx in self.sorted_merged_pairs
        assert pair in self.sorted_merged_pairs[old_idx]
        assert (pair not in self.sorted_merged_pairs[new_idx]) if new_idx in self.sorted_merged_pairs else new_idx not in self.sorted_merged_pairs

        # Delete the old index and clean it
        del self.sorted_merged_pairs[old_idx][
            self.sorted_merged_pairs[old_idx].index(pair)
        ]
        if len(self.sorted_merged_pairs[old_idx]) == 0:
            del self.sorted_merged_pairs[old_idx]

        # Create the new entry
        if new_idx not in self.sorted_merged_pairs: self.sorted_merged_pairs[new_idx] = []
        self.sorted_merged_pairs[new_idx].append(pair)
    
    def _get_the_maximum_pair(self) -> tuple[int, tuple[int, int]]:
        max_idx = sorted(self.sorted_merged_pairs.keys())[-1]
        max_pair = sorted(self.sorted_merged_pairs[max_idx], \
            key = lambda p: self._lookup(p[0])+self._lookup(p[1]), \
            reverse = True
        )
        
        if max_idx != 0: 
            # print("I'm throwing the max value", max_pair[0], "from", self.sorted_merged_pairs)
            return max_idx, max_pair[0]
        else: raise NoMaximumPairError

    def _update_frequency_table(self):
        """We've got all the merged pair table, now use it to update frequency_table

        Args: 
            target_vocab_size (int): The target to hit for the vocabulary size.
        """
        assert self.target_vocab_size >= len(self.special_tokens) + 256

        count = 0
        while self.vocab_size <= self.target_vocab_size:
            count+=1
            """objective: find the lexicographically largest pair and update frequency_table, merged_table and the maximum_merged_pairs table
            """
            max_idx, maximum_pair = self._get_the_maximum_pair()
            breakpoint()
            
            # Update vocab
            new_id = self.vocab_size # The newly updated vocab id is the length the previous vocab
            self._update_vocab(maximum_pair)
            #self._update_maximum_pairs(maximum_pair)

            # Merge the pair in frequency table
            for index in self.merged_pairs[maximum_pair].indices:
                count = self.frequency_table[index]
                new_index = self._update_adjacent_pairs(maximum_pair, index, new_id) # handle all merged pairs' frequency change and throw a new index pair
                self.frequency_table[new_index] = count # Update the frequency table with new index and delete the old one
                del self.frequency_table[index] # Delete the old frequency table index and replace it with the new one but maintain the frequency

            # Now the dynamics of maximum_pairs has changed, we need to recaculate the new ones
            # Now we need to update the maximum_pair
            breakpoint()
            del self.sorted_merged_pairs[max_idx][
                self.sorted_merged_pairs[max_idx].index(maximum_pair)
            ]
            self.sorted_merged_pairs[0].append(maximum_pair)

            # Now we need to update the merged_pair of maximum_pair
            del self.merged_pairs[maximum_pair]

    def _update_adjacent_pairs(self, pair_to_merge:tuple[int, int], t: tuple[int, ...], new_id: int) -> tuple[int, ...]:
        """Update the frequency table index and change the merged_pairs at the same time
        """
        assert len(t) >= 2 # Other wise there shouldn't be anything to merge, something must went wrong in the caller function.
        new_idx_l = []
        new_idx: tuple[int, ...]
        abs_delta = self.frequency_table[t] # the absolute delta value is the count value of the current frequency table entry
        pairs_to_decrease = []
        pairs_to_increase = []

        if len(t) == 2:
            assert t == pair_to_merge
            new_idx_l.append(new_id)
            pairs_to_decrease.append(pair_to_merge)
        
        else:
            if t[:2] == pair_to_merge: 
                new_idx_l.append(new_id)
                pairs_to_decrease.append(t[1:3])
                pairs_to_increase.append((new_id, t[2]))
            else:
                new_idx_l.append(t[0])
            i = 1
            while i < len(t)-2:
                if t[i:i+2] == pair_to_merge:
                    new_idx_l.append(new_id)
                    pairs_to_decrease.append(t[i-1:i+1]) # update adjacent pairs frequency
                    pairs_to_decrease.append(t[i+1:i+3])
                    pairs_to_increase.append((t[i-1], new_id))
                    pairs_to_increase.append((new_id, t[i+2]))
                    i += 2
                else:
                    new_idx_l.append(t[i])
                    i += 1
            if i == len(t)-2: # 说明循环中的最后一对没有合并
                if t[-2:] == pair_to_merge: # last pair to merge
                    new_idx_l.append(new_id)
                    pairs_to_decrease.append(t[-3:-1])
                    pairs_to_increase.append((t[-3],new_id))
                else:
                    new_idx_l.append(t[-2])
                    new_idx_l.append(t[-1])
            else:
                new_idx_l.append(t[-1])

        new_idx = tuple(new_idx_l) # build the new index
        
        # Update merged pairs' frequency and indices
        for p2d in pairs_to_decrease:
            self._update_a_merged_pair(p2d, t, -abs_delta)
            if p2d in self.merged_pairs:
                del self.merged_pairs[p2d].indices[
                    self.merged_pairs[p2d].indices.index(t)
                ]
                self.merged_pairs[p2d].indices.append(new_idx)

        for p2i in pairs_to_increase:
            if p2i not in self.merged_pairs:
                self.merged_pairs[p2i] = self._create_a_new_merged_pair(p2i, new_idx, +abs_delta)
            else:
                self._update_a_merged_pair(p2i, new_idx, +abs_delta)
                del self.merged_pairs[p2i].indices[
                    self.merged_pairs[p2i].indices.index(t)
                ]
                self.merged_pairs[p2i].indices.append(new_idx)
        
        # 还需要更新zip中的其他pair的index

        return new_idx

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
        bpe._init_frequency_table(p)
        print(bpe.debug_iter('hello'))
        print(bpe._update_frequency_table())
        
