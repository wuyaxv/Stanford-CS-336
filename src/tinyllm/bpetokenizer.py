from tinyllm.tokenizer import Tokenizer
from tinyllm.patterns import GPT_2_PATTERN as gpt2_pattern
from utils.utils import timeit

from typing import List, Dict, Tuple, Set
from typing import BinaryIO
from dataclasses import dataclass
from os import PathLike
from collections import defaultdict, Counter
import os
import regex as re
import heapq
import concurrent.futures
import multiprocessing
from functools import total_ordering

@dataclass
class BPECorpusStats:
    sequence: List[bytes]
    sequence_id: List[int]
    count: int = 0

@dataclass
@total_ordering
class HeapqPair:
    count: int
    pair: Tuple[int, int]
    pair_bytes: tuple[bytes, bytes]

    def __lt__(self, other):
        # inverse comparison
        if self.count == other.count: 
            return self.pair_bytes > other.pair_bytes
        else: return self.count > other.count

class BPETokenizer(Tokenizer):
    
    def __init__(self,
        vocab_size: int = 30000,
        special_tokens: List[str]|None=['<|endoftext|>',]
    ):
        super().__init__(vocab_size, special_tokens)

        self.merges: List[Tuple[bytes, bytes]] = list()             # Record for merges
        self.pair_freq: Dict[Tuple[int, int], int] = Counter()      # Frequency of pairs
        self.corpus: Dict[bytes, BPECorpusStats] = dict()           # Corpus

        self._initialize_vocab()

    def train(self, input_path: str|PathLike, **kwargs) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        if 'vocab_size' in kwargs:
            self.vocab_size = kwargs['vocab_size']
        if 'special_tokens' in kwargs:
            self.special_tokens = kwargs['special_tokens']

        self._train(input_path)
        return self.vocab, self.merges


    def encode(self):
        ...

    def decode(self):
        ...

    def _initialize_vocab(self):
        for i in range(256):
            self._update_vocab(i.to_bytes())

        for st in self.special_tokens if self.special_tokens else []:
            self._update_vocab(st.encode('utf-8'))

    def _update_vocab(self, new_word: bytes):
        """Update Vocabulary"""
        if new_word not in self.inverse_vocab:
            new_id = self.get_vocab_size
            self.vocab.update({new_id: new_word})           # Update vocab
            self.inverse_vocab.update({new_word: new_id})   # Update inverse_vocab

    def _lookup(self, _id: int) -> bytes:
        if _id not in self.vocab: raise ValueError("{} not in vocab".format(_id))
        return self.vocab[_id]

    def _inverse_lookup(self, word: bytes) -> int:
        if word not in self.inverse_vocab: raise ValueError("{} not in inverse_vocab".format(word))
        return self.inverse_vocab[word]

    def _assemble_string(self, pair: Tuple[int, int]) -> bytes:
        """Assemble string based on given pair and return the assembled string"""
        return self._lookup(pair[0]) + self._lookup(pair[1])

    @timeit
    def _train(self, file_path: PathLike|str):
        """Training code"""

        """Serial processing code...
        # Read training corpus, corpus are in Dict[str, BPECorpusStats]
        self._read_training_corpus(file_path)

        # Now we've finished reading all of the corpus, let's start merging
        # We need to build all of the pairs using `inverse_vocab` and track each pair
        #+To track each pair, we need to add a mapping for each pair

        mapping: Dict[Tuple[int, int], Set[bytes]] = defaultdict(set)

        # initialize mapping
        for k,v in self.corpus.items():
            new_pairs = [(self._inverse_lookup(a), self._inverse_lookup(b)) for a,b in zip(v.sequence[:-1], v.sequence[1:])]    # Each items in new_pairs is Tuple[int, int]
            for pair in new_pairs:
                self.pair_freq[pair] += v.count   # Update pair_freq
                mapping[pair].add(k)              # update mapping
        """

        mapping: Dict[Tuple[int, int], Set[bytes]] = defaultdict(set)

        self.pair_freq, mapping, self.corpus = self._preparation(file_path)

        # Now we've built all mappings, let's start merging
        # First we can implement a priority queue to find the frequent pair
        # Here we initialize the priority queue
        heapq_freq = [HeapqPair(count, pair, tuple(map(self._lookup, pair))) for pair,count in self.pair_freq.items()]
        heapq.heapify(heapq_freq)

        index = 0

        while self.get_vocab_size < self.vocab_size and heapq_freq:

            pair: Tuple[int, int] = (-1, -1)
            while heapq_freq:
                # Lazy deletion
                heapq_element = heapq.heappop(heapq_freq)
                pair = heapq_element.pair
                if self.pair_freq[pair] == heapq_element.count:
                    break
            index += 1
                
            self.merges.append((self._lookup(pair[0]), self._lookup(pair[1])))  # Update merges
            self._update_vocab(b''.join(map(self._lookup, pair)))               # Update vocab with merged pair

            to_delete = defaultdict(set)
            to_add    = defaultdict(set)
            freq_changed_pairs = set()

            for pre_token in mapping[pair]:
                factor = self.corpus[pre_token].count
                old_pairs, new_pairs = self._merge_pair(pre_token, pair)    # Get old_pairs and new_pairs for comparison

                old_pairs_dict = defaultdict(int)
                new_pairs_dict = defaultdict(int)
                delta = defaultdict(int)

                for k in set(old_pairs):
                    old_pairs_dict[k] += old_pairs.count(k)
                for k in set(new_pairs):
                    new_pairs_dict[k] += new_pairs.count(k)

                for old_pair,count in old_pairs_dict.items():
                    if old_pair in new_pairs_dict:
                        count_n = new_pairs_dict[old_pair]
                        delta[old_pair] = count_n - count                   # pairs to be changed
                    else:
                        # old pair is not in new pair no more
                        delta[old_pair] = -count
                        to_delete[old_pair].add(pre_token)

                
                for new_pair,count in new_pairs_dict.items():
                    if new_pair not in old_pairs_dict: 
                        delta[new_pair] = count
                        to_add[new_pair].add(pre_token)

                # Update pair_freq
                for _pair,_delta in delta.items():
                    self.pair_freq[_pair] += _delta * factor
                    if self.pair_freq[_pair] == 0: del self.pair_freq[_pair]
                    else: freq_changed_pairs.add(_pair)

            # Update mapping
            for old_pair,pre_tokens in to_delete.items():
                for pre_token in pre_tokens: 
                    mapping[old_pair].remove(pre_token)
                if not len(mapping[old_pair]): del mapping[old_pair]

            for new_pair,pre_tokens in to_add.items():
                for pre_token in pre_tokens:
                    mapping[new_pair].add(pre_token)

            # Update priority queue
            for pair in freq_changed_pairs:
                count = self.pair_freq[pair]
                in_bytes = self._assemble_string(pair)
                if count: heapq.heappush(heapq_freq, HeapqPair(count, pair, tuple(map(self._lookup, pair))))

            # Clean up
            freq_changed_pairs.clear()

    def _save_vocab(self, file_path: PathLike|str):
        ...

    def _save_merges(self, file_path: PathLike|str):
        ...

    def _preparation(self, file_path: PathLike|str):
        """General Initialization, This function does following things(in parallel):
        1. Read training corpuses
        2. do pre-tokenization for each read chunk
        3. update pair to bytes mapping
        4. update pair_freq mapping
        """
        mapping: Dict[Tuple[int, int], Set[bytes]] = defaultdict(set)
        pair_freq: Dict[Tuple[int, int], int] = Counter()  # Frequency of pairs
        corpus: Dict[bytes, BPECorpusStats] = dict()         # Corpus


        number_of_processes = 4 if not os.cpu_count() else os.cpu_count()

        with open(file_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, number_of_processes, b"<|endoftext|>")

        futures = []
        queue = multiprocessing.Manager().Queue()
        
        with concurrent.futures.ProcessPoolExecutor(number_of_processes) as executor:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                futures.append(executor.submit(self._preparation_worker, file_path, start, end, queue))
            
        for future in concurrent.futures.as_completed(futures):
          try:
              future.result()  # 获取结果，会抛出工作进程中的异常
          except Exception as e:
              print(f"Worker failed: {e}")

        while not queue.empty():
            _pair_freq, _mapping, _corpus = queue.get()

            for pair,freq in _pair_freq.items():
                pair_freq[pair] += freq

            for pair,bytes_set in _mapping.items():
                mapping[pair].update(bytes_set)

            corpus.update(_corpus)

        return pair_freq, mapping, corpus



    def _preparation_worker(self, file_path, start: int, end: int, q):
        # corpus: Dict[bytes, BPECorpusStats] = {}
        # pair_freq: Dict[Tuple[int, int], int] = Counter()
        # mapping: Dict[Tuple[int, int], Set[bytes]] = defaultdict(set)
        corpus = {}
        pair_freq = Counter()
        mapping = defaultdict(set)

        with open(file_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end-start).decode('utf-8', errors='ignore')
            _corpus = self._pre_tokenization(chunk)

            for k,v in _corpus.items():
                if k not in self.corpus:
                    corpus[k] = BPECorpusStats([i.to_bytes() for i in k], 
                                                    [self._inverse_lookup(c.to_bytes()) for c in k],
                                                    count=v)
                else:
                    corpus[k].count += v
            
            for k,v in corpus.items():
                new_pairs = [tuple(map(self._inverse_lookup, (a, b))) for a,b, in zip(v.sequence[:-1], v.sequence[1:])]
                for pair in new_pairs:
                    pair_freq[pair] += v.count
                    mapping[pair].add(k)

        q.put((pair_freq, mapping, corpus))     # final prodcut is (pair_freq, mapping)


    def _merge_pair(self, pre_token: bytes, pair: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Merge new pair in pre_token and returns ...
        Not Parallelizable...
        """
        target = self.corpus[pre_token]
        _merged = b''.join(map(self._lookup, pair))
        
        assert _merged in self.inverse_vocab 

        _merged_id = self._inverse_lookup(_merged)

        # Before merging happens, calcuate all old pairs
        old_pairs = [(self._inverse_lookup(a), self._inverse_lookup(b)) for a,b in 
                     zip(target.sequence[:-1], target.sequence[1:])]

        i = 0
        while i < len(target.sequence_id)-1:
            if pair[0] == target.sequence_id[i] and pair[1] == target.sequence_id[i+1]:
                target.sequence_id[i] = _merged_id
                target.sequence_id.pop(i+1)
                i += 1
            else:
                i += 1

        # Merge sequence
        target.sequence = [self._lookup(i) for i in target.sequence_id]

        # After merging, calculate all new pairs
        new_pairs = [(self._inverse_lookup(a), self._inverse_lookup(b)) for a,b in 
                     zip(target.sequence[:-1], target.sequence[1:])]

        return old_pairs, new_pairs
        

    def _read_training_corpus(self, file_path: PathLike|str):
        """Read training file and return an iterator of corpus"""
        corpus: Dict[bytes, int] = defaultdict(int)
        with open(file_path, 'rb') as f:
            num_processes = 4
            boundaries = self._find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end-start).decode("utf-8", errors="ignore")

                # Run pre-tokenization process on each chunk.
                corpus = self._pre_tokenization(chunk)
                for k,v in corpus.items():
                    if k not in self.corpus:
                        self.corpus[k] = BPECorpusStats([i.to_bytes() for i in k], 
                                                        [self._inverse_lookup(c.to_bytes()) for c in k],
                                                        count=v)
                    else:
                        self.corpus[k].count += v

    def _pre_tokenization(self, text: str) -> Dict[bytes, int]:

        corpus: Dict[bytes, int] = defaultdict(int)
        
        # Remove all special token inside current chunk
        pattern = '|'.join([re.escape(st) for st in self.special_tokens])
        pattern = re.compile(pattern)

        splitted_chunk = pattern.split(text)
        
        for cleaned_text in splitted_chunk:
            for token in re.finditer(gpt2_pattern, cleaned_text):
                corpus[token.group(0).encode('utf-8')] += 1

        return corpus

    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
    
        chunk_size = file_size // desired_num_chunks
    
        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size
    
        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
    
                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
    
                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size
    
        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

if __name__ == '__main__':
    from pathlib import Path

    root = Path('../../data')
    bpe = BPETokenizer(1000)

    bpe._train(root / 'TinyStoriesV2-GPT4-train.txt')
    

