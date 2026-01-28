from tokenizer import Tokenizer
from typing import List, Dict, Tuple, Set, Union, Iterator, Any, Optional
from typing import Iterator, Iterable
from typing import BinaryIO
from dataclasses import dataclass
from os import PathLike
from collections import defaultdict, Counter
import os

# from functools import total_ordering

import regex as re
import heapq

from patterns import GPT_2_PATTERN as gpt2_pattern

@dataclass
class BPECorpusStats:
    sequence: List[bytes]
    sequence_id: List[int]
    count: int = 0

class BPETokenizer(Tokenizer):
    
    def __init__(self,
        vocab_size: int = 30000,
        special_tokens: List[str]|None=['<|endoftext|>',]
    ):
        super().__init__(vocab_size, special_tokens)

        self.merges: List[Tuple[int, int]] = list()             # Record for merges
        self.pair_freq: Dict[Tuple[int, int], int] = Counter()  # Frequency of pairs
        self.corpus: Dict[bytes, BPECorpusStats] = dict()         # Corpus

    def train(
        self,
        corpus: Dict[str, int]
    ):
        pass

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
        if word not in self.inverse_vocab: raise ValueError("{} not in vocab".format(word))
        return self.inverse_vocab[word]

    def _assemble_string(self, pair: Tuple[int, int]) -> bytes:
        """Assemble string based on given pair and return the assembled string"""
        return self._lookup(pair[0]) + self._lookup(pair[1])

    def _train(self, file_path: PathLike|str):

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
                self.pair_freq[pair] += 1   # Update pair_freq
                mapping[pair].add(k)        # update mapping

        # Now we've built all mappings, let's start merging
        # First we can implement a priority queue to find the frequent pair
        # Here we initialize the priority queue
        heapq_freq = [(-count, self._assemble_string(pair), pair) for pair,count in self.pair_freq.items()]
        heapq.heapify(heapq_freq)

        while self.get_vocab_size <= self.vocab_size and heapq_freq:
            # TODO: Changes the way how heapq pop elements
            # Lazy deletion
            
            pair: Tuple[int, int] = (-1, -1) # For initialization only.
            while heapq_freq:
                heapq_element = heapq.heappop(heapq_freq)           # New pair to be updated
                pair = heapq_element[-1]
                if self.pair_freq[pair] == -heapq_element[0]:
                    break
                
            self.merges.append(pair)                                # Update merges
            self._update_vocab(b''.join(map(self._lookup, pair)))    # Update vocab with merged pair


            # Merging:
            #+Make changes based on mapping. i.e. We need to :
            #+Update the corpus
            #+New pairs emerge
            #+Adjacent pairs might needs to be changed! So update it.
            #+Update pair_freq
            #+Update mapping
            #+Update priority queue
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
                        delta[old_pair] = count_n - count       # pairs to be changed
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
                in_str = self._assemble_string(pair)
                if count: heapq.heappush(heapq_freq, (-count, in_str, pair))

            # Clean up
            freq_changed_pairs.clear()
                

    def _merge_pair(self, pre_token: bytes, pair: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Merge new pair in pre_token and returns ...
        """
        target = self.corpus[pre_token]
        _merged = b''.join(map(self._lookup, pair))
        
        # Debug: Make sure we've added the new merged pair into vocabulary
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
        corpuse: Dict[str, int] = defaultdict(int)
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
        cleaned_text = re.sub(pattern, '', text)
        
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
    bpe = BPETokenizer(10000)
    bpe._initialize_vocab()
    bpe._train('tests/fixtures/tinystories_sample_5M.txt')
    print(bpe.vocab)
    print(bpe.merges)
        

