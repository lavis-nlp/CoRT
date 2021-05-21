import logging
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from random import choices, choice
import re
from typing import Union, Sequence

import msgpack
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class PreProcessor:
    def __init__(self):
        self._invalid_char_pattern = re.compile(r"[^a-zA-Z\d\.,\!\?;\/:]+")

    def prep(self, text: Union[str, Sequence[str]]):
        if isinstance(text, str):
            return self._invalid_char_pattern.sub(" ", text).strip()
        elif isinstance(text, Sequence):
            return [self._invalid_char_pattern.sub(" ", t).strip() for t in text]
        else:
            raise ValueError("wrong type")


class CollectionManager:
    """Loads and tokenizes a collection. This manager uses a cache to avoid unnecessary re-tokenization for
    re-occurring combinations of collection abd tokenizer used. Tokenization here means, each token is mapped
    on an (unsigned) integer

    Due to the binary encoding of tokenized sequences (using np.ndarray.bytes), this manager is memory efficient.
    It can run in 16-bit mode (to save even more memory) when there are less then 2^15 unique tokens to map on.

    Keyword arguments:
    original_file -- Path to the original .tsv collection
    tokenizer -- Tokenizer that converts passages to list of ids. Can be None if loading from cache
    tokenizer_name -- tokenizer identifier, needed to cache correctly. Should contain #dim
    cache_dir -- directory that will be used for caching
    """

    def __init__(
        self,
        original_file,
        tokenizer,
        tokenizer_name,
        cache_dir="./loader_cache/",
        int16=False,
        overwrite=False,
        qrels_file=None,
        sort_by_len=False,
        max_tokens=512,
    ):

        self.log = logging.getLogger("CoRT." + self.__class__.__name__)
        self.dtype = np.int16 if int16 else np.int32
        self.max_tokens = max_tokens
        self._handle_cache(
            cache_dir,
            int16,
            max_tokens,
            original_file,
            overwrite,
            tokenizer,
            tokenizer_name,
        )

        if sort_by_len:
            self.log.info("sorting collection by sequence length")
            self.passage_ids, _ = zip(
                *sorted(self.passages.items(), key=lambda x: len(x[1]))
            )
        else:
            self.passage_ids = list(self.passages.keys())

        self.qrels = load_qrels(qrels_file) if qrels_file else None

    def _handle_cache(
        self,
        cache_dir,
        int16,
        max_tokens,
        original_file,
        overwrite,
        tokenizer,
        tokenizer_name,
    ):
        cache_name = "{}.{}.max{}.int{}.msg".format(
            Path(original_file).name,
            tokenizer_name.replace("/", "_"),
            max_tokens,
            "16" if int16 else "32",
        )
        cache_file = Path(cache_dir) / cache_name
        if cache_file.exists() and not overwrite:
            self.log.info(str(cache_file) + " cached - loading cache...")
            with open(cache_file, "rb") as r:
                self.passages = msgpack.load(r)
        else:
            self.log.info(str(cache_file) + " not cached - tokenizing from scratch...")
            assert tokenizer is not None
            if not Path(cache_dir).exists():
                os.mkdir(cache_dir)
            self.passages = self._load_collection(original_file, tokenizer)
            with open(cache_file, "wb") as w:
                msgpack.pack(self.passages, w)

    @staticmethod
    def _tokenize_worker(line, tokenizer, dtype=np.int32):
        splitted = line.strip().split("\t", 1)
        x = int(splitted[0])
        tokenized = tokenizer(splitted[1])
        return x, np.array(tokenized, dtype=dtype).tobytes()

    def _load_collection(self, fname, tokenizer, multi_processed=False) -> dict:
        passages = {}
        if multi_processed:
            with open(fname, "r", encoding="utf-8") as r, Pool(
                processes=os.cpu_count() // 2
            ) as pool:
                func = partial(
                    self._tokenize_worker, tokenizer=tokenizer, dtype=self.dtype
                )
                for pid, tokenized in tqdm(pool.imap_unordered(func, r, chunksize=512)):
                    passages[pid] = tokenized
        else:
            func = partial(self._tokenize_worker, tokenizer=tokenizer, dtype=self.dtype)
            with open(fname, "r", encoding="utf-8") as r:
                for pid, tokenized in tqdm(map(func, r)):
                    passages[pid] = tokenized
        return passages

    def __getitem__(self, key):
        if type(key) in (int, np.int32, np.int64):
            return torch.from_numpy(np.frombuffer(self.passages[key], self.dtype))
        if type(key) in (list, tuple, np.ndarray):
            return pad_sequence([self[i] for i in key], batch_first=True)
        raise NotImplementedError(type(key))

    def __contains__(self, key):
        return key in self.passages

    def __len__(self):
        return len(self.passages)

    def __iter__(self):
        return iter(self.passages)

    def items(self):
        return ((k, self[k]) for k in self.passages)

    def get_random_batch(self, batch_size) -> torch.Tensor:
        return self[choices(self.passage_ids, k=batch_size)]


class TripletDataset(torch.utils.data.Dataset):
    """Dataset for supervised training. Each sample consists of a positive, sampled from
    qrel_file, and a negative, sampled from scores_file. If qrel_only=False, a positive sample
    will be sample from scores_file as well when no qrel is available.
    """

    def __init__(
        self,
        qrel_file,
        scores_file,
        query_manager,
        collection_manager,
        minresults=10,
        ws_ranks=3,
        min_rank=5,
        max_rank=100,
        cache_dir="./loader_cache",
        overwrite=False,
        qrel_only=True,
        valid=False,
    ):
        super(TripletDataset, self).__init__()

        self.ws_ranks = ws_ranks
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.minresults = minresults
        self.valid = valid
        self.query_manager = query_manager
        self.collection_manager = collection_manager

        self.log = logging.getLogger("CoRT." + self.__class__.__name__)

        self._handle_cache(cache_dir, minresults, overwrite, scores_file)

        self.qrels = load_qrels(qrel_file) if qrel_file else None

        if self.qrels and qrel_only:
            self.rankings = [x for x in self.rankings if x[0] in self.qrels]

        self.log.info(
            "Created Dataset from scorefile '{}' and qrelfile '{}' with {} samples "
            "(qrel_only={})".format(scores_file, qrel_file, len(self), qrel_only)
        )

    def _handle_cache(self, cache_dir, minresults, overwrite, scores_file):
        cache_name = "{}.min{}.msg".format(Path(scores_file).name, minresults)
        cache_file = Path(cache_dir) / cache_name
        if cache_file.exists() and not overwrite:
            self.log.info(str(cache_file) + " cached - loading cache...")
            self.rankings = msgpack.load(open(cache_file, "rb"))
        else:
            self.log.info(str(cache_file) + " not cached - loading from scratch")
            self.rankings = self.load_rankings(scores_file)
            msgpack.pack(self.rankings, open(cache_file, "wb"))

    def __len__(self):
        return len(self.rankings)

    def __getitem__(self, i):
        qid, ranking = self.rankings[i]
        all_qrels = self.qrels[qid] if self.qrels and qid in self.qrels else []
        possible_negs = self._generate_negatives(all_qrels, qid, ranking)

        if self.valid:
            #  when validating, always chose the same pair
            neg = possible_negs[i % len(possible_negs)]
            pos = all_qrels[0] if all_qrels else choice(ranking[: self.ws_ranks])
        else:
            neg = choice(possible_negs)
            pos = choice(all_qrels) if all_qrels else choice(ranking[: self.ws_ranks])

        q_tokens = self.query_manager[qid]
        pos_tokens = self.collection_manager[pos]
        neg_tokens = self.collection_manager[neg]

        return q_tokens, pos_tokens, neg_tokens

    def _generate_negatives(self, all_qrels, qid, ranking):
        qrel_set = set(all_qrels)
        possible_negs = [
            p
            for p in ranking[max(self.min_rank - 1, 0): self.max_rank]
            if p not in qrel_set
        ]
        if len(possible_negs) == 0:
            self.log.debug(
                (
                    "negative sampling: min_rank could not be satisfied for query {}. "
                    "Taking the least instead."
                ).format(qid, self.min_rank)
            )
            possible_negs = [p for p in ranking if p not in qrel_set][-1:]
        return possible_negs

    def load_rankings(self, filename):
        rankings = load_msmarco_rankings(filename)
        rankings = [(k, v) for k, v in rankings.items() if len(v) >= self.minresults]
        return rankings

    @staticmethod
    def collate(batch):
        return [
            pad_sequence(x, batch_first=True).to(dtype=torch.long) for x in zip(*batch)
        ]


class TestDataset(torch.utils.data.Dataset):
    """
    Dataset for Testing. Each sample consists of a query and all available positive passages.
    """

    def __init__(self, qrel_file, query_manager):
        super(TestDataset, self).__init__()
        self.query_manager = query_manager
        self.log = logging.getLogger("CoRT." + self.__class__.__name__)
        self.qrels = list(load_qrels(qrel_file).items())

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, i):
        qid, qrel_set = self.qrels[i]
        return self.query_manager[qid], qid, qrel_set

    @staticmethod
    def collate(batch):
        q_tokens, qids, qrel_sets = zip(*batch)
        padded = pad_sequence(q_tokens, batch_first=True).to(dtype=torch.long)
        return padded, qids, qrel_sets


def load_qrels(qrel_file):
    qrel_tuple = (
        (int(q_id), int(p_id))
        for q_id, _, p_id, _ in map(str.split, open(qrel_file, "r"))
    )
    qrels = {}
    for q_id, p_id in qrel_tuple:
        if q_id not in qrels:
            qrels[q_id] = []
        qrels[q_id].append(p_id)
    return qrels


def load_msmarco_rankings(input_file):
    rankings = dict()
    with open(input_file, "r") as r:
        for l in tqdm(r):
            qid, pid, rank = (int(x) for x in l.split("\t"))
            if qid not in rankings:
                rankings[qid] = []
            rankings[qid].append(pid)
            assert len(rankings[qid]) == rank
    return rankings
