from pathlib import Path
from typing import Tuple

import numpy as np
import torch


class CortGPUIndex:
    def __init__(self, vectors, id_lookup):
        self.vectors = vectors
        self.lookup = id_lookup
        self.index = None

    @staticmethod
    def from_dir(index_dir, embedding_size="auto"):
        index_path = Path(index_dir)
        vec_file = index_path / "vectors.npy"
        id_file = index_path / "ids.npy"

        if embedding_size == "auto":
            embedding_size = vec_file.stat().st_size // id_file.stat().st_size

        with open(vec_file, "rb") as r:
            x = np.frombuffer(r.read(), dtype=np.float32)
            x = x.reshape(-1, embedding_size)
        with open(id_file, "rb") as r:
            lookup = np.frombuffer(r.read(), dtype=np.int32)
        return CortGPUIndex(x.copy(), lookup.copy())

    def setup(self, devices):
        """
        Splits the index on the given devices and sets sets it up for searching.
        @param devices: sequence of device strings (e.g. ['cuda:0', 'cuda:1'].
         The first device will be used as aggregation device during multi-gpu search
        """
        if len(devices) > 1:
            split_size = int(np.ceil(len(self.vectors) / len(devices)))
            split_start_indexes = range(0, len(self.vectors), split_size)
            index_splits = [
                torch.tensor(
                    self.vectors[start : start + split_size], device=devices[i]
                )
                for i, start in enumerate(split_start_indexes)
            ]
        else:
            index_splits = [torch.tensor(self.vectors, device=devices[0])]
        # normalize to norm 1 to perform search with dot-product
        for split in index_splits:
            torch.div(split, split.norm(dim=1).view(-1, 1), out=split)
        self.index = index_splits
        self.lookup = torch.tensor(self.lookup, dtype=torch.int32, device=devices[0])

    def search(self, queries, k=1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determines the search method and maps the result-indices on the document-ids
        @param queries: query batch, already encoded
        @param k: Number of retrieved documents
        @return: tuple of tensors, scores and document ids
        """

        search_method = self._search_multi if len(self.index) > 1 else self._search_single
        scores, indices = search_method(queries, k)
        looked_up_indices = torch.take(self.lookup, indices)
        return scores, looked_up_indices

    def _search_single(self, queries, k) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.einsum(
            "ne,be->bn", self.index[0], queries.to(device=self.index[0].device)
        )
        return scores.topk(k, sorted=True)

    def _search_multi(self, queries, k) -> Tuple[torch.Tensor, torch.Tensor]:
        offset = len(self.index[0])
        agg_device = self.index[0].device

        # Perform scoring on each split. Works as simple as that due to cuda's async behavior <3
        results = [
            torch.einsum(
                "ne,be->bn", index_split, queries.to(device=index_split.device)
            ).topk(k)
            for index_split in self.index
        ]

        # aggregate the results from the splits (only the top-k) on agg_device
        scores, indices = zip(*results)
        scores = [s.to(device=agg_device) for s in scores]

        # choose the top-k among the results from the splits based on the scores
        final_scores, local_indices = torch.cat(scores, dim=1).topk(k, sorted=True)

        # translate indices back to original positions as if the search index was not splitted
        original_indices = torch.cat(
            [
                indices[x].to(device=agg_device) + x * offset
                for x in range(0, len(self.index))
            ],
            dim=1,
        )
        final_indices = torch.gather(original_indices, 1, local_indices)

        return final_scores, final_indices
