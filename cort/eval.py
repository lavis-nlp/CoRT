import argparse
from pprint import pprint
from typing import List, Union, Dict, Sequence

import numpy as np
import torch

from cort.utils import load_trec_rankings, load_qrels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ranking_file", type=str)
    parser.add_argument("qrels_file", type=str)
    args = parser.parse_args()
    pprint(evaluate_impl(**vars(args)))


def evaluate_impl(
    ranking_file,
    qrels_file,
):
    rankings = load_trec_rankings(ranking_file)
    qrels = load_qrels(qrels_file)
    ranking_list, qrels_list = zip(*[(rankings[qid], qrels[qid]) for qid in qrels])
    results = calc_ranking_metrics(ranking_list, qrels_list, to_tensor=False)
    return results


def calc_ranking_metrics(
    rankings: Union[Dict, List],
    qrels: Union[Dict, List],
    recall_cuts=(100, 200, 1000),
    mrr_cuts=(10, 1000),
    ndcg_cuts=(20, 1000),
    count_rankings=True,
    to_tensor=True,
):
    if type(rankings) is type(qrels) is dict:
        rankings, qrels = zip(*[(rankings[qid], qrels[qid]) for qid in qrels])

    assert (
        isinstance(rankings, Sequence)
        and isinstance(qrels, Sequence)
        and len(rankings) == len(qrels)
    )
    results = dict()
    for ranking, query_qrels in zip(rankings, qrels):
        assert len(query_qrels) >= 1
        labels = np.array([1 if pid in query_qrels else 0 for pid in ranking])
        hits_rank = np.where(labels == 1)[0] + 1

        # MAP
        ap = np.sum(np.arange(1, len(hits_rank) + 1) / hits_rank) / len(query_qrels)
        if "map" not in results:
            results["map"] = []
        results["map"].append(ap)

        # RPREC
        rprec = np.mean(labels[: len(query_qrels)])
        if "rprec" not in results:
            results["rprec"] = []
        results["rprec"].append(rprec)

        # RECALL
        for cut in recall_cuts:
            if "recall@{}".format(cut) not in results:
                results["recall@{}".format(cut)] = []
            results["recall@{}".format(cut)].append(
                sum(labels[:cut]) / len(query_qrels)
            )

        # MRR
        for cut in mrr_cuts:
            if "mrr@{}".format(cut) not in results:
                results["mrr@{}".format(cut)] = []
            if len(hits_rank) and (hits_rank[0] <= cut):
                first_hit_rank = hits_rank[0]
                results["mrr@{}".format(cut)].append(1 / first_hit_rank)
            else:
                results["mrr@{}".format(cut)].append(0)

        # NDCG
        for cut in ndcg_cuts:
            if "ndcg@{}".format(cut) not in results:
                results["ndcg@{}".format(cut)] = []
            results["ndcg@{}".format(cut)].append(
                calc_ndcg(ranking[:cut], query_qrels, cut)
            )

    # calculate metric means
    results = {k: np.mean(v) for k, v in results.items()}

    # add the number of rankings if wanted
    if count_rankings:
        results["#rankings"] = len(rankings)

    # convert to tensors if necessary
    if to_tensor:
        results = {k: torch.tensor(v) for k, v in results.items()}

    return results


def calc_ndcg(ranking: list, qrels, p=10):
    if not hasattr(calc_ndcg, "idcgs"):
        calc_ndcg.idcgs = dict()
    if p not in calc_ndcg.idcgs:
        calc_ndcg.idcgs[p] = [
            sum(np.ones(nq) / np.log2(np.arange(1, nq + 1) + 1)) for nq in range(p + 1)
        ]
    rel = np.array([1 if pid in qrels else 0 for i, pid in enumerate(ranking[:p])])
    dcg = sum(rel / np.log2(np.arange(1, len(rel) + 1) + 1))
    return dcg / calc_ndcg.idcgs[p][min(len(qrels), p)]


if __name__ == "__main__":
    main()
