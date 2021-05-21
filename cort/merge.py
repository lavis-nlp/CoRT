from pprint import pformat
import argparse
import logging
from itertools import chain, zip_longest
from pathlib import Path

from tqdm import tqdm

from cort.utils import (
    init_root_logger,
    load_trec_rankings,
    save_rankings,
    load_qrels,
)
from cort.eval import calc_ranking_metrics

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rankings_a", type=str)
    parser.add_argument("rankings_b", type=str)
    parser.add_argument("-o", "--out_file", type=str, default="merged.tsv")
    parser.add_argument("--maxrank", default=1000, type=int)
    parser.add_argument("--qrels_file", type=str, default=None)
    parser.add_argument("--log_file", default="logs/merge.log", type=str)
    args = parser.parse_args()

    init_root_logger(
        loglevel=logging.INFO,
        file=args.log_file,
    )

    merge_impl(**vars(args))


def merge_impl(rankings_a, rankings_b, out_file, maxrank, qrels_file, **kw_args):
    log.info("Loading rankings...")
    ra = load_trec_rankings(rankings_a)
    rb = load_trec_rankings(rankings_b)
    log.info("Interleaving...")
    rankings = interleave_rankings(ra, rb, maxrank=maxrank)
    log.info(f"Saving merged ranking to {Path(out_file).resolve()}")
    save_rankings(rankings, out_file)

    if qrels_file is not None:
        log.info("Evaluating...")
        qrels = load_qrels(qrels_file)
        ranking_list, qrels_list = zip(*[(rankings[qid], qrels[qid]) for qid in qrels])
        log.info(
            "\tResults:\n"
            + pformat(calc_ranking_metrics(ranking_list, qrels_list, to_tensor=False))
        )


def interleave_rankings(rankings_a, rankings_b, maxrank):
    merge_rank = {}
    not_found = []
    for k in tqdm(rankings_a.keys() | rankings_b.keys()):
        new_ranking = []
        dup_check_set = set()
        if k not in rankings_a or k not in rankings_b:
            not_found.append(k)
        ranks_a = rankings_a.get(k, [])[:maxrank]
        ranks_b = rankings_b.get(k, [])[:maxrank]
        for x in chain(*zip_longest(ranks_a, ranks_b)):
            if x is not None and x not in dup_check_set:
                new_ranking.append(x)
                dup_check_set.add(x)
                if len(new_ranking) >= maxrank:
                    break
        merge_rank[k] = new_ranking
    if len(not_found):
        logging.warning(
            "{} qids not found in one of the rankings".format(len(not_found))
        )
    return merge_rank


if __name__ == "__main__":
    main()
