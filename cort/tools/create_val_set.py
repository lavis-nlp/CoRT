import argparse
import random


def main(source_qrels, exclude_queries_from, num_samples, out_qrels, query_file, out_queries, magic_number):

    source_qids = set()
    with open(source_qrels) as r:
        for line in r:
            source_qids.add(line.split(maxsplit=1)[0])

    with open(exclude_queries_from) as r:
        for line in r:
            qid = line.split(maxsplit=1)[0]
            if qid in source_qids:
                source_qids.remove(qid)

    random.seed(magic_number)
    valid_qids = set(random.sample(list(source_qids), k=num_samples))

    with open(query_file) as r, open(out_queries, "w") as w:
        for line in r:
            qid = line.split(maxsplit=1)[0]
            if qid in valid_qids:
                w.write(line)

    with open(source_qrels) as r, open(out_qrels, "w") as w:
        for line in r:
            qid = line.split(maxsplit=1)[0]
            if qid in valid_qids:
                w.write(line)
                valid_qids.remove(qid)  # only one per qid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_qrels",  type=str, default="data/qrels.dev.tsv",
                        help="qrels from which the query-ids for validation set will be sampled")
    parser.add_argument("--exclude_queries_from", type=str, default="data/qrels.dev.small.tsv",
                        help="queries that should not be used in the validation set")
    parser.add_argument("-n", "--num_samples", default=1000, type=int,
                        help="How many query the validation set should contain. "
                             "Must be smaller than number of source queries - excluded queries")
    parser.add_argument("--out_qrels", type=str, default="data/qrels.valid.tsv",
                        help="output file")
    parser.add_argument("--query_file", type=str, default="data/queries.dev.tsv",
                        help="source tsv file containing the queries")
    parser.add_argument("--out_queries", type=str, default="data/queries.valid.tsv",
                        help="file that contains the raw queries")
    parser.add_argument("--seed", default=42, help="rng seed")
    args = parser.parse_args()
    main(**vars(args))
