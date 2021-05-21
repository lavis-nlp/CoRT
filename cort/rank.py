from functools import partial
from pprint import pformat
import argparse
import logging
import time
from pathlib import Path


import torch
from tqdm import tqdm

from cort.dataloading import PreProcessor
from cort.model import CortModel
from cort.utils import (
    init_root_logger,
    load_qrels,
    save_rankings,
    batched_collection_loader,
)
from cort.eval import calc_ranking_metrics
from cort.index import CortGPUIndex

encode_times, search_times, total_times, iterations = 0, 0, 0, 0

log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_dir", type=str)
    parser.add_argument("queries", type=str)
    parser.add_argument("-o", "--out_file", default="ranking.tsv")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-d", "--device", default="0")
    parser.add_argument("--query_encoder", type=str, default=None,
                        help="Use a query encoder other than the one in index_dir" )
    parser.add_argument("--rank_len", default=1000, type=int)
    parser.add_argument("--qrels_file", default=None, type=str)
    parser.add_argument("--strict_model_loading", action="store_true")
    parser.add_argument("--log_file", default="logs/rank.log", type=str)
    args = parser.parse_args()
    init_root_logger(
        loglevel=logging.INFO,
        file=args.log_file,
    )
    rank_impl(**vars(args))


def rank_impl(
    index_dir,
    queries,
    out_file,
    query_encoder,
    device,
    batch_size,
    rank_len,
    qrels_file,
    strict_model_loading,
    **kw_args
):
    torch.set_grad_enabled(False)
    if device == "cpu":
        devices = ["cpu"]
    else:
        devices = ["cuda:{}".format(x) for x in device.split(",")]

    model_file = query_encoder or Path(index_dir)/"model.pt"
    log.info(f"loading model {Path(model_file).resolve()} (strict={strict_model_loading})")
    model, tokenizer = CortModel.from_file(
        model_file, devices[0], strict=strict_model_loading
    )
    model.eval()

    log.info("loading index")
    cort_index = CortGPUIndex.from_dir(index_dir)
    cort_index.setup(devices)

    log.info("starting retrieval")
    all_rankings = perform_ranking(
        model, cort_index, queries, rank_len, batch_size, tokenizer
    )

    log.info("time stats:\n" + pformat(calc_time_stats(batch_size)))

    if qrels_file is not None:
        log.debug("computing evaluation metrics")
        qrels = load_qrels(qrels_file)
        ranking_list, qrels_list = zip(
            *[(all_rankings[qid], qrels[qid]) for qid in qrels]
        )
        log.info(
            "Evaluation results:\n"
            + pformat(calc_ranking_metrics(ranking_list, qrels_list, to_tensor=False))
        )

    out_file = Path(out_file)
    log.info("writing rankings to file {}".format(out_file.resolve()))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_rankings(all_rankings, out_file)


def perform_ranking(
    model: CortModel,
    cort_index: CortGPUIndex,
    query_file,
    rank_len,
    batch_size,
    tokenizer,
):
    all_rankings = {}
    preprocessor = PreProcessor()
    prog_bar = tqdm()
    batch_gen = batched_collection_loader(query_file, batch_size)

    for qid_batch, text_batch in batch_gen:
        processed_text = preprocessor.prep(text_batch)

        tokenized_batch = tokenizer.batch_encode_plus(
            processed_text,
            truncation=True,
            max_length=512,
            padding=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        rankings = rank_batch(tokenized_batch, cort_index, model, rank_len)
        all_rankings.update({k: v for k, v in zip(qid_batch, rankings)})
        update_prog_bar(prog_bar, batch_size)
    prog_bar.close()
    return all_rankings


def update_prog_bar(prog_bar: tqdm, batch_size):
    time_stats = calc_time_stats(batch_size)
    prog_bar.set_postfix(time_stats)
    prog_bar.update(batch_size)


def calc_time_stats(batch_size):
    factor = (1000 / batch_size) / iterations
    time_stats = {
        "enc_time": "{:.1f}ms/query".format(encode_times * factor),
        "search_time": "{:.1f}ms/query".format(search_times * factor),
        "total_time": "{:.1f}ms/query".format(total_times * factor),
    }
    return time_stats


def rank_batch(batch, index: CortGPUIndex, model: CortModel, rank_len):
    t = time.time()
    batch = torch.tensor(batch, device=model.device, dtype=torch.long)

    sync_device = (
        None
        if index.index[0].device in ("cpu", torch.device("cpu"))
        else index.index[0].device
    )

    # encode query batch
    query_embeddings, enc_time = measure_time(
        partial(model.forward, token_type=1), (batch,), sync_device
    )
    global encode_times
    encode_times += enc_time

    # search
    search_args = (query_embeddings, rank_len)
    (scores, indices), search_time = measure_time(
        index.search, search_args, sync_device
    )
    global search_times
    search_times += search_time

    # map pids
    looked_up_indices = torch.take(index.lookup, indices.to(dtype=torch.int64))
    rankings = [r.tolist() for r in looked_up_indices]
    global total_times
    total_times += time.time() - t
    global iterations
    iterations += 1
    return rankings


def measure_time(fn, args, sync_device=None):
    if sync_device:
        torch.cuda.synchronize(sync_device)
    t = time.time()
    res = fn(*args)
    if sync_device:
        torch.cuda.synchronize(sync_device)
    return res, time.time() - t


if __name__ == "__main__":
    main()
