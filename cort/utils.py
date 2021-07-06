import logging
from pathlib import Path
import argparse
from typing import List, Dict, Union
from io import IOBase
import sys

default_log_formatter = logging.Formatter(
    " %(asctime)s [%(levelname)-6s][%(name)s] %(message)s"
)


def init_root_logger(
    loglevel=logging.DEBUG,
    file=None,
    stream=True,
    log_formatter=default_log_formatter,
    file_log_level=logging.DEBUG,
) -> logging.Logger:
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(min(loglevel, file_log_level))

    if stream:
        ch = logging.StreamHandler()
        ch.setLevel(loglevel)
        ch.setFormatter(log_formatter)
        logger.addHandler(ch)

    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file)
        fh.setLevel(file_log_level)
        fh.setFormatter(log_formatter)
        logger.addHandler(fh)
        logger.info(f"logging to file {Path(file).resolve()}")

    logger.debug(f"call: {' '.join(sys.argv)}")

    return logger


def file_or_open(f: Union[str, Path, IOBase], mode="r"):
    file = open(f, encoding="utf-8", mode=mode) if isinstance(f, (str, Path)) else f
    assert file.mode == mode
    return file


def batched_collection_loader(file_path, batch_size):
    batch = ([], [])
    with open(file_path) as r:
        for line in r:
            pid, text = line.rstrip("\n").split("\t", maxsplit=1)
            pid = int(pid)
            batch[0].append(pid)
            batch[1].append(text)
            if len(batch[0]) >= batch_size:
                yield batch
                batch = ([], [])
    if len(batch[0]):
        yield batch


def load_qrels(qrel_file, qrel_format="trec"):
    """
    Loads QREL-File while assuming each line contains
    equally relevant pairs of QID and PIDS
    format 'trec':
        QID    UNUSED    PID     RELEVANCE
    format 'pairs':
        QID    PID
    """
    with file_or_open(qrel_file, "r") as r:
        p_id_idx = 1 if qrel_format == "pairs" else 2
        qrel_tuple = (
            (int(splitted[0]), int(splitted[p_id_idx]))
            for splitted in (line.rstrip().split("\t") for line in r)
        )
        qrels = {}
        for q_id, p_id in qrel_tuple:
            if q_id not in qrels:
                qrels[q_id] = set()
            qrels[q_id].add(p_id)
    return qrels


def save_rankings(rankings, out):
    with file_or_open(out, "w") as w:
        for qid, ranking in rankings.items():
            for i, pid in enumerate(ranking):
                w.write("{}\t{}\t{}\n".format(qid, pid, i + 1))


def load_trec_rankings(
    input_file,
) -> Dict[int, List[int]]:  # actually is msmarco format
    rankings = dict()
    with file_or_open(input_file, "r") as r:
        for l in r:
            splitted = l.rstrip().split("\t")
            qid, pid, rank = (int(splitted[x]) for x in range(3))
            if qid not in rankings:
                rankings[qid] = []
            rankings[qid].append(pid)
            assert len(rankings[qid]) == rank
    return rankings


def type_int_or_float(value):
    try:
        x = float(value)
        if x >= 2.0:
            return int(x)
        return x
    except Exception:
        raise argparse.ArgumentTypeError("must be int or float")


def type_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("must be boolean")
