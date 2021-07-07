from pathlib import Path
import argparse
from functools import partial
import logging


import torch
from tqdm import tqdm
import numpy as np

from cort.dataloading import PreProcessor
from cort.model import CortModel
from cort.utils import batched_collection_loader, init_root_logger
from more_itertools import divide

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("collection", type=str)
    parser.add_argument("-o", "--out_dir", required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="per device batch-size")
    parser.add_argument("-d", "--device", default="0")
    parser.add_argument("--log_file", default="logs/encode.log", type=str)
    args = parser.parse_args()

    init_root_logger(
        loglevel=logging.INFO,
        file=args.log_file,
    )

    encode_impl(**vars(args))


def encode_impl(
    model,
    collection,
    out_dir,
    batch_size,
    device,
    token_type=0,
    **kwargs
):
    out_dir = _check_out_dir(out_dir)
    log.info(f"Index will be created in directory {out_dir.resolve()}")
    log.info("Initializing model and devices")
    devices = _setup_devices(device)
    models, tokenizer = _setup_model(model, devices)

    # determine encoder function (single/multi) and bind parameters
    if len(models)==1:
        log.info("Using a single GPU")
        encoder_function = partial(_encode_batch_single, model=models[0], token_type=token_type)
    else:
        log.info(f"Using {len(models)} GPUs")
        encoder_function = partial(_encode_batch_multi, models=models, token_type=token_type)

    _perform_encoding(encoder_function, collection, tokenizer, out_dir, batch_size * len(models))

    # save model for query encoding
    with (out_dir/"model.pt").open("wb") as f:
        torch.save(models[0].state_dict(), f)


def _perform_encoding(encoder_function, collection, tokenizer, out_dir, batch_size):
    preprocessor = PreProcessor()
    with open(out_dir / "vectors.npy", "wb") as vec_writer, \
         open(out_dir / "ids.npy", "wb") as id_writer:
        batch_gen = batched_collection_loader(collection, batch_size)
        for doc_ids, text in tqdm(batch_gen, unit_scale=batch_size):
            processed_text = preprocessor.prep(text)
            tokenized = tokenizer.batch_encode_plus(processed_text,
                                                    truncation=True,
                                                    max_length=512,
                                                    padding=True,
                                                    return_attention_mask=False,
                                                    return_token_type_ids=False, )["input_ids"]
            vecs = encoder_function(tokenized)
            ids = np.array(doc_ids, dtype=np.int32)
            vec_writer.write(vecs.tobytes())
            id_writer.write(ids.tobytes())


def _setup_model(checkpoint, devices):
    # strict=False to load models from transformers<=3.0.2
    model_tuples = [CortModel.from_file(checkpoint, device=d, strict=False) for d in devices]
    models, tokenizers = zip(*model_tuples)
    for m in models:
        m.eval()
    tokenizer = tokenizers[0]
    return models, tokenizer


def _setup_devices(device):
    torch.set_grad_enabled(False)
    if device == "cpu":
        devices = ["cpu"]
    else:
        devices = ["cuda:{}".format(x) for x in device.split(",")]
    return devices


def _check_out_dir(out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "vectors.npy").exists():
        raise Exception(f"{out_dir} already contains an index")
    return out_dir


def _encode_batch_single(tokenized_batch, model, token_type=0):
    x = model.forward(
        torch.tensor(tokenized_batch, dtype=torch.long, device=model.device),
        token_type=token_type,
    )
    vecs = x.cpu().numpy()
    return  vecs


def _encode_batch_multi(tokenized_batch, models, token_type=0):
    chunks = map(list, divide(len(models), tokenized_batch))
    forwarded = [model.forward(
        torch.tensor(tokenized_chunk, dtype=torch.long, device=model.device),
        token_type=token_type,
    ) for model, tokenized_chunk in zip(models, chunks) if tokenized_chunk]

    vecs = torch.cat([x.cpu() for x in forwarded]).numpy()
    return vecs


if __name__ == "__main__":
    main()
