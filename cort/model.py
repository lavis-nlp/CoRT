from math import pi
import subprocess
import logging
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import transformers
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from tqdm import tqdm
from more_itertools import chunked
import numpy as np

from cort.eval import calc_ranking_metrics
from cort.dataloading import (
    CollectionManager,
    TripletDataset,
    TestDataset,
    PreProcessor,
)


class CortModel(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        super(CortModel, self).__init__()
        self.log = logging.getLogger("CoRT." + self.__class__.__name__)
        self.hparams = hparams
        self.loss_history = []
        self.index_tensor = self.pid_map = None

        self.transformer = self._load_transformer()
        self.f1 = nn.Linear(self.transformer.config.hidden_size, hparams.embedding_size)

    @classmethod
    def from_file(
        cls, file, device=None, strict=True
    ) -> Tuple["CortModel", PreTrainedTokenizerBase]:
        """
        Determines if file contains a lightning checkpoint or state_dict only and delegate respectively
        @return: Tuple of the Module and the corresponding tokenizer
        """
        loaded = torch.load(file, map_location=device)
        if "hyper_parameters" in loaded:
            return cls.from_ckpt(loaded, device)
        else:
            return cls.from_state_dict(loaded, device, strict=strict)

    @classmethod
    def from_ckpt(
        cls, ckpt, device, strict=True
    ) -> Tuple["CortModel", PreTrainedTokenizerBase]:

        if not isinstance(ckpt, dict):
            ckpt = torch.load(ckpt, map_location=device)

        module = cls(ckpt["hyper_parameters"])
        module = module.to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(module.hparams.pretrained_transformer)
        module.load_state_dict(ckpt["state_dict"], strict=strict)
        return module, tokenizer

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Union[str, Path, dict],
        device,
        hparams_update=None,
        strict=True,
    ) -> Tuple["CortModel", PreTrainedTokenizerBase]:

        # load state_dict if a path is given
        if not isinstance(state_dict, dict):
            state_dict = torch.load(state_dict, map_location=device)

        # infer embedding_size
        embedding_size = len(state_dict["f1.bias"])

        # get default params
        parser = ArgumentParser()
        parser = cls.add_model_specific_args(parser)
        hparams = parser.parse_args(["--embedding_size={}".format(embedding_size)])

        # update hparams if needed
        if hparams_update:
            hparams_dict = vars(hparams)
            hparams_dict.update(hparams_update)
            hparams = Namespace(**hparams_dict)

        module = cls(hparams)
        module = module.to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(module.hparams.pretrained_transformer)
        module.load_state_dict(state_dict, strict=strict)
        return module, tokenizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False,
                                formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument("--margin", type=float, default=0.1)
        parser.add_argument(
            "--pretrained_transformer", type=str, default="albert-base-v2"
        )
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--lr", type=float, default=2e-05)
        parser.add_argument("--embedding_size", type=int, default=128)
        parser.add_argument("--warmup_batches", type=int, default=2000)
        parser.add_argument("--weight_decay", type=float, default=0.1)
        parser.add_argument(
            "--index_dir",
            type=str,
            default="index",
            help="Root directory for indexes. Test runs will save "
            "created indexes here.",
        )
        parser.add_argument(
            "--use_test_index",
            type=str,
            default=None,
            help="Use specified document vectors for testing to avoid re-indexing. "
            "Only makes sense if the corresponding checkpoint is loaded "
            "for query encoding",
        )
        parser.add_argument(
            "--test_rank_len",
            type=int,
            default=1000,
            help="Length of compiled rankings for testing",
        )
        parser.add_argument(
            "--save_test_rankings",
            type=str,
            default=None,
            help="If not None, the  rankings ",
        )
        return parser

    def _load_transformer(self):
        pretrained_transformer = self.hparams.pretrained_transformer
        if not pretrained_transformer:
            raise Exception("no transformer identifier specified")
        return AutoModel.from_pretrained(pretrained_transformer)

    def forward(self, x, token_type=0):
        """
        The forward Step.
        @param x: Tensor of Batch X Input_Ids
        @param token_type: 0 for passages, 1 for queries
        @return: Embeddings - Batch X Embedding_size
        """
        attention_mask = (x != 0).to(dtype=torch.float)
        token_type_ids = torch.ones_like(x) if token_type == 1 else torch.zeros_like(x)
        x = self.transformer.forward(
            x, attention_mask=attention_mask, token_type_ids=token_type_ids
        )[0][:, 0]
        x = self.f1(x)
        return torch.tanh(x)

    def _loss_fn(self, query, positive, negative):
        """Processes a triplet of a query, a posiitve passage and a negative one to determine
        the corresponding loss value. Inputs should be tensors of embeddings"""
        passages = torch.cat((positive, negative))
        scores = torch.cosine_similarity(
            query.unsqueeze(1), passages.unsqueeze(0), dim=2
        )
        scores = torch.sub(1, torch.acos(scores) / pi)
        positives = torch.eye(*scores.shape).to(device=self.device)

        loss_raw = (
            (scores - scores.diag().unsqueeze(1) + self.hparams.margin)
            * torch.sub(1, positives)
        ).clamp_min(0)
        loss = loss_raw.sum()
        return loss

    def training_step(self, batch, batch_idx):
        query, positive, negative = batch
        forwarded_q = self.forward(query, token_type=1)
        forwarded_pos = self.forward(positive, token_type=0)
        forwarded_neg = self.forward(negative, token_type=0)

        loss = self._loss_fn(forwarded_q, forwarded_pos, forwarded_neg)
        self.logger.agg_and_log_metrics({"train/loss": loss.item()}, self.global_step)
        result = pl.TrainResult(minimize=loss)
        return result

    def validation_step(self, batch, batch_idx):
        query, positive, negative = batch
        forwarded_q = self.forward(query, token_type=1)
        forwarded_pos = self.forward(positive, token_type=0)
        forwarded_neg = self.forward(negative, token_type=0)

        loss = self._loss_fn(forwarded_q, forwarded_pos, forwarded_neg)

        pos_scores = torch.cosine_similarity(forwarded_q, forwarded_pos)
        neg_scores = torch.cosine_similarity(forwarded_q, forwarded_neg)
        accuracy = (pos_scores > neg_scores).to(dtype=torch.float32).mean()

        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        result.log("val/acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return result

    def on_test_epoch_start(self):
        if self.hparams.use_test_index:
            index_dir = Path(self.hparams.use_test_index)
            self.log.info(
                "loaded encoded vectors from {}".format(self.hparams.use_test_index)
            )
            with open(index_dir / "vectors.npy", "rb") as r:
                x = np.frombuffer(r.read(), dtype=np.float32)
                x = x.reshape(-1, self.hparams.embedding_size)
            self.index_tensor = torch.tensor(x).to(device=self.device)

            with open(index_dir / "ids.npy", "rb") as r:
                pids = np.frombuffer(r.read(), dtype=np.int32)
            self.pid_map = torch.tensor(
                pids.copy(), device=self.device, dtype=torch.int32
            )

        else:
            index_dir = (
                Path(self.trainer.default_root_dir)
                / self.hparams.index_dir
                / self.hparams.run_id
            )
            index_dir.mkdir(parents=True, exist_ok=True)
            collection: CollectionManager = self.trainer.datamodule.passages
            self.index_tensor = torch.zeros(
                len(collection),
                self.hparams.embedding_size,
                dtype=torch.float,
                device=self.device,
            )
            bs = self.hparams.batch_size * 2  # no triplets, so bs*2 should be safe
            logging.info("Encoding corpus and saving vectors to {}".format(index_dir))
            with torch.no_grad(), open(index_dir / "vectors.npy", "wb") as w:
                for i, pid_chunk in enumerate(
                    chunked(tqdm(collection.passage_ids), bs)
                ):
                    x = self.forward(
                        collection[pid_chunk].to(device=self.device, dtype=torch.long),
                        token_type=0,
                    )
                    self.index_tensor[i * bs : (i + 1) * bs] = x
                    w.write(x.cpu().numpy().tobytes())

            with open(index_dir / "ids.npy", "wb") as w:
                w.write(np.array(collection.passage_ids, dtype=np.int32).tobytes())
            self.pid_map = torch.tensor(
                collection.passage_ids, device=self.device, dtype=torch.int32
            )
            # save model for later query encoding
            with (index_dir / "model.pt").open("wb") as f:
                torch.save(self.state_dict(), f)

        # normalize index tensor thus dot product will suffice for ranking
        self.index_tensor /= self.index_tensor.norm(dim=1).view(
            len(self.index_tensor), 1
        )

    def test_step(self, batch, batch_idx):
        query_tokens, batch_qids, batch_qrels = batch
        queries = self.forward(query_tokens, token_type=1)
        scores, indices = torch.einsum("ae, be->ba", self.index_tensor, queries).topk(
            self.hparams.test_rank_len, sorted=True
        )
        rankings = [row.tolist() for row in torch.take(self.pid_map, indices)]

        if self.hparams.save_test_rankings:
            with open(self.hparams.save_test_rankings, "a") as w:
                for qid, ranking in zip(batch_qids, rankings):
                    for i, pid in enumerate(ranking):
                        w.write("{}\t{}\t{}\n".format(qid, pid, i + 1))

        metrics = calc_ranking_metrics(rankings, batch_qrels, count_rankings=False)
        metrics = {"test/" + k: v for k, v in metrics.items()}
        result = pl.EvalResult()
        result.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return result

    def on_before_zero_grad(self, optimizer):
        # log lr but only before zero_grad
        lr = optimizer.param_groups[0]["lr"]
        self.logger.log_metrics({"train/lr": lr}, self.global_step)
        self.trainer.add_progress_bar_metrics({"lr": lr, "step": self.global_step})

    def configure_optimizers(self):
        optimizer = transformers.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_steps = (
            self.hparams.warmup_batches // self.trainer.accumulate_grad_batches
        )
        total_steps = (
            self.trainer.datamodule.num_training_samples * self.trainer.max_epochs
        ) // self.trainer.accumulate_grad_batches

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]


class TSVDataModule(pl.LightningDataModule):
    def __init__(self, hparams, setup_val=True):
        super(TSVDataModule, self).__init__()
        self.hparams = hparams
        self.passages: Union[type(None), CollectionManager] = None
        self.train_queries = self.train_dataset = None
        self.val_queries = self.val_dataset = None
        self.test_queries = self.test_dataset = None
        self.setup_val = setup_val

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False,
                                formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument("--passage_file", type=str, default="./data/collection.tsv")
        parser.add_argument(
            "--train_negrank_file", type=str, default="./data/anserini.train.100.tsv"
        )
        parser.add_argument(
            "--train_query_file", type=str, default="./data/queries.train.tsv"
        )
        parser.add_argument(
            "--train_qrel_file", type=str, default="./data/qrels.train.tsv"
        )
        parser.add_argument(
            "--val_negrank_file", type=str, default="./data/anserini.valid.100.tsv"
        )
        parser.add_argument(
            "--val_query_file", type=str, default="./data/queries.valid.tsv"
        )
        parser.add_argument(
            "--val_qrel_file", type=str, default="./data/qrels.valid.tsv"
        )
        parser.add_argument(
            "--test_query_file", type=str, default="./data/queries.dev.small.tsv"
        )
        parser.add_argument(
            "--test_qrel_file", type=str, default="./data/qrels.dev.small.tsv"
        )
        parser.add_argument("--max_tokens", type=int, default=512)
        parser.add_argument(
            "--negative_min_rank",
            "--min_rank",
            dest="negative_min_rank",
            type=int,
            default=8,
        )
        parser.add_argument(
            "--negative_max_rank",
            "--max_rank",
            dest="negative_max_rank",
            type=int,
            default=100,
        )
        parser.add_argument("--loader_workers", type=int, default=2)
        return parser

    @staticmethod
    def build_tokenizer(pretrain_id) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(pretrain_id)

    def setup(self, stage=None):
        tok_id = self.hparams.pretrained_transformer
        tokenizer = text_pipeline(tok_id, max_tokens=self.hparams.max_tokens)

        if self.passages is None:
            self.passages = CollectionManager(
                self.hparams.passage_file,
                tokenizer,
                tok_id,
                int16=False,
                max_tokens=self.hparams.max_tokens,
            )

        if stage == "fit" or stage is None:
            # training data
            self.train_queries = CollectionManager(
                self.hparams.train_query_file,
                tokenizer,
                tok_id,
                int16=False,
                max_tokens=self.hparams.max_tokens,
            )
            self.train_dataset = TripletDataset(
                self.hparams.train_qrel_file,
                self.hparams.train_negrank_file,
                self.train_queries,
                self.passages,
                min_rank=self.hparams.negative_min_rank,
                max_rank=self.hparams.negative_max_rank,
            )

            if self.setup_val:
                # validation data
                self.val_queries = CollectionManager(
                    self.hparams.val_query_file,
                    tokenizer,
                    tok_id,
                    int16=False,
                    max_tokens=self.hparams.max_tokens,
                )
                self.val_dataset = TripletDataset(
                    self.hparams.val_qrel_file,
                    self.hparams.val_negrank_file,
                    self.val_queries,
                    self.passages,
                    valid=True,
                    min_rank=self.hparams.negative_min_rank,
                    max_rank=self.hparams.negative_max_rank,
                )

        if stage == "test" or stage is None:
            self.test_queries = CollectionManager(
                self.hparams.test_query_file,
                tokenizer,
                tok_id,
                int16=False,
                max_tokens=self.hparams.max_tokens,
            )
            self.test_dataset = TestDataset(
                self.hparams.test_qrel_file, self.test_queries
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.loader_workers,
            collate_fn=TripletDataset.collate,
            shuffle=True,
        )

    @property
    def num_training_samples(self):
        if self.train_dataset:
            return len(self.train_dataset) // self.hparams.batch_size
        else:
            return int(
                subprocess.check_output(
                    ["/usr/bin/wc", "-l", self.hparams.train_qrel_file]
                ).split()[0]
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.loader_workers,
            collate_fn=TripletDataset.collate,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size * 2,
            num_workers=self.hparams.loader_workers,
            collate_fn=TestDataset.collate,
            shuffle=False,
        )


def text_pipeline(pretrained_transformer, max_tokens=512):
    tok = AutoTokenizer.from_pretrained(pretrained_transformer)
    preprocessor = PreProcessor()

    def tokenizer(text):
        if type(text) is str:
            return tok.encode(
                preprocessor.prep(text), max_length=max_tokens, truncation=True
            )
        else:
            return [
                tok.encode(preprocessor.prep(t), max_length=max_tokens) for t in text
            ]

    return tokenizer
