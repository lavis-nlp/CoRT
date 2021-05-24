# CoRT: Complementary Rankings from Transformers
    
This repository contains code to repoduce the results of CoRT on the [MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) dataset.
CoRT is a simple neural first-stage ranking model that leverages contextual representations from pretrained language models (here: ALBERT via ) to complement term-based ranking models. Thus, CoRT increases the recall of re-ranking candidates resulting in improved re-ranking quality with less candidates.
The implementations in this repository are based on [_Pytorch_](https://github.com/pytorch/pytorch), [_Pytorch Lightning_](https://github.com/PyTorchLightning/pytorch-lightning) and [_HuggingFace's Transformers_](https://github.com/huggingface/transformers). We also use [Anserini](https://github.com/castorini/anserini) to produce BM25 rankings.
 
## Baseline Comparison

| MS MARCO passage (dev.small)   | MRR@10   | Recall@200 | Recall@1000 |
|--------------------------------|---------:|-----------:|------------:|
| BM25                           |     18.7 |       73.8 |        85.7 |
| [doc2query](https://github.com/nyu-dl/dl4ir-doc2query)                      |     21.5 |          - |        89.3 |
| [DeepCT](https://github.com/AdeDZY/DeepCT)                         |     24.3 |       81.0 |        90.9 |
| [docTTTTTquery](https://github.com/castorini/docTTTTTquery)                  |     27.7 |       86.9 |        94.7 |
| CoRT                           |     27.1 |       78.6 |        88.0 |
| CoRT + BM25                    |     27.4 |       87.3 |        94.9 |
| CoRT* + docTTTTTquery          | **28.8** |   **90.0** |    **96.5** |

*CoRT was trained on complementing BM25, but still adds value to other lexical approaches.  

## Getting started

To start, clone the repository, enter the created directory and install the package using pip

```
$ git clone https://github.com/mwrzalik/CoRT.git
$ cd CoRT && pip install -e .
```


### Prepare the Data

When using the default parameters, a directory `./data/` is expected in the working directory, containing the MS MARCO passage files (e.g. `collection.tsv` or `qrels.train.tsv`).
If you have already downloaded and extracted them somewhere it is most convenient to create a symlink to the 
corresponding directory. In the other case, download and extract the original MS MARCO passage data first:

```
$ mkdir data && cd data
$ wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
$ tar -zxvf collectionandqueries.tar.gz
```

To train a complementary ranker, it is necessary to provide example rankings from the ranking model that should
 be complemented. Hence, for each training query, the corresponding BM25 ranking needs to be
 provided. It's most convenient to download the required rankings together with other required files for validation and testing from the author:

```
$ wget http://lavis.cs.hs-rm.de/storage/cort/data/anserini.train.100.tsv \ 
    http://lavis.cs.hs-rm.de/storage/cort/data/queries.valid.tsv \
    http://lavis.cs.hs-rm.de/storage/cort/data/qrels.valid.tsv \
    http://lavis.cs.hs-rm.de/storage/cort/data/anserini.valid.100.tsv \
    http://lavis.cs.hs-rm.de/storage/cort/data/anserini.dev.small.tsv 
```

Otherwise, the example rankings (i.e. BM25 rankings) need to be compiled upfront. For validation we use a separated set from the MS MARCO dev queries, disjoint to _dev.small_. The corresponding files are included in the downloads above, but can also be created using the `create_val_set.py` script in `cort.tools`

### Start a Training Run

If all files were prepared as described above, the training can be started with default 
parameters. The size of embedded queries and passages can be set with `--embedding_size`. Lower embedding sizes speed up 
retrieval and reduce the memory footprint at the cost of retrieval quality. We recommend to use 
`--embedding_size=128`, which is default. As a `--metric_logger` either tensorboard or wanbd (needs the wandb client to be installed) can be specified .  To give your run a name, you can specify the `--run_id`. More training (hyper-) parameters can be found here [here](#train-a-model). A training run with default parameters can be started as follows

```
$ cort train --run_id=my_run --embedding_size=128 --metric_logger=tensorboard
```

On the very first run, the entire collection will be tokenized and cached to spare cpu resources during
 training, which may take a few hours. After that, the training will run for 10 epochs by default
 taking about 33 hours on a RTX 2080 Ti. After training, a test run will automatically be 
 started. The expensive part of testing is encoding the whole collection, which should take about 
 7 hours on a RTX 2080 Ti. The calculated passage embeddings will be saved for later use in the `./runs/index/<run_id>/` 
 directory. Depending on the embedding size, the corresponding files can occupy several gigabytes of
 disk storage. The actual retrieval and calculation of the evaluation metrics will be done after 1-2
 minutes. The evaluation metric results will be logged and saved be the metric logger.  
 
### View Metric Logs

By default, train and test metrics are logged into tensorboard files in the `runs/cort` directory. Hence,
`tensorboard --logdir runs/` will start a tensorboard instance showing the corresponding metrics.
 Alternatively, if `wandb` is installed and configured in your environment, logging can be switched 
 to *Weights & Biases* with `--metric_logger=wandb` as parameters for the training command.

## Inference

The follows describes Inference with this CoRT implementation. We currently do not 
support the ANN-retrieval described in the Paper. Retrieval is performed on one or more GPUs only.  

### Trained Models

The models we trained for our experiments can be downloaded and directly used to reproduce our results or try the model on an arbitrary collection. We have trained our models with various representation sizes (only affects the projection layer), which can be downloaded from the table below.

| Representation Size | Recall@200 (Standalone) | Recall@200 (+BM25) | Download |
|:--------------------:|------------------------:|-------------------:|:------|
| 32  |     70.6 |       85.2 | [cort.msmarco.e32.pt](http://lavis.cs.hs-rm.de/storage/cort/models/cort.msmarco.e32.pt)    |
| 64  |     75.6 |       86.2 | [cort.msmarco.e64.pt](http://lavis.cs.hs-rm.de/storage/cort/models/cort.msmarco.e64.pt)    |
| 128 |     77.2 |       87.0 | [cort.msmarco.e128.pt](http://lavis.cs.hs-rm.de/storage/cort/models/cort.msmarco.e128.pt)    |
| 256 |     78.2 |       87.0 | [cort.msmarco.e256.pt](http://lavis.cs.hs-rm.de/storage/cort/models/cort.msmarco.e256.pt)   |
| 768 |     78.6 |       87.3 | [cort.msmarco.e768.pt](http://lavis.cs.hs-rm.de/storage/cort/models/cort.msmarco.e768.pt)    |

### Create an Index

By default, when training is performed, an index is created for the specified collection based on the last model state. This index can be found in `runs/index/<run_id>`. If an previously trained CoRT model should be used, an index has to be created using the command `cort encode`:

```
$ cort encode runs/cort/my_run/checkpoints/last.ckpt data/collection.tsv -o runs/index/my_run
```

 
### Perform Ranking

Rankings for given queries are produced based on a previously created index . A corresponding `--qrels_file` can be specified to cause an automatic evaluation after ranking has been performed. 

```
$ cort rank runs/index/my_run/ data/queries.dev.small.tsv \
    --qrels_file data/qrels.dev.small.tsv --batch_size 32 -o data/my_ranking.tsv
```

### Merge with BM25

The final step is interleaving the ranking from CoRT with those from the targetted model (i.e. BM25). This can be done as follows:

```
$ cort merge data/my_ranking.tsv data/anserini.dev.small.tsv \
    --qrels_file data/qrels.dev.small.tsv -o data/my_merged_ranking.tsv
```

    

## CLI Command Reference

For convenience, this implementation of CoRT comes with CLI commands, which are automatically installed with `$pip install -e .`. However, please enter the directory of the cloned repository before you call any of the commands (`$cd /my/path/to/CoRT`). In the following sections we summarize the available commands and their parameters.

### Train a Model

Starts the training of a CoRT Model

```
usage: cort train [-h] [--run_id RUN_ID] [--margin MARGIN]
                  [--num_epochs NUM_EPOCHS]
                  [--batch_size BATCH_SIZE] [--lr LR]
                  [--embedding_size EMBEDDING_SIZE] 
                  [--weight_decay WEIGHT_DECAY]
                  [--device DEVICE] [--seed SEED]
                  [--pretrained_transformer PRETRAINED_TRANSFORMER]
                  [--passage_file PASSAGE_FILE]
                  [--train_negrank_file TRAIN_NEGRANK_FILE]
                  [--train_query_file TRAIN_QUERY_FILE]
                  [--train_qrel_file TRAIN_QREL_FILE]
                  [--val_negrank_file VAL_NEGRANK_FILE]
                  [--val_query_file VAL_QUERY_FILE]
                  [--val_qrel_file VAL_QREL_FILE]
                  [--test_query_file TEST_QUERY_FILE]
                  [--test_qrel_file TEST_QREL_FILE]
                  [--negative_min_rank NEGATIVE_MIN_RANK]
                  [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES]
                  [--metric_logger {wandb,tensorboard,none}]
                  [--validate VALIDATE] [--test TEST]
```

#### Parameters 

 - `--passage_file (default="./data/collection.tsv")`: The tsv-file containing all passages used for training and testing (MS MARCO format) 
 - `--{train,val,test}_query_file (default="./data/queries.<set-name>.tsv")`: The queries used for training/validation/testing
 - `--{train,val,test}_negrank_file (default=./data/anserini.<set-name>.tsv")`: Rankings from the target ranker to be complemented for training/validation/testing
 - `--{train,val,test}_qrel_file (default=./data/qrels.<set-name>.tsv")`: The query relevance labels for training/validation/testing
 - `--pretrained_transformer (default="albert-base-v2")`: The contextual encoder that is used by CoRT. Refers to [HuggingFace Model identifiers](https://huggingface.co/models).
 - `--accumulate_grad_batches (default=100)`: The number batch gradients that are accumulated per update step
 - `--batch_size (default=6)`: The number of triplets per batch.
 - `--device (default="0")`: The GPU index. "0" corresponds to "cuda:0"
 - `--embedding_size (default=128)`: The output representation size (e)
 - `--lr (defaul=2e-5)`: The learning rate at the highest point of the lr schedule.
 - `--margin (default=0.1)`: The margin of the triplet margin loss. From `{0.05, 0.1, 0.2}` we found 0.1 works best.
 - `--metric_logger (default=tensorboard)`: The metric logger used - choices: {wandb,tensorboard,none}  
 - `--num_epochs (default=10)`: The number of training epochs. 
 - `--validate (default=True)`: Determines if validation should be performed during training (requires val_* files)"
 - `--test (default=True)`:  Determines if test should be performed after training (requires test_* files)"

 
 ### Create an Index

Creates an Index by encoding each item from a given collection with the given model. Increasing the
batch_size can speed up this process. 

 ```
usage: cort encode [-h] -o OUT_DIR [-b BATCH_SIZE] [-d DEVICE]
                   model collection
```

`model` either refers to a checkpoint (e.g. `runs/cort/my_run/last.ckpt`) or persisted state_dict (e.g. `cort.msmarco.e128.pt`)

`collection` is the tsv-file containing the passages to be encoded

### Perform ranking

Produces rankings for given `queries` from a tsv file according to the given `index`. The result can be evaluated directly by specifying the `--qrels_file`.

```
usage: cort rank [-h] [-o OUT_FILE] [-b BATCH_SIZE] [-d DEVICE]
                 [--rank_len RANK_LEN] [--qrels_file QRELS_FILE]
                 [--strict_model_loading] [--log_file LOG_FILE]
                 index queries
```


### Merge Rankings

Interleaves two ranking files, starting with `rankings_a`. The result can be evaluated directly by specifying the `qrels_file`.

```
usage: cort merge [-h] [-o OUT_FILE] [--maxrank MAXRANK]
                  [--qrels_file QRELS_FILE] [--log_file LOG_FILE]
                  rankings_a rankings_b
```

### Evaluate Ranking

Calculate the evaluation metrics for a `ranking_file` and the relevance labels in `qrels_file`.

```
usage: cort eval [-h] ranking_file qrels_file
```

## How do I cite this work?

```
@inproceedings{wrzalik-krechel-2021-cort,
    title = "{C}o{RT}: Complementary Rankings from Transformers",
    author = "Wrzalik, Marco  and
      Krechel, Dirk",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.331",
    pages = "4194--4204",
}
```
