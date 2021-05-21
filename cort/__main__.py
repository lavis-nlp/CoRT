import argparse
import sys
import importlib


USAGE = """cort <command> [<args>]
    
The commands are:
    train   Train a new model
    encode  Encode passages using a trained model
    rank    Rank passages to given queries using pre-encoded representations
            and the corresponding trained model for query encoding
    merge   Merge two rankings
    eval    Evaluate ranking
    """


def main():
    parser = argparse.ArgumentParser(usage=USAGE)

    parser.add_argument("command", choices=["train", "encode", "rank", "eval", "merge"])
    args = parser.parse_args(sys.argv[1:2])

    sys.argv[0] = " ".join(sys.argv[:2])
    del sys.argv[1]

    target_module = importlib.import_module("cort." + args.command)

    target_module.main()


if __name__ == "__main__":
    main()
