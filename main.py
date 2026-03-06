import argparse
import os


def run(cmd):
    os.system(cmd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--train_baseline", action="store_true")
    parser.add_argument("--train_sparse", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    if args.collect:
        run("python data/collect_data.py")

    if args.train_baseline:
        run("python training/train_baseline.py")

    if args.train_sparse:
        run("python training/train_sparse.py")

    if args.evaluate:
        run("python evaluation/compute_metrics.py")

    if args.visualize:
        run("python evaluation/evaluate_prediction.py")