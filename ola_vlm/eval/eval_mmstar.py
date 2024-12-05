import os
import argparse
import json

from ola_vlm.eval.mmstar.evaluate import MMStar_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, default="./playground/data/eval/mmstar_results.jsonl")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    MMStar_eval(args.results_file)
