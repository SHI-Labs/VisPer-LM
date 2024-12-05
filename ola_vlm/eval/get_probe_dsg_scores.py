import argparse
import torch

import json
import os
from tqdm import tqdm
from icecream import ic
import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="llava-1.5-7b")
    parser.add_argument("--mode", type=str, default="gen")
    args = parser.parse_args()

    mode = args.mode
    name = args.ckpt.split("/")[-1]

    with open(f'plots/probe_scores/{name}/{args.mode}.json') as file:
        scores = json.load(file)
    
    layer_scores = {}

    for img, v in tqdm(scores.items()):
        for layer, score in v.items():
            if layer not in layer_scores:
                layer_scores[layer] = []
            layer_scores[layer].append(score)
    
    for layer, scores in layer_scores.items():
        layer_scores[layer] = np.mean(scores)

    with open(f"plots/probe_scores/{name}/{mode}_scores.json", "w") as f:
        json.dump(layer_scores, f, indent=2)
    
    print(f"================Scores: {mode}===============")
    for layer, score in layer_scores.items():
        print(f"Layer: {layer}, Score: {score}")
    print("===========================================")