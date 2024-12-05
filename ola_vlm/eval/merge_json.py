import os
import json
import argparse

parser = argparse.ArgumentParser(
    description='Probe eval')
parser.add_argument('--ckpt',
                    help='ckpt',
                    default='probe_llava-1.5-vicuna-7b-lr-1e-3')
parser.add_argument('--mode',
                    help='mode',
                    default='gen')
parser.add_argument("--num-chunks", type=int, default=1)


def save_merged_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    merge_data = {}
    name = args.ckpt.split("/")[-1]

    for i in range(args.num_chunks):
        with open(f'plots/probe_scores/{name}/{args.mode}/{args.num_chunks}_{i}.json', 'r') as file:
           data = json.load(file)
        merge_data.update(data)
    
    save_merged_json(merge_data, f'plots/probe_scores/{name}/{args.mode}.json')