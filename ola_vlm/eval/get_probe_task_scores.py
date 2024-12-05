import argparse
import torch
from PIL import Image
import json
import os
from tqdm import tqdm
import warnings
import random
import numpy as np
from ola_vlm.eval.fid_score import compute_fid
import multiprocessing as mp
from multiprocessing import Pool
warnings.filterwarnings("ignore")


def parse_json(file):
    with open(file) as f:
        data = json.load(f)
    return data

def prepare_coco(json_file):
    coco_data = parse_json(json_file)

    id_to_filename = {image["id"]: image["file_name"] for image in coco_data["images"]}
    images = []
    prompts = []
    answers = []

    im_ids = []

    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        if image_id in im_ids:
            continue
        file_name = id_to_filename[image_id]
        images.append(os.path.join(json_file.split("/annotations")[0], "val2017", file_name))
        im_ids.append(image_id)
        answers.append(caption)
        prompts.append("Describe the image in two lines.")

    return images, prompts, answers

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_image(image_file):
    image = Image.open(image_file)
    return image

def mask_iou(gt, pred):
    gt = np.array(gt).astype(np.uint8)
    pred = np.array(pred).astype(np.uint8)

    iou_scores = []
    for category in np.unique(gt):
        if category == 255:
            continue
        gt_mask = (gt == category)
        pred_mask = (pred == category)

        intersection = np.logical_and(gt_mask, pred_mask)
        union = np.logical_or(gt_mask, pred_mask)
        if np.sum(union) == 0:
            iou_scores.append(1.0)
        else:
            iou_scores.append(np.sum(intersection) / np.sum(union))
    
    return np.mean(iou_scores)

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

# Helper function for multiprocessing in evaluate_seg
def process_iou(args):
    gt_path, layer_folder, dir, fname = args
    gt_data = load_image(os.path.join(gt_path, fname.replace("jpg", "png")))
    pred = load_image(os.path.join(layer_folder, dir, fname))
    return mask_iou(gt_data, pred)

def evaluate_seg(args):
    images, _, _ = prepare_coco("datasets/coco/annotations/captions_val2017.json")
    fnames = [img.split("/")[-1] for img in images][:8]

    name = args.ckpt.split("/")[-1]
    gt_path = "datasets/coco/annotations/panoptic_semseg_val2017"
    layer_folder = f"plots/probes_task/{name}/seg"

    scores = {"m_iou": []}
    dirs = os.listdir(layer_folder)
    
    with mp.Pool() as pool:
        for dir in dirs:
            print(f"Evaluating mask iou for {dir}")
            args_list = [(gt_path, layer_folder, dir, fname) for fname in fnames]
            m_iou = list(tqdm(pool.imap(process_iou, args_list), total=len(args_list), desc=f"Processing {dir}"))
            scores["m_iou"].append({dir: round(np.mean(m_iou) * 100, 2)})

    return scores

# Helper function for multiprocessing in evaluate_depth
def process_depth(args):
    depth_map, point_1, point_2, answer = args
    return score_points(depth_map, point_1, point_2, answer)

def score_points(depth_map, point_1, point_2, answer):
    pt1_depth = depth_map[point_1[0], point_1[1]]
    pt2_depth = depth_map[point_2[0], point_2[1]]

    if isinstance(pt1_depth, np.ndarray):
        pt1_depth = pt1_depth.mean()
    if isinstance(pt2_depth, np.ndarray):
        pt2_depth = pt2_depth.mean()

    return (answer == "point2") if pt1_depth < pt2_depth else (answer == "point1")

def load_and_process_image(args):
    folder, fname, entry = args
    gt_path = os.path.join("plots/dav2_da2k", fname.split("/")[-1].split(".")[0] + ".jpg")
    pred_path = os.path.join(folder, fname.split("/")[-1])

    gt = load_image(gt_path)
    pred = load_image(pred_path)
    pred = pred.resize(gt.size)
    pred = np.array(pred) / 255.0

    # Process depth for each entry within the image
    return [process_depth((pred, entry["point1"], entry["point2"], entry["closer_point"])) for entry in entry["entries"]]

def score_da2k_parallel(folder, anns):
    pred_scores = []
    tasks = [(folder, fname, {"entries": entries}) for fname, entries in anns.items()]

    with Pool() as pool:
        results = list(tqdm(pool.imap(load_and_process_image, tasks), total=len(tasks), desc="Processing images"))
        for res in results:
            if res is not None:
                pred_scores.extend(res)

    return np.mean(pred_scores) if pred_scores else 0

def evaluate_depth(args):
    anns = parse_json("datasets/eval/DA-2K/annotations.json")

    name = args.ckpt.split("/")[-1]
    layer_folder = f"plots/probes_task/{name}/depth"

    scores = {"da2k_acc": []}
    dirs = os.listdir(layer_folder)
    
    for dir in dirs:
        print(f"Evaluating da2k_acc for {dir}")
        pred_scores = score_da2k_parallel(os.path.join(layer_folder, dir), anns)
        scores["da2k_acc"].append({dir: round(pred_scores * 100, 2)})

    return scores

def evaluate_fid(args):
    name = args.ckpt.split("/")[-1]
    gt_path = os.path.join("plots/coco_gt")
    layer_folder = f"plots/probes_task/{name}/gen"

    scores = {"fid": []}
    dirs = os.listdir(layer_folder)
    
    for dir in dirs:
        print(f"Evaluating fid for {dir}")
        paths = [gt_path, os.path.join(layer_folder, dir)]
        fid_score = compute_fid(paths)
        scores["fid"].append({dir.replace("_", "-"): round(fid_score, 2)})
    
    return scores

import re

def print_sorted_scores(scores, metric_name):
    # Extract numeric part from layer names for sorting
    sorted_scores = sorted(scores[metric_name], key=lambda x: int(re.search(r'\d+', list(x.keys())[0]).group()))
    
    layers = [list(score.keys())[0] for score in sorted_scores]
    values = [list(score.values())[0] for score in sorted_scores]

    # Print sorted layers and scores in the requested format
    print("\n=========================Results===============================")
    print(" & ".join(layers))
    print(" & ".join([f"{value}" for value in values]))
    print(f"Average score: {round(np.mean(values), 2)}")
    print("================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="llava-1.5-7b")
    parser.add_argument("--mode", type=str, default="gen")
    args = parser.parse_args()

    mode = args.mode

    if mode == "gen":
        scores = evaluate_fid(args)
        
        print("\n=========================FID-Scores===============================")
        for score in scores["fid"]:
            for key, value in score.items():
                print(f"{key} -> {value}")
        print("================================================================")
    
    elif mode == "seg":
        scores = evaluate_seg(args)
        
        print("\n=========================Mask-IOU===============================")
        print_sorted_scores(scores, "m_iou")

    elif mode == "depth":
        scores = evaluate_depth(args)
        
        print("\n=========================DA2K-Acc===============================")
        print_sorted_scores(scores, "da2k_acc")

    else:
        print("Invalid mode. Choose from [gen, seg, depth]")
