import pandas as pd
import json
import argparse

def load_jsonl(f):
    lines = open(f, encoding='utf-8').readlines()
    lines = [x.strip() for x in lines]
    if lines[-1] == '':
        lines = lines[:-1]
    data = [json.loads(x) for x in lines]
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="cv-bench_answer.jsonl")
    args = parser.parse_args()

    answers = load_jsonl(args.results_file)

    data = {
         "source": [],
         "result": [],
         "task": [],
    }
    import re
    for a in answers:
        data["source"].append(a["source"][0])
        if "(" in a["prediction"]:
            match = re.search(r'\(([A-Z])\)', a["prediction"])
            if match:
                pred = "(" + match.group(1) + ")"
        else:
            pred = "(" + a["prediction"][0] + ")"
        data["result"].append(pred == a["answer"][0])
        data["task"].append(a["task"][0])

    df = pd.DataFrame(data)

    def calculate_accuracy(df, source):
        source_df = df[df['source'] == source]
        accuracy = (source_df['result']).mean()
        return accuracy
    
    def calculate_task_accuracy(df, task):
        source_df = df[df['task'] == task]
        accuracy = (source_df['result']).mean()
        return accuracy

    accuracy_2d_ade = calculate_accuracy(df, 'ADE20K')
    accuracy_2d_coco = calculate_accuracy(df, 'COCO')
    accuracy_3d_omni = calculate_accuracy(df, 'Omni3D')

    tasks = ["Count", "Depth", "Relation", "Distance"]

    scores = {}

    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni

    combined_accuracy = (accuracy_2d + accuracy_3d) / 2

    scores["Overall"] = combined_accuracy

    scores["3D"] = accuracy_3d
    scores["2D"] = accuracy_2d

    for t in tasks:
        accuracy = calculate_task_accuracy(df, t)
        scores[t] = accuracy

    print("\n=========================CV-Bench Scores===============================")
    for key, value in scores.items():
        print(f"{key} -> {value}")
    print("================================================================")

    with open(args.results_file.replace('.jsonl', '_score.json'), "w") as f:
        json.dump(scores, f, indent=2)