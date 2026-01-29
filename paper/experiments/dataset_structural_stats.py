#!/usr/bin/env python3
"""Compute structural stats for all datasets with gold boundaries."""
import json
from pathlib import Path
import numpy as np

DATASETS_DIR = Path(__file__).parent.parent.parent / "datasets"


def load_dataset(dataset_name):
    """Load a dataset with standard format."""
    file_path = DATASETS_DIR / dataset_name / "segmentation_file_test.json"
    if not file_path.exists():
        # Try train split
        file_path = DATASETS_DIR / dataset_name / "segmentation_file_train.json"
        if not file_path.exists():
            return None

    with open(file_path) as f:
        data = json.load(f)

    dial_data = data.get("dial_data", data)
    dialogues = []

    for source_key, source_dialogs in dial_data.items():
        if not isinstance(source_dialogs, list):
            continue
        for dialog in source_dialogs:
            turns = dialog.get("turns", [])

            gold_boundaries = set()
            prev_topic = None
            user_idx = 0

            for turn in turns:
                role = turn.get("role", "user")
                if role == "user":
                    topic = turn.get("topic_id") or turn.get("topic_name")
                    if prev_topic is not None and topic != prev_topic:
                        gold_boundaries.add(user_idx)
                    prev_topic = topic
                    user_idx += 1

            if user_idx > 0:
                dialogues.append({
                    "num_turns": user_idx,
                    "boundaries": gold_boundaries
                })

    return dialogues


def compute_stats(dialogues):
    """Compute structural statistics for a dataset."""
    if not dialogues:
        return None

    total_dialogues = len(dialogues)
    total_turns = sum(d["num_turns"] for d in dialogues)
    total_boundaries = sum(len(d["boundaries"]) for d in dialogues)

    # Turns per segment: for each dialogue, compute segment lengths
    all_segment_lengths = []
    for d in dialogues:
        turns = d["num_turns"]
        bounds = sorted(d["boundaries"])

        # Segment boundaries (including start and end)
        segment_starts = [0] + list(bounds)
        segment_ends = list(bounds) + [turns]

        for start, end in zip(segment_starts, segment_ends):
            seg_len = end - start
            if seg_len > 0:
                all_segment_lengths.append(seg_len)

    if all_segment_lengths:
        turns_per_seg_median = np.median(all_segment_lengths)
        turns_per_seg_mean = np.mean(all_segment_lengths)
    else:
        turns_per_seg_median = 0
        turns_per_seg_mean = 0

    # Boundary density
    boundary_density = total_boundaries / total_turns if total_turns > 0 else 0

    # Boundaries per dialogue
    bounds_per_dialog = [len(d["boundaries"]) for d in dialogues]
    bounds_per_dialog_mean = np.mean(bounds_per_dialog)
    bounds_per_dialog_median = np.median(bounds_per_dialog)

    return {
        "dialogues": total_dialogues,
        "turns_per_seg_median": turns_per_seg_median,
        "turns_per_seg_mean": turns_per_seg_mean,
        "boundary_density": boundary_density,
        "bounds_per_dialog_mean": bounds_per_dialog_mean,
        "bounds_per_dialog_median": bounds_per_dialog_median,
    }


def main():
    datasets = [
        "dialseg711",
        "superseg",
        "tiage",
        "dailydialog",
        "topicalchat",
        "qmsum",
        "multiwoz",
        "taskmaster",
    ]

    # Print header
    print("| Dataset      | Dialogues | Turns/Seg (median) | Turns/Seg (mean) | Boundary Density | Bounds/Dialog |")
    print("|--------------|-----------|--------------------| -----------------|------------------|---------------|")

    for name in datasets:
        dialogues = load_dataset(name)
        if dialogues is None:
            print(f"| {name:12} | N/A       | N/A                | N/A              | N/A              | N/A           |")
            continue

        stats = compute_stats(dialogues)
        if stats is None:
            print(f"| {name:12} | N/A       | N/A                | N/A              | N/A              | N/A           |")
            continue

        print(f"| {name:12} | {stats['dialogues']:9} | {stats['turns_per_seg_median']:18.1f} | {stats['turns_per_seg_mean']:16.1f} | {stats['boundary_density']:16.3f} | {stats['bounds_per_dialog_mean']:13.1f} |")


if __name__ == "__main__":
    main()
