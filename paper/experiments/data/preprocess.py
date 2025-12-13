#!/usr/bin/env python3
"""
Preprocess downloaded datasets for evaluation.

This script converts raw dataset formats to the canonical format used
in the paper experiments:
- Boundaries indexed by USER TURN positions (matching training format)
- Unified JSON format with messages and gold_boundaries
- Train/val/test splits as used in the paper
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict


def load_segmentation_file(path: Path) -> Dict[str, Any]:
    """Load a segmentation JSON file in standard format."""
    with open(path) as f:
        return json.load(f)


def convert_to_canonical_boundaries(
    messages: List[Dict[str, str]],
    raw_boundaries: Set[int],
    boundary_format: str = "turn_index"
) -> Set[int]:
    """
    Convert raw boundary indices to canonical user-turn indices.

    The paper uses boundaries indexed by USER TURN position, where
    boundary k means "new topic starts at user turn k".

    Args:
        messages: List of {role, content} dicts
        raw_boundaries: Original boundary indices from dataset
        boundary_format: How raw boundaries are indexed:
            - "turn_index": Index into all turns (0-based)
            - "user_turn": Already user-turn indexed
            - "message_index": 1-based message index

    Returns:
        Set of user-turn indices where boundaries occur
    """
    if boundary_format == "user_turn":
        return raw_boundaries

    # Build mapping from message index to user turn index
    msg_to_user_turn = {}
    user_turn = 0
    for i, msg in enumerate(messages):
        if msg.get("role", "").lower() in ("user", "human", "customer"):
            msg_to_user_turn[i] = user_turn
            user_turn += 1

    canonical = set()
    for boundary in raw_boundaries:
        if boundary_format == "turn_index":
            # Find the user turn at or after this message
            if boundary in msg_to_user_turn:
                canonical.add(msg_to_user_turn[boundary])
            else:
                # Find next user turn
                for j in range(boundary, len(messages)):
                    if j in msg_to_user_turn:
                        canonical.add(msg_to_user_turn[j])
                        break
        elif boundary_format == "message_index":
            # 1-based to 0-based
            idx = boundary - 1
            if idx in msg_to_user_turn:
                canonical.add(msg_to_user_turn[idx])

    return canonical


def preprocess_dialseg711(input_dir: Path, output_dir: Path):
    """Preprocess DialSeg711 dataset."""
    print("Processing DialSeg711...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # DialSeg711 format: segmentation_file_test.json with dial_data
    test_file = input_dir / "segmentation_file_test.json"
    if not test_file.exists():
        print(f"  File not found: {test_file}")
        return

    data = load_segmentation_file(test_file)

    processed = []
    role_map = {"user": "user", "agent": "assistant", "system": "assistant"}

    for dataset_key, dialogues in data.get("dial_data", {}).items():
        for dialogue in dialogues:
            turns = dialogue.get("turns", [])

            messages = []
            boundaries = set()
            user_turn_idx = 0

            for i, turn in enumerate(turns):
                role = role_map.get(turn.get("role", "user"), "user")
                content = turn.get("utterance", "")

                # Track if this turn starts a new topic
                if turn.get("topic_start", False) or turn.get("new_topic", False):
                    if role == "user":
                        boundaries.add(user_turn_idx)

                messages.append({"role": role, "content": content})

                if role == "user":
                    user_turn_idx += 1

            if len(messages) >= 4:  # Minimum dialogue length
                processed.append({
                    "dialogue_id": dialogue.get("dial_id", f"dial_{len(processed)}"),
                    "messages": messages,
                    "gold_boundaries": sorted(boundaries),
                    "num_user_turns": user_turn_idx,
                })

    output_file = output_dir / "dialseg711_canonical.json"
    with open(output_file, "w") as f:
        json.dump(processed, f, indent=2)

    print(f"  Processed {len(processed)} dialogues -> {output_file}")


def preprocess_superseg(input_dir: Path, output_dir: Path):
    """Preprocess SuperDialseg dataset."""
    print("Processing SuperDialseg...")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        split_file = input_dir / f"segmentation_file_{split}.json"
        if not split_file.exists():
            print(f"  File not found: {split_file}")
            continue

        data = load_segmentation_file(split_file)
        processed = []

        # SuperDialseg has similar format to DialSeg711
        for dataset_key, dialogues in data.get("dial_data", {}).items():
            for dialogue in dialogues:
                turns = dialogue.get("turns", [])

                messages = []
                boundaries = set()
                user_turn_idx = 0

                for turn in turns:
                    role = "user" if turn.get("role") == "user" else "assistant"
                    content = turn.get("utterance", "")

                    if turn.get("topic_start", False):
                        if role == "user":
                            boundaries.add(user_turn_idx)

                    messages.append({"role": role, "content": content})

                    if role == "user":
                        user_turn_idx += 1

                if len(messages) >= 4:
                    processed.append({
                        "dialogue_id": dialogue.get("dial_id", f"dial_{len(processed)}"),
                        "messages": messages,
                        "gold_boundaries": sorted(boundaries),
                        "num_user_turns": user_turn_idx,
                    })

        output_file = output_dir / f"superseg_{split}_canonical.json"
        with open(output_file, "w") as f:
            json.dump(processed, f, indent=2)

        print(f"  Processed {len(processed)} dialogues ({split}) -> {output_file}")


def preprocess_tiage(input_dir: Path, output_dir: Path):
    """Preprocess TIAGE dataset."""
    print("Processing TIAGE...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # TIAGE may have different format - adjust as needed
    test_file = input_dir / "segmentation_file_test.json"
    if not test_file.exists():
        # Try alternative paths
        for alt in ["test.json", "tiage_test.json"]:
            alt_path = input_dir / alt
            if alt_path.exists():
                test_file = alt_path
                break

    if not test_file.exists():
        print(f"  No test file found in {input_dir}")
        return

    data = load_segmentation_file(test_file)
    processed = []

    # Process based on actual format
    if "dial_data" in data:
        # Same format as DialSeg711
        for dataset_key, dialogues in data.get("dial_data", {}).items():
            for dialogue in dialogues:
                turns = dialogue.get("turns", [])

                messages = []
                boundaries = set()
                user_turn_idx = 0

                for turn in turns:
                    role = "user" if turn.get("role") == "user" else "assistant"
                    content = turn.get("utterance", "")

                    if turn.get("topic_start", False):
                        if role == "user":
                            boundaries.add(user_turn_idx)

                    messages.append({"role": role, "content": content})

                    if role == "user":
                        user_turn_idx += 1

                if len(messages) >= 4:
                    processed.append({
                        "dialogue_id": dialogue.get("dial_id", f"dial_{len(processed)}"),
                        "messages": messages,
                        "gold_boundaries": sorted(boundaries),
                        "num_user_turns": user_turn_idx,
                    })

    output_file = output_dir / "tiage_canonical.json"
    with open(output_file, "w") as f:
        json.dump(processed, f, indent=2)

    print(f"  Processed {len(processed)} dialogues -> {output_file}")


def compute_dataset_stats(processed_dir: Path):
    """Compute and print statistics for processed datasets."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    for json_file in sorted(processed_dir.glob("*_canonical.json")):
        with open(json_file) as f:
            data = json.load(f)

        total_dialogues = len(data)
        total_messages = sum(len(d["messages"]) for d in data)
        total_boundaries = sum(len(d["gold_boundaries"]) for d in data)
        avg_boundaries = total_boundaries / total_dialogues if total_dialogues > 0 else 0

        print(f"\n{json_file.stem}:")
        print(f"  Dialogues: {total_dialogues}")
        print(f"  Total messages: {total_messages}")
        print(f"  Total boundaries: {total_boundaries}")
        print(f"  Avg boundaries/dialogue: {avg_boundaries:.2f}")


def main(datasets_dir: Path, output_dir: Path):
    """Run all preprocessing."""
    print("=" * 60)
    print("Preprocessing datasets to canonical format")
    print("=" * 60)

    # Process each dataset
    if (datasets_dir / "dialseg711").exists():
        preprocess_dialseg711(datasets_dir / "dialseg711", output_dir)

    if (datasets_dir / "superseg").exists():
        preprocess_superseg(datasets_dir / "superseg", output_dir)

    if (datasets_dir / "tiage").exists():
        preprocess_tiage(datasets_dir / "tiage", output_dir)

    # Compute stats
    compute_dataset_stats(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "datasets",
        help="Directory containing raw datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "processed",
        help="Output directory for processed files",
    )

    args = parser.parse_args()
    main(args.datasets_dir, args.output_dir)
