#!/usr/bin/env python3
"""
Download datasets for paper experiments.

Due to dataset licensing restrictions, we do not redistribute raw dialogue data.
This script downloads and preprocesses all datasets used in the paper.

Datasets:
- DialSeg711 (Liu et al., EMNLP 2022)
- SuperDialseg (Liu et al., EMNLP 2023)
- TIAGE (Sun et al., COLING 2020)
- DailyDialog (Li et al., IJCNLP 2017)
- MultiWOZ (Budzianowski et al., EMNLP 2018)
- Taskmaster (Byrne et al., SIGDIAL 2019)
- Topical-Chat (Amazon, 2019)
- QMSum (Zhong et al., NAACL 2021)
"""

import os
import json
import hashlib
import zipfile
import tarfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from urllib.error import HTTPError

# Dataset sources and checksums
DATASETS = {
    "dialseg711": {
        "description": "DialSeg711: A Benchmark for Dialogue Topic Segmentation",
        "source": "https://github.com/Coldog2333/SuperDialseg",
        "type": "git",
        "git_url": "https://github.com/Coldog2333/SuperDialseg.git",
        "subpath": "data/dialseg711",
        "citation": "Liu et al., EMNLP 2022",
    },
    "superseg": {
        "description": "SuperDialseg: A Large-Scale Benchmark for Dialogue Topic Segmentation",
        "source": "https://github.com/Coldog2333/SuperDialseg",
        "type": "git",
        "git_url": "https://github.com/Coldog2333/SuperDialseg.git",
        "subpath": "data/superdialseg",
        "citation": "Liu et al., EMNLP 2023",
    },
    "tiage": {
        "description": "TIAGE: Topic Identification and Segmentation in Task-Oriented Dialogue",
        "source": "https://github.com/HaoSunTJU/TIAGE",
        "type": "git",
        "git_url": "https://github.com/HaoSunTJU/TIAGE.git",
        "subpath": "data",
        "citation": "Sun et al., COLING 2020",
    },
    "dailydialog": {
        "description": "DailyDialog: A Manually Labelled Multi-Turn Dialogue Dataset",
        "source": "http://yanran.li/dailydialog",
        "type": "url",
        "url": "http://yanran.li/files/ijcnlp_dailydialog.zip",
        "citation": "Li et al., IJCNLP 2017",
    },
    "multiwoz": {
        "description": "MultiWOZ: A Large-Scale Multi-Domain Wizard-of-Oz Dataset",
        "source": "https://github.com/budzianowski/multiwoz",
        "type": "git",
        "git_url": "https://github.com/budzianowski/multiwoz.git",
        "subpath": "data",
        "citation": "Budzianowski et al., EMNLP 2018",
    },
    "taskmaster": {
        "description": "Taskmaster: Toward Multimodal, Task-Oriented Dialogue",
        "source": "https://github.com/google-research-datasets/Taskmaster",
        "type": "git",
        "git_url": "https://github.com/google-research-datasets/Taskmaster.git",
        "subpath": "TM-1-2019",
        "citation": "Byrne et al., SIGDIAL 2019",
    },
    "topical_chat": {
        "description": "Topical-Chat: Towards Knowledge-Grounded Open-Domain Conversations",
        "source": "https://github.com/alexa/Topical-Chat",
        "type": "git",
        "git_url": "https://github.com/alexa/Topical-Chat.git",
        "subpath": "conversations",
        "citation": "Gopalakrishnan et al., 2019",
    },
    "qmsum": {
        "description": "QMSum: Query-based Multi-domain Meeting Summarization",
        "source": "https://github.com/Yale-LILY/QMSum",
        "type": "git",
        "git_url": "https://github.com/Yale-LILY/QMSum.git",
        "subpath": "data",
        "citation": "Zhong et al., NAACL 2021",
    },
}


def compute_checksum(filepath: Path) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL."""
    print(f"  Downloading from {url}...")
    try:
        urlretrieve(url, dest)
        return True
    except HTTPError as e:
        print(f"  Error downloading: {e}")
        return False


def clone_repo(git_url: str, dest: Path) -> bool:
    """Clone a git repository."""
    print(f"  Cloning {git_url}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", git_url, str(dest)],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error cloning: {e}")
        return False


def extract_archive(archive_path: Path, dest: Path) -> bool:
    """Extract zip or tar archive."""
    print(f"  Extracting {archive_path.name}...")
    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest)
        elif archive_path.suffix in (".gz", ".tgz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(dest)
        elif archive_path.suffix == ".tar":
            with tarfile.open(archive_path, "r:") as tf:
                tf.extractall(dest)
        return True
    except Exception as e:
        print(f"  Error extracting: {e}")
        return False


def download_dataset(name: str, output_dir: Path, force: bool = False) -> bool:
    """Download a single dataset."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        return False

    config = DATASETS[name]
    dataset_dir = output_dir / name

    if dataset_dir.exists() and not force:
        print(f"  {name}: already exists (use --force to re-download)")
        return True

    print(f"\nDownloading {name}...")
    print(f"  Source: {config['source']}")
    print(f"  Citation: {config['citation']}")

    temp_dir = output_dir / f".tmp_{name}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    success = False

    if config["type"] == "git":
        repo_dir = temp_dir / "repo"
        if clone_repo(config["git_url"], repo_dir):
            # Copy relevant subpath
            src = repo_dir / config.get("subpath", "")
            if src.exists():
                if dataset_dir.exists():
                    shutil.rmtree(dataset_dir)
                shutil.copytree(src, dataset_dir)
                success = True
            else:
                print(f"  Subpath not found: {config.get('subpath', '')}")

    elif config["type"] == "url":
        url = config["url"]
        archive_name = url.split("/")[-1]
        archive_path = temp_dir / archive_name

        if download_file(url, archive_path):
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            if extract_archive(archive_path, extract_dir):
                # Find the data directory in extracted content
                # This may need adjustment per dataset
                if dataset_dir.exists():
                    shutil.rmtree(dataset_dir)
                shutil.copytree(extract_dir, dataset_dir)
                success = True

    # Cleanup temp dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    if success:
        print(f"  {name}: downloaded successfully")
    else:
        print(f"  {name}: download failed")

    return success


def download_all(output_dir: Path, datasets: Optional[list] = None, force: bool = False):
    """Download all or specified datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = list(DATASETS.keys())

    print("=" * 60)
    print("Downloading datasets for paper experiments")
    print("=" * 60)

    results = {}
    for name in datasets:
        results[name] = download_dataset(name, output_dir, force=force)

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")

    failed = [n for n, s in results.items() if not s]
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        print("See dataset sources for manual download instructions.")


def print_sources():
    """Print dataset sources for manual download."""
    print("=" * 60)
    print("Dataset Sources")
    print("=" * 60)
    for name, config in DATASETS.items():
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Source: {config['source']}")
        print(f"  Citation: {config['citation']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets for paper experiments")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "datasets",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Specific datasets to download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if exists",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print dataset sources and exit",
    )

    args = parser.parse_args()

    if args.list_sources:
        print_sources()
    else:
        download_all(args.output_dir, args.datasets, args.force)
