"""
download_datasets.py

Planner Dataset Downloader (CORRECTED)

IMPORTANT — READ BEFORE RUNNING
--------------------------------
`git clone` on these repos does NOT get you the datasets. Every deepfake
corpus here is EULA-gated and hosted separately from its GitHub repo:

  - FaceForensics++ repo  = docs + download.py, ~60MB. The actual videos
    (hundreds of GB) are only released after you fill the EULA form and
    the maintainers email you credentials for their download.py script.
  - Celeb-DF v2 repo      = docs + a request-form link, not the mp4s.
  - DFDC "repo" here      = NTech-Lab's competition-winning SOLUTION CODE,
    not the dataset. The actual DFDC videos (~470GB) are on Kaggle behind
    a competition EULA.
  - DiffusionDB repo      = the website/loader/notebooks (~7MB), not the
    1.6M generated images. Those come via huggingface-cli or the repo's
    own parquet shards.
  - ADE20K repo           = toolkit/docs. The annotated dataset needs
    registration at https://groups.csail.mit.edu/vision/datasets/ADE20K/

This script does two separate things and labels them separately:
  1. Clones the small CODE repos that are genuinely useful as tooling
     (DFDC solution code for face-tracking, DiffusionDB's loader).
  2. Prints the exact manual steps for each EULA-gated dataset — these
     steps cannot be automated, by design, because the license requires
     a human to accept terms.

Nothing under raw_datasets/ is considered "ready" until the manual steps
below are followed and files actually land in the class-labeled folders
organize_raw_datasets.py expects. Running this script alone is NOT
sufficient to proceed to organize_raw_datasets.py.

Usage
    python training/download_datasets.py
"""

import subprocess
from pathlib import Path

ROOT = Path("raw_datasets")
ROOT.mkdir(exist_ok=True)

# Only repos that are genuinely code/tooling, not the dataset itself.
# Cloned for convenience; does not fulfill any dataset requirement.
CODE_REPOS = {
    "dfdc_solution_code": {
        "git": "https://github.com/NTech-Lab/deepfake-detection-challenge.git",
        "folder": "_tooling/dfdc_solution_code",
        "note": "Face-tracking/training code only. Videos are NOT in this repo.",
    },
    "diffusiondb_loader": {
        "git": "https://github.com/poloclub/diffusiondb.git",
        "folder": "_tooling/diffusiondb_loader",
        "note": "Loader/notebooks only. Use huggingface-cli for the images (below).",
    },
}

# Everything that actually requires a manual, license-gated step.
MANUAL_STEPS = {
    "faceforensicspp": {
        "target": "raw_datasets/ffpp/",
        "steps": [
            "1. Fill the EULA at https://github.com/ondyari/FaceForensics",
            "2. Wait for the maintainers to email you download credentials.",
            "3. Run their download.py with those credentials, e.g.:",
            "     python download.py raw_datasets/ffpp -d all -c c23 -t videos",
            "4. Confirm you get original_sequences/ and manipulated_sequences/"
            "<Method>/ subfolders with real .mp4 files, not just docs.",
        ],
    },
    "celeb_df": {
        "target": "raw_datasets/celebdf/",
        "steps": [
            "1. Submit the request form linked from "
            "https://github.com/yuezunli/celeb-deepfakeforensics",
            "2. Download the provided archive once approved.",
            "3. Extract so you have Celeb-real/, Celeb-synthesis/, "
            "YouTube-real/ under raw_datasets/celebdf/.",
        ],
    },
    "dfdc": {
        "target": "raw_datasets/dfdc/",
        "steps": [
            "1. Accept the competition rules at "
            "https://www.kaggle.com/c/deepfake-detection-challenge",
            "2. kaggle competitions download -c deepfake-detection-challenge "
            "-p raw_datasets/dfdc",
            "3. Unzip so each chunk folder + its metadata.json sits under "
            "raw_datasets/dfdc/.",
        ],
    },
    "diffusiondb": {
        "target": "raw_datasets/diffusiondb/",
        "steps": [
            "1. huggingface-cli download poloclub/diffusiondb "
            "--repo-type dataset --local-dir raw_datasets/diffusiondb "
            "--include '*.parquet' '*.zip'  (2k subset is enough for a demo)",
            "2. Unzip the image shards so .png files sit directly under "
            "raw_datasets/diffusiondb/.",
        ],
    },
    "ade20k": {
        "target": "raw_datasets/ade20k/",
        "steps": [
            "1. Register at "
            "https://groups.csail.mit.edu/vision/datasets/ADE20K/",
            "2. Download the SceneParsing release and extract images under "
            "raw_datasets/ade20k/.",
        ],
    },
    "openimages": {
        "target": "raw_datasets/openimages/",
        "steps": [
            "1. Follow "
            "https://storage.googleapis.com/openimages/web/download.html "
            "(no EULA, but it's a large CLI pull, not a git clone).",
            "2. Store extracted images under raw_datasets/openimages/.",
        ],
    },
}


def clone_code_repos() -> None:
    print("=" * 80)
    print("Cloning tooling repos (code only -- NOT the datasets)")
    print("=" * 80)
    for name, info in CODE_REPOS.items():
        destination = ROOT / info["folder"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            print(f"[skip] {name} already present at {destination}")
            continue
        print(f"[+] Cloning {name} ({info['note']})")
        subprocess.run(
            ["git", "clone", "--depth", "1", info["git"], str(destination)]
        )


def print_manual_steps() -> None:
    print()
    print("=" * 80)
    print("MANUAL STEPS REQUIRED -- these cannot be scripted (EULA/registration)")
    print("=" * 80)
    for name, info in MANUAL_STEPS.items():
        print(f"\n[{name}] -> {info['target']}")
        for step in info["steps"]:
            print(f"    {step}")


def check_readiness() -> None:
    """Reports what's actually present vs. still missing, so you don't
    move on to organize_raw_datasets.py with an empty raw_datasets/ tree."""
    print()
    print("=" * 80)
    print("Readiness check")
    print("=" * 80)
    any_missing = False
    for name, info in MANUAL_STEPS.items():
        target = Path(info["target"])
        # A dataset is "ready" only if the target dir exists AND has more
        # than a token number of files -- a couple of stray files (e.g. a
        # README someone dropped in) shouldn't read as "done."
        n_files = sum(1 for _ in target.rglob("*")) if target.exists() else 0
        status = "READY" if n_files > 20 else "NOT READY"
        if status == "NOT READY":
            any_missing = True
        print(f"  {name:20s} {status:10s} ({n_files} files under {target})")

    print()
    if any_missing:
        print(
            "At least one dataset is not ready. organize_raw_datasets.py will "
            "loudly refuse to proceed on an empty manifest (see its own "
            "corrections) -- finish the manual steps above first."
        )
    else:
        print("All datasets appear populated. Proceed to organize_raw_datasets.py.")


if __name__ == "__main__":
    clone_code_repos()
    print_manual_steps()
    check_readiness()
    print()
    print("Next step (only after the manual steps above are actually done):")
    print("    python training/organize_raw_datasets.py")