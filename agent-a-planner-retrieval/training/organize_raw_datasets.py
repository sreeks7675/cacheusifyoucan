"""
organize_raw_datasets.py

Adapter A Planner Dataset Organizer (CORRECTED)

Purpose
-------
Creates a unified dataset manifest from multiple public datasets
without changing the existing project folder structure.

Supported datasets

- FaceForensics++   (forensic: faceswap / gan families)
- Celeb-DF v2        (forensic: faceswap family)
- DFDC               (forensic: mixed/unknown family)
- DiffusionDB        (diffusion family)
- ADE20K / OpenImages (semantic_scene family -- object/scene grounding for
  the "semantic" routing category; these are NOT real/fake datasets, they
  just give the planner real-world scene examples with no manipulation
  cues, which is why v2 PLANNER_TRAINING_DESIGN.md / DATASET_SOURCES.md
  call for them. The previous version of this script silently dropped
  them even though the design docs require them.)

Expects the FLAT layout download_datasets.py's MANUAL_STEPS produce:
    raw_datasets/ffpp/<class>/...
    raw_datasets/celebdf/<class>/...
    raw_datasets/dfdc/<class>/...
    raw_datasets/diffusiondb/<class>/...
    raw_datasets/ade20k/...        (no class subfolders -- scene images)
    raw_datasets/openimages/...    (no class subfolders -- scene images)

Output

planner_dataset/
    manifest.csv
    train.csv
    val.csv
    test.csv

The actual RGB images remain in their original locations.
Only metadata files are generated.

CORRECTIONS vs. previous version
---------------------------------
1. Folder names now match what download_datasets.py actually stages
   (raw_datasets/ffpp, not raw_datasets/faceforensicspp) -- these were
   silently mismatched before, which would have produced an empty
   manifest for every dataset even after downloading was fixed.
2. Added ade20k/openimages support for the semantic_scene family, which
   the v2 design docs require but this script previously omitted.
3. Refuses to proceed (raises, does not just print a warning) if the
   final manifest is empty, or if any family has too few samples for
   the stratified train/val/test split -- previously this would either
   crash deep inside sklearn with a cryptic error, or (worse) silently
   write empty train/val/test CSVs that downstream scripts wouldn't
   notice were empty.
"""

from pathlib import Path

from sklearn.model_selection import train_test_split
import pandas as pd

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASETS = PROJECT_ROOT / "raw_datasets"

OUTPUT_DIR = PROJECT_ROOT / "planner_dataset"

OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Datasets with real/fake or method-labeled class subfolders.
CLASS_LABELED_DATASETS = {
    "ffpp": {
        "classes": [
            "real",
            "deepfakes",
            "faceswap",
            "face2face",
            "neuraltextures",
            "faceshifter",
        ]
    },
    "celebdf": {"classes": ["real", "fake"]},
    "dfdc": {"classes": ["real", "fake"]},
    "diffusiondb": {"classes": ["real", "fake_diffusion"]},
}

# Datasets that are just a pool of real-world scene images (no class
# subfolders, no manipulation labels) -- used for the semantic_scene
# family so the planner sees plausible real scenes for object/context
# consistency checks.
SCENE_ONLY_DATASETS = ["ade20k", "openimages"]

# ---------------------------------------------------------------------
# Manipulation family mapping
# ---------------------------------------------------------------------

MANIPULATION_FAMILY = {
    "real": "real",
    "deepfakes": "faceswap",
    "faceswap": "faceswap",
    "face2face": "gan",
    "neuraltextures": "gan",
    "faceshifter": "faceswap",
    "fake": "unknown",
    "fake_diffusion": "diffusion",
}

# ---------------------------------------------------------------------
# Planner routing hints
# ---------------------------------------------------------------------

ROUTING_HINTS = {
    "real": [],
    "faceswap": ["boundary", "landmarks", "eyes"],
    "gan": ["frequency", "noise", "compression"],
    "diffusion": ["generator", "semantic"],
    "unknown": ["forensic"],
    "semantic_scene": ["semantic"],
}

# ---------------------------------------------------------------------
# Scan helpers
# ---------------------------------------------------------------------


def scan_class_labeled(dataset_name: str, config: dict, records: list) -> int:
    dataset_root = RAW_DATASETS / dataset_name
    if not dataset_root.exists():
        print(f"[WARNING] Missing dataset: {dataset_name} (expected {dataset_root})")
        return 0

    found = 0
    for cls in config["classes"]:
        class_dir = dataset_root / cls
        if not class_dir.exists():
            continue
        for file in class_dir.rglob("*"):
            if file.suffix.lower() not in IMAGE_EXTS:
                continue
            family = MANIPULATION_FAMILY.get(cls, "unknown")
            records.append(
                {
                    "dataset": dataset_name,
                    "class": cls,
                    "family": family,
                    "image_path": str(file),
                    "routing_hints": ",".join(ROUTING_HINTS.get(family, [])),
                }
            )
            found += 1
    return found


def scan_scene_only(dataset_name: str, records: list) -> int:
    dataset_root = RAW_DATASETS / dataset_name
    if not dataset_root.exists():
        print(f"[WARNING] Missing dataset: {dataset_name} (expected {dataset_root})")
        return 0

    found = 0
    for file in dataset_root.rglob("*"):
        if file.suffix.lower() not in IMAGE_EXTS:
            continue
        records.append(
            {
                "dataset": dataset_name,
                "class": "scene",
                "family": "semantic_scene",
                "image_path": str(file),
                "routing_hints": ",".join(ROUTING_HINTS["semantic_scene"]),
            }
        )
        found += 1
    return found


# ---------------------------------------------------------------------
# Run the scan
# ---------------------------------------------------------------------

records: list = []

print("=" * 60)
print("Scanning raw_datasets/")
print("=" * 60)

for dataset_name, config in CLASS_LABELED_DATASETS.items():
    n = scan_class_labeled(dataset_name, config, records)
    print(f"  {dataset_name:15s} {n} images")

for dataset_name in SCENE_ONLY_DATASETS:
    n = scan_scene_only(dataset_name, records)
    print(f"  {dataset_name:15s} {n} images")

manifest = pd.DataFrame(records)

# ---------------------------------------------------------------------
# Hard stop instead of silently producing a broken/empty manifest
# ---------------------------------------------------------------------

if manifest.empty:
    raise RuntimeError(
        "Manifest is empty -- no images were found under any dataset in "
        f"{RAW_DATASETS}. This almost always means the manual download "
        "steps in download_datasets.py haven't been completed yet, or the "
        "downloaded files aren't sitting in the expected class subfolders "
        "(see this script's docstring for the exact expected layout). "
        "Fix the data staging before re-running -- do not proceed to "
        "prepare_dataset.py with an empty manifest."
    )

family_counts = manifest["family"].value_counts()
print()
print("=" * 60)
print("Planner Dataset Summary")
print("=" * 60)
print(manifest.groupby(["dataset", "family"]).size())

# train_test_split with stratify requires at least 2 members per class in
# each split. Catch this explicitly rather than letting sklearn raise a
# generic ValueError deep in the stack, or (if we didn't stratify) letting
# a rare family silently vanish from one of the splits.
too_small = family_counts[family_counts < 4]
if not too_small.empty:
    raise RuntimeError(
        "These families have fewer than 4 samples, which isn't enough for "
        "a stratified train/val/test split:\n"
        f"{too_small.to_string()}\n"
        "Either gather more data for them, or drop them from "
        "CLASS_LABELED_DATASETS / SCENE_ONLY_DATASETS before re-running."
    )

manifest.to_csv(OUTPUT_DIR / "manifest.csv", index=False)

# ---------------------------------------------------------------------
# Train / Validation / Test
# ---------------------------------------------------------------------

train_df, temp_df = train_test_split(
    manifest, test_size=0.30, stratify=manifest["family"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["family"], random_state=42
)

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print()
print("=" * 60)
print("Planner dataset successfully organized.")
print(f"Total Images : {len(manifest)}")
print(f"Train        : {len(train_df)}")
print(f"Validation   : {len(val_df)}")
print(f"Test         : {len(test_df)}")
print("=" * 60)