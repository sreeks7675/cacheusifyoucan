"""
prepare_dataset.py

Adapter A Dataset Preparation Pipeline

Purpose
-------
Converts the unified manifest into a planner-ready dataset.

For every image this script generates

1. RGB reference
2. FFT magnitude image
3. Lightweight metadata
4. Surface routing labels
5. Planner manifest

This script DOES NOT perform deepfake detection.

It only prepares data for Adapter A.

Author:
Hackathon Planner Pipeline v2.0
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import json
from PIL import Image, ExifTags
from tqdm import tqdm

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLANNER_ROOT = PROJECT_ROOT / "planner_dataset"

RGB_DIR = PLANNER_ROOT / "rgb"
FFT_DIR = PLANNER_ROOT / "fft"
META_DIR = PLANNER_ROOT / "metadata"

RGB_DIR.mkdir(parents=True, exist_ok=True)
FFT_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = PLANNER_ROOT / "manifest.csv"

OUTPUT_MANIFEST = PLANNER_ROOT / "planner_manifest.csv"

# ----------------------------------------------------------
# Surface Observation Mapping
# ----------------------------------------------------------

SURFACE_OBSERVATIONS = {

    "real": [],

    "faceswap": [
        "boundary_blending",
        "landmark_shift",
        "eye_inconsistency"
    ],

    "gan": [
        "frequency_artifact",
        "checkerboard_pattern",
        "compression_difference",
        "noise_difference"
    ],

    "diffusion": [
        "frequency_artifact",
        "texture_repetition",
        "semantic_irregularity",
        "generator_pattern"
    ],

    "unknown": [
        "possible_manipulation"
    ]
}

# ----------------------------------------------------------
# Metadata Extraction
# ----------------------------------------------------------

def extract_metadata(image_path):

    metadata = {}

    try:

        img = Image.open(image_path)

        metadata["width"] = img.width
        metadata["height"] = img.height
        metadata["mode"] = img.mode

        try:

            exif = img._getexif()

            if exif is None:
                metadata["has_exif"] = False

            else:

                metadata["has_exif"] = True

                for tag, value in exif.items():

                    decoded = ExifTags.TAGS.get(tag, tag)

                    if decoded in [

                        "Make",
                        "Model",
                        "Software",
                        "DateTime"

                    ]:

                        metadata[decoded] = str(value)

        except Exception:

            metadata["has_exif"] = False

    except Exception:

        metadata["has_exif"] = False

    return metadata

# ----------------------------------------------------------
# FFT Generation
# ----------------------------------------------------------

def generate_fft(image_path):

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    fft = np.fft.fft2(img)

    fft_shift = np.fft.fftshift(fft)

    magnitude = np.log(np.abs(fft_shift) + 1)

    magnitude = cv2.normalize(

        magnitude,

        None,

        0,

        255,

        cv2.NORM_MINMAX

    )

    magnitude = magnitude.astype(np.uint8)

    return magnitude

# ----------------------------------------------------------
# Compression Statistics
# ----------------------------------------------------------

def compression_features(image):

    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    return {

        "laplacian_variance":

        float(laplacian.var())

    }

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

if not MANIFEST.exists():
    raise RuntimeError(
        f"{MANIFEST} not found. Run organize_raw_datasets.py first."
    )

manifest = pd.read_csv(MANIFEST)

if manifest.empty:
    raise RuntimeError(
        f"{MANIFEST} exists but has zero rows. organize_raw_datasets.py "
        "should have already refused to write an empty manifest -- check "
        "its output before re-running this script."
    )

planner_records = []

print()

print("=" * 70)
print("Preparing Planner Dataset")
print("=" * 70)

for idx, row in tqdm(

    manifest.iterrows(),

    total=len(manifest)

):

    image_path = Path(row["image_path"])

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    rgb_out = RGB_DIR / image_path.name

    cv2.imwrite(

        str(rgb_out),

        image

    )

    fft_image = generate_fft(image_path)

    fft_out = FFT_DIR / (

        image_path.stem + "_fft.png"

    )

    cv2.imwrite(

        str(fft_out),

        fft_image

    )

    metadata = extract_metadata(image_path)

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )

    metadata.update(

        compression_features(gray)

    )

    metadata_path = META_DIR / (

        image_path.stem + ".json"

    )

    with open(

        metadata_path,

        "w"

    ) as f:

        json.dump(

            metadata,

            f,

            indent=4

        )

    family = row["family"]

    observations = SURFACE_OBSERVATIONS.get(

        family,

        []

    )

    planner_records.append({

        "rgb":

            str(rgb_out),

        "fft":

            str(fft_out),

        "metadata":

            str(metadata_path),

        "family":

            family,

        "surface_observations":

            "|".join(observations),

        "routing_hints":

            row["routing_hints"]

    })

planner_df = pd.DataFrame(

    planner_records

)

planner_df.to_csv(

    OUTPUT_MANIFEST,

    index=False

)

print()

print("=" * 70)

print("Planner Dataset Created Successfully")

print()

print(f"Samples : {len(planner_df)}")

print()

print("RGB Images")

print(RGB_DIR)

print()

print("FFT Images")

print(FFT_DIR)

print()

print("Metadata")

print(META_DIR)

print("=" * 70)