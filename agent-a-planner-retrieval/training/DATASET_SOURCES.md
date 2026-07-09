# Dataset Sources → Planner Training Mapping
Version 2.0

---

# Objective

The Planner Agent is **NOT** trained to classify images as real or fake.

Instead, it learns to identify **surface-level evidence** that determines
which specialist agent(s) should be invoked.

The planner therefore learns from a combination of

- real images
- manipulated images
- semantic datasets
- generator diversity
- frequency-domain representations
- teacher model outputs

The downstream forensic, semantic and retrieval agents perform the detailed
analysis.

---

# Planner Routing Categories

The planner predicts routing based on three families.

## 1. Forensic

Surface observations only.

Examples

- blending boundaries
- facial inconsistencies
- eye inconsistencies
- reflections
- specular highlights
- FFT anomalies
- checkerboard artifacts
- periodic frequency peaks
- high-frequency imbalance
- compression artifacts
- sensor noise
- metadata anomalies

The planner NEVER verifies them.

It only decides

"Forensic analysis recommended."

---

## 2. Semantic

Planner learns

- object consistency
- scene consistency
- context realism
- physical plausibility
- lighting
- shadow consistency
- OCR artifacts
- watermark artifacts
- text rendering
- perspective consistency

Again,

only surface observations.

---

## 3. Retrieval / Comparison

Planner learns

- generator fingerprints
- repeated texture patterns
- reverse-search candidates
- duplicate likelihood
- style similarity
- diffusion appearance
- GAN appearance

The retrieval agent performs the actual search.

---

# Dataset Mapping

The following datasets are used to manufacture planner supervision.

-------------------------------------------------------------------------------

## FaceForensics++

Repository

https://github.com/ondyari/FaceForensics

Planner contribution

✓ Face blending

✓ Facial landmarks

✓ Eye inconsistencies

✓ Compression

✓ Frequency-domain artifacts

✓ Reflection inconsistencies

✓ Metadata anomalies

Maps to

data/

    real/

    fake_faceswap/

    fake_compression/

-------------------------------------------------------------------------------

## Celeb-DF v2

Repository

https://github.com/yuezunli/celeb-deepfakeforensics

Planner contribution

✓ High quality face swaps

✓ Boundary transitions

✓ Skin blending

✓ Eyes

✓ Expression inconsistencies

Maps to

data/

    fake_faceswap/

-------------------------------------------------------------------------------

## DFDC

Repository

https://github.com/NTech-Lab/deepfake-detection-challenge

Planner contribution

✓ Large manipulation diversity

✓ Unknown generators

✓ Real-world compression

✓ Social-media artifacts

Maps to

data/

    real/

    fake_faceswap/

    fake_gan/

-------------------------------------------------------------------------------

## WildDeepfake

Planner contribution

✓ Internet deepfakes

✓ Cropped faces

✓ Compression

✓ Unknown manipulation families

Maps to

data/

    fake_faceswap/

    fake_compression/

-------------------------------------------------------------------------------

## DiffusionDB

Repository

https://github.com/poloclub/diffusiondb

Planner contribution

✓ Diffusion appearance

✓ Texture repetition

✓ Frequency smoothing

✓ Generator fingerprints

Maps to

data/

    fake_diffusion/

-------------------------------------------------------------------------------

## OpenImages / COCO

Repository

https://github.com/bethgelab/openimages2coco

Planner contribution

✓ Object consistency

✓ Scene consistency

✓ Physical plausibility

✓ Context reasoning

✓ Semantic routing

Maps to

data/

    semantic_scene/

-------------------------------------------------------------------------------

## ADE20K

Repository

https://github.com/CSAILVision/ADE20K

Planner contribution

✓ Indoor / outdoor reasoning

✓ Scene layout

✓ Context consistency

✓ Object relationships

Maps to

data/

    semantic_scene/

-------------------------------------------------------------------------------

# Planner Feature Space

Each image contributes

RGB

↓

FFT Magnitude

↓

Noise Residual

↓

JPEG Error Map

↓

Metadata Summary

↓

Teacher Signals

↓

Planner Training Sample

-------------------------------------------------------------------------------

# Teacher Distillation

Planner supervision is generated from the specialist models.

Teacher outputs include

Forensic

Semantic

Retrieval

Each teacher produces

{
    "score": 0.91,

    "confidence": 0.95,

    "reason": "Checkerboard frequency pattern"
}

Planner learns

surface observations

↓

dispatch

NOT

generator labels.

-------------------------------------------------------------------------------

# Directory Layout

CORRECTED: flat, not nested under forensic/diffusion/semantic -- this is
what download_datasets.py stages and organize_raw_datasets.py actually
reads. wilddeepfake was dropped (no longer in download_datasets.py's
MANUAL_STEPS); add it back to both scripts together if you want it.

raw_datasets/

    ffpp/
        real/
        deepfakes/
        faceswap/
        face2face/
        neuraltextures/
        faceshifter/

    celebdf/
        real/
        fake/

    dfdc/
        real/
        fake/

    diffusiondb/
        real/
        fake_diffusion/

    ade20k/            (no class subfolders -- scene images only)

    openimages/         (no class subfolders -- scene images only)

planner_dataset/

    rgb/

    fft/

    noise/

    jpeg/

    metadata/

    labels/

-------------------------------------------------------------------------------

# Final Planner Labels

The planner should learn the following surface categories.

Forensic

- boundary_blending
- face_landmarks
- eye_consistency
- reflections
- specular_highlights
- compression
- metadata
- frequency_analysis
- checkerboard_artifacts
- periodic_fft_peaks
- spectral_discontinuity
- high_frequency_energy
- sensor_noise

Semantic

- object_consistency
- scene_consistency
- lighting_consistency
- shadow_consistency
- perspective_consistency
- context_realism
- text_rendering
- OCR_errors
- watermark_detection

Retrieval

- generator_fingerprint
- reverse_search_candidate
- duplicate_pattern
- nearest_embedding_match
- style_similarity

-------------------------------------------------------------------------------

# Training Objective

Teach Adapter A

WHAT

to notice,

NOT

whether an image is fake.

The output should always be

surface observations

↓

dispatch

↓

confidence

↓

reasoning

Never

real/fake classification.

-------------------------------------------------------------------------------