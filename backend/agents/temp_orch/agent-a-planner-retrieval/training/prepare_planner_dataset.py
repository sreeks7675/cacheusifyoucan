"""
prepare_planner_dataset.py

Builds instruction-tuning data for Adapter A.

Adapter A is NOT a forensic detector.

It is a routing planner.

Inputs
------

RGB Image

FFT Magnitude

Metadata Summary

Outputs

Instruction tuning JSONL
for

Qwen2-VL

Phi-3.5 Vision

LLaVA

etc.
"""

import json
import random
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLANNER_ROOT = PROJECT_ROOT / "planner_dataset"

MANIFEST = PLANNER_ROOT / "planner_manifest.csv"

OUTPUT = PROJECT_ROOT / "training" / "data" / "adapter_a_training_samples.jsonl"

random.seed(42)

# ---------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are Adapter A.

You are the orchestration planner.

Your job is NOT to determine whether the image is fake.

Your responsibility is to perform ONLY shallow inspection.

You must decide which specialist agent(s) should analyze the image.

Possible agents

- forensic

- semantic

- retrieval

You may choose multiple agents.

Never perform deep reasoning.

Always explain which surface observations caused the routing.
"""

# ---------------------------------------------------------
# Observation Libraries
# ---------------------------------------------------------

FORENSIC_OBS = [

"boundary blending",

"boundary discontinuity",

"frequency anomaly",

"checkerboard spectrum",

"periodic FFT peaks",

"high-frequency imbalance",

"compression mismatch",

"sensor noise inconsistency",

"landmark inconsistency",

"eye asymmetry",

"reflection mismatch",

"specular inconsistency",

"metadata anomaly"

]

SEMANTIC_OBS = [

"lighting mismatch",

"shadow inconsistency",

"object inconsistency",

"scene inconsistency",

"watermark artifact",

"OCR artifact",

"impossible geometry",

"context mismatch",

"reflection inconsistency"

]

RETRIEVAL_OBS = [

"possible diffusion appearance",

"possible GAN appearance",

"known manipulation family",

"possible duplicate",

"possible stock image",

"generator fingerprint hint",

"texture repetition"

]

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def choose_confidence(dispatch):

    if len(dispatch) == 1:

        return round(random.uniform(0.82,0.95),2)

    if len(dispatch)==2:

        return round(random.uniform(0.88,0.97),2)

    return round(random.uniform(0.90,0.99),2)

# ---------------------------------------------------------

def build_observations(dispatch):

    obs=[]

    if "forensic" in dispatch:

        obs.extend(random.sample(FORENSIC_OBS,3))

    if "semantic" in dispatch:

        obs.extend(random.sample(SEMANTIC_OBS,2))

    if "retrieval" in dispatch:

        obs.extend(random.sample(RETRIEVAL_OBS,2))

    return obs

# ---------------------------------------------------------

def reasoning(dispatch):

    reasons=[]

    if "forensic" in dispatch:

        reasons.append(
            "surface forensic indicators detected"
        )

    if "semantic" in dispatch:

        reasons.append(
            "possible semantic inconsistencies detected"
        )

    if "retrieval" in dispatch:

        reasons.append(
            "possible known manipulation patterns detected"
        )

    return "; ".join(reasons)

# ---------------------------------------------------------
# Routing Rules
# ---------------------------------------------------------

def planner_dispatch(row):

    family=row["family"]

    dispatch=[]

    if family=="real":

        dispatch=[]

    elif family=="faceswap":

        dispatch=["forensic"]

    elif family=="gan":

        dispatch=["forensic","retrieval"]

    elif family=="diffusion":

        dispatch=["semantic","retrieval"]

    else:

        dispatch=["forensic"]

    return dispatch

# ---------------------------------------------------------
# Conversation Builder
# ---------------------------------------------------------

if not MANIFEST.exists():
    raise RuntimeError(
        f"{MANIFEST} not found. Run organize_raw_datasets.py then "
        "prepare_dataset.py first."
    )

manifest=pd.read_csv(MANIFEST)

if manifest.empty:
    raise RuntimeError(
        f"{MANIFEST} exists but has zero rows -- check prepare_dataset.py's "
        "output before re-running this script."
    )

print()

print("="*70)

print("Generating Planner JSONL")

print("="*70)

count=0

with open(OUTPUT,"w") as writer:

    for _,row in manifest.iterrows():

        dispatch=planner_dispatch(row)

        observations=build_observations(dispatch)

        response={

            "dispatch":dispatch,

            "confidence":choose_confidence(dispatch),

            "priority":

                "HIGH"

                if len(dispatch)>=2

                else "MEDIUM",

            "surface_observations":

                observations,

            "reasoning":

                reasoning(dispatch)

        }

        sample={

            "messages":[

                {

                    "role":"system",

                    "content":SYSTEM_PROMPT

                },

                {

                    "role":"user",

                    "content":[

                        {

                            "type":"image",

                            "image":row["rgb"]

                        },

                        {

                            "type":"image",

                            "image":row["fft"]

                        },

                        {

                            "type":"text",

                            "text":

                            f"""

Metadata

{row["metadata"]}

Inspect the RGB image together with the frequency spectrum.

Perform ONLY surface inspection.

Return valid JSON.

"""

                        }

                    ]

                },

                {

                    "role":"assistant",

                    "content":

                        json.dumps(

                            response,

                            indent=2

                        )

                }

            ]

        }

        writer.write(

            json.dumps(sample)

        )

        writer.write("\n")

        count+=1

print()

print("="*70)

print("Planner Dataset Complete")

print()

print(f"Samples : {count}")

print()

print("Saved to")

print(OUTPUT)

print("="*70)