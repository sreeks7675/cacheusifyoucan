# Adapter A Planner Training Design
Version 2.0

------------------------------------------------------------------------------

# Objective

Adapter A is the orchestration planner of the multi-agent pipeline.

It DOES NOT determine whether an image is fake.

Instead, it performs rapid, explainable, surface-level triage and dispatches
the image to one or more specialist agents.

The planner should behave like an experienced investigator performing an
initial inspection before handing the evidence to specialists.

------------------------------------------------------------------------------

# Overall Pipeline

                Image
                  │
                  ▼
      ┌────────────────────┐
      │   Planner Agent    │
      │  (Adapter A LoRA)  │
      └────────────────────┘
                  │
     ┌────────────┼────────────┐
     │            │            │
     ▼            ▼            ▼
Forensic      Semantic     Retrieval
 Agent          Agent         Agent
     └────────────┼────────────┘
                  ▼
         Evidence Fusion
                  ▼
           Debate / Consensus
                  ▼
            Final Explanation

------------------------------------------------------------------------------

# Philosophy

Planner performs

Broad

Fast

Explainable

Low-cost reasoning.

Specialists perform

Detailed

Evidence-rich

Expensive reasoning.

------------------------------------------------------------------------------

# Planner Responsibilities

Planner should identify only SURFACE observations.

Examples

✓ boundary blending

✓ unusual FFT

✓ checkerboard artifacts

✓ strange lighting

✓ text artifacts

✓ impossible shadows

✓ possible generator fingerprint

✓ duplicate likelihood

Planner should NEVER conclude

"This is fake."

Planner instead concludes

"This image should be investigated by the forensic agent."

------------------------------------------------------------------------------

# Planner Inputs

Every sample contains

1 RGB Image

2 FFT Magnitude Image

3 Metadata Summary

4 Instruction Prompt

5 Ground Truth Dispatch

------------------------------------------------------------------------------

# RGB Input

RGB image provides

Object layout

Faces

Lighting

Scene context

Text

Watermarks

Composition

------------------------------------------------------------------------------

# FFT Input

FFT provides

Frequency distribution

Checkerboard artifacts

GAN upsampling traces

Periodic patterns

High-frequency anomalies

Compression signatures

Aliasing

Planner only recognizes

possible

frequency abnormalities.

Detailed interpretation belongs to the forensic agent.

------------------------------------------------------------------------------

# Metadata Input

Planner observes

Resolution

Aspect ratio

Compression level

EXIF availability

Camera metadata availability

Color profile

Timestamp availability

Planner simply notes

"Metadata missing"

or

"Metadata appears inconsistent"

------------------------------------------------------------------------------

# Surface Observations

Planner predicts observations instead of conclusions.

Example

surface_observations

[
"boundary blending",

"checkerboard frequency",

"eye asymmetry"
]

NOT

"This is a deepfake."

------------------------------------------------------------------------------

# Routing Categories

Three primary routing decisions exist.

1.

Forensic

2.

Semantic

3.

Retrieval

Multiple dispatches are allowed.

------------------------------------------------------------------------------

# Forensic Routing

Planner routes here when observing

Boundary blending

Frequency artifacts

Checkerboard FFT

Spectral spikes

Noise mismatch

Compression mismatch

Sensor inconsistencies

Eye abnormalities

Landmark inconsistencies

Reflection mismatch

Specular inconsistency

Metadata irregularities

------------------------------------------------------------------------------

# Semantic Routing

Planner routes here when observing

Object inconsistency

Scene inconsistency

Lighting mismatch

Shadow mismatch

Perspective inconsistency

Impossible geometry

Context inconsistency

OCR issues

Watermarks

Physical impossibilities

------------------------------------------------------------------------------

# Retrieval Routing

Planner routes here when observing

Possible diffusion image

Possible GAN image

Known manipulation family

Possible duplicate

Known generator fingerprint

Repeated texture signatures

Possible stock image

Internet image candidate

------------------------------------------------------------------------------

# Confidence

Planner predicts confidence

0.00

to

1.00

Confidence reflects

routing certainty

NOT

fake probability.

------------------------------------------------------------------------------

# Priority

Planner predicts

LOW

MEDIUM

HIGH

HIGH means

multiple routing cues detected

or

strong forensic evidence

------------------------------------------------------------------------------

# Dispatch Rules

Example

Only FFT anomaly

↓

Forensic

Only lighting mismatch

↓

Semantic

Generator fingerprint

↓

Retrieval

Lighting

+

FFT

↓

Forensic

+

Semantic

Generator

+

Boundary

↓

Retrieval

+

Forensic

------------------------------------------------------------------------------

# Multi-Agent Dispatch

Planner may dispatch

One

Two

or

Three

agents.

Examples

Forensic only

Semantic only

Retrieval only

Forensic + Semantic

Forensic + Retrieval

Semantic + Retrieval

All three

------------------------------------------------------------------------------

# Frequency-Domain Features

Planner learns shallow recognition of

High-frequency energy imbalance

Checkerboard spectra

Periodic peaks

Spectral discontinuities

Aliasing

GAN upsampling traces

Diffusion smoothing

FFT asymmetry

Planner never performs spectral analysis.

------------------------------------------------------------------------------

# Semantic Features

Planner learns

Lighting

Shadows

Object placement

Scene realism

Perspective

Watermarks

OCR

Text rendering

Reflection consistency

------------------------------------------------------------------------------

# Retrieval Features

Planner learns

Generator appearance

Stable Diffusion style

Flux style

GAN style

Known manipulation family

Repeated diffusion textures

Internet duplicate likelihood

------------------------------------------------------------------------------

# Output Schema

Planner returns

{
  "dispatch":[
      "forensic"
  ],

  "confidence":0.91,

  "priority":"HIGH",

  "surface_observations":[
      "checkerboard spectrum",
      "boundary blending",
      "compression mismatch"
  ],

  "reasoning":"Surface-level forensic cues detected. Recommend specialist analysis."
}

------------------------------------------------------------------------------

# Training Objective

Teach

routing

NOT

classification.

Wrong Objective

Image

↓

Fake

Correct Objective

Image

↓

Needs Forensic Agent

------------------------------------------------------------------------------

# Positive Examples

Planner should learn

Boundary blending

↓

Forensic

Scene inconsistency

↓

Semantic

Generator appearance

↓

Retrieval

FFT anomaly

↓

Forensic

Lighting

+

Perspective

↓

Semantic

FFT

+

Boundary

↓

Forensic

Diffusion appearance

↓

Retrieval

------------------------------------------------------------------------------

# Negative Examples

Real image

↓

No dispatch

Natural blur

↓

No forensic dispatch

JPEG compression only

↓

Low confidence forensic

Normal lighting variation

↓

No semantic dispatch

------------------------------------------------------------------------------

# Hard Negatives

Real compressed selfies

Real CCTV

Low-light photos

Motion blur

Portrait mode

Old scanned images

These prevent planner over-triggering.

------------------------------------------------------------------------------

# Dataset Balance

Recommended

40%

Real

60%

Manipulated

Manipulated

20%

Faceswap

20%

GAN

20%

Diffusion

20%

Compression

20%

Mixed

------------------------------------------------------------------------------

# Fine-Tuning Strategy

Backbone

Qwen2-VL-7B-Instruct

or

Phi-3.5-Vision-Instruct

Technique

LoRA

Vision encoder frozen

Planner head adapted

Mixed precision

Gradient checkpointing

Flash Attention

------------------------------------------------------------------------------

# Evaluation Metrics

Dispatch Accuracy

Dispatch Precision

Dispatch Recall

F1

Confidence Calibration

False Dispatch Rate

Average Routing Latency

------------------------------------------------------------------------------

# Success Criteria

Planner should

✔ Detect surface forensic cues.

✔ Detect semantic inconsistencies.

✔ Detect retrieval hints.

✔ Explain why routing occurred.

✔ Avoid deep reasoning.

✔ Route to appropriate specialists.

------------------------------------------------------------------------------

End of File