"""
Quick sanity-check harness for semantic_agent.py

Usage:
    python test_semantic_agent.py <folder_with_images> [--limit N]

Point <folder_with_images> at an extracted dataset folder that contains
BOTH a 0_real and a 1_fake subfolder (e.g. from mfv, or an ffpp method
folder) once you've untarred it.

IMPORTANT: this harness now explicitly requires images from BOTH classes.
It will not silently run on "whatever it finds" — if either 0_real or
1_fake is missing/empty, it stops and tells you, unless you pass
--allow-single-class to override for a quick one-off check.

For every image tested, the harness attaches the ground-truth label
("real" / "fake") to the agent's output (the agent itself never sees or
uses this label - it's just for your own eyeballing/bookkeeping), and
prints a small summary at the end broken down by class.
"""

import argparse
import json
import os
import sys
import time

from semantic_agent import SemanticContextAgent

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_images(folder):
    """
    Walk `folder` and bucket every image into real / fake / other based on
    whether "0_real" or "1_fake" appears in its path. Returns three sorted
    lists of paths: (real_images, fake_images, other_images).
    """
    real_images = []
    fake_images = []
    other_images = []

    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                path = os.path.join(root, f)

                if "0_real" in root:
                    real_images.append(path)
                elif "1_fake" in root:
                    fake_images.append(path)
                else:
                    other_images.append(path)

    real_images.sort()
    fake_images.sort()
    other_images.sort()

    return real_images, fake_images, other_images


def build_sample(real_images, fake_images, other_images, limit, allow_single_class):
    """
    Build a labeled sample list of (path, label) tuples, explicitly split
    ~half real / half fake up to `limit`. Raises a clear error if one class
    is missing, unless allow_single_class is set.
    """
    if not real_images and not fake_images:
        if not other_images:
            print(f"No images found at all.")
            sys.exit(1)
        print(
            "WARNING: no '0_real' or '1_fake' subfolders detected — "
            "found images but couldn't tell which class they belong to. "
            "Falling back to untagged 'unknown' labels.\n"
        )
        sample = [(p, "unknown") for p in other_images[:limit]]
        return sample

    if not real_images or not fake_images:
        missing = "0_real" if not real_images else "1_fake"
        msg = (
            f"Only found images for one class — the '{missing}' folder is "
            f"missing or empty under this path. This harness is meant to "
            f"test BOTH real and fake images together.\n"
            f"  real found: {len(real_images)}\n"
            f"  fake found: {len(fake_images)}\n"
        )
        if allow_single_class:
            print("WARNING: " + msg + "Continuing anyway (--allow-single-class set).\n")
        else:
            print(
                "ERROR: " + msg +
                "Point the folder at a location containing both classes, "
                "or re-run with --allow-single-class to proceed anyway.\n"
            )
            sys.exit(1)

    half = limit // 2
    remainder = limit - half

    chosen_real = real_images[:half]
    chosen_fake = fake_images[:remainder]

    # If one class ran short, backfill from the other so we still return
    # up to `limit` total images (best-effort, not required to be balanced).
    shortfall = limit - (len(chosen_real) + len(chosen_fake))
    if shortfall > 0:
        extra_real = real_images[len(chosen_real):len(chosen_real) + shortfall]
        chosen_real += extra_real
        shortfall -= len(extra_real)
    if shortfall > 0:
        extra_fake = fake_images[len(chosen_fake):len(chosen_fake) + shortfall]
        chosen_fake += extra_fake

    sample = [(p, "real") for p in chosen_real] + [(p, "fake") for p in chosen_fake]
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder",
        help="Folder containing images (searched recursively)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Max number of images to test, split as evenly as possible "
             "between real and fake"
    )
    parser.add_argument(
        "--skip-explain",
        action="store_true",
        help="Skip the Ollama call and just print raw findings "
             "(use this if the backbone isn't served yet)"
    )
    parser.add_argument(
        "--allow-single-class",
        action="store_true",
        help="Proceed even if only real or only fake images are found "
             "(by default the harness stops and asks for both)"
    )

    args = parser.parse_args()

    real_images, fake_images, other_images = find_images(args.folder)
    sample = build_sample(
        real_images, fake_images, other_images, args.limit, args.allow_single_class
    )

    if not sample:
        print(f"No images found under {args.folder}")
        sys.exit(1)

    n_real = sum(1 for _, label in sample if label == "real")
    n_fake = sum(1 for _, label in sample if label == "fake")
    n_unknown = sum(1 for _, label in sample if label == "unknown")

    print(
        f"Found {len(sample)} image(s) to test "
        f"(real={n_real}, fake={n_fake}, unknown={n_unknown}, limit={args.limit})\n"
    )

    agent = SemanticContextAgent()

    results = []

    for path, label in sample:
        print(f"--- [{label}] {path} ---")
        t0 = time.time()

        try:
            findings = agent.run_checks(path)
            record = {
                "image": path,
                "true_label": label,
                "findings": findings.__dict__,
            }

            if args.skip_explain:
                print(json.dumps(record, indent=2))
            else:
                try:
                    explanation = agent.explain(findings)
                    record["explanation"] = explanation
                    print(json.dumps(record, indent=2))
                except Exception as e:
                    print(
                        f"  [findings ok, explain() failed - "
                        f"is Ollama running? {e}]"
                    )
                    print(json.dumps(record, indent=2))

            results.append(record)

        except Exception as e:
            print(f"  [FAILED: {e}]")
            results.append({"image": path, "true_label": label, "error": str(e)})

        print(f"  ({time.time() - t0:.2f}s)\n")

    out_path = "semantic_agent_test_results.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary by class so it's obvious at a glance both sides were exercised.
    print("=== Summary ===")
    for label in ("real", "fake", "unknown"):
        subset = [r for r in results if r.get("true_label") == label]
        if subset:
            failed = sum(1 for r in subset if "error" in r)
            print(f"  {label}: {len(subset)} tested, {failed} failed")

    print(f"\nWrote {len(results)} result(s) to {out_path}")


if __name__ == "__main__":
    main()
