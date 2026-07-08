import argparse
import json
import os
import sys
import time

from semantic_agent import SemanticContextAgent

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_images(folder, limit):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(root, f))
                if len(paths) >= limit:
                    return paths
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--skip-explain", action="store_true")
    args = parser.parse_args()

    images = find_images(args.folder, args.limit)
    if not images:
        print(f"No images found under {args.folder}")
        sys.exit(1)

    print(f"Found {len(images)} image(s) to test (limit={args.limit})\n")
    agent = SemanticContextAgent()

    results = []
    for path in images:
        print(f"--- {path} ---")
        t0 = time.time()
        try:
            findings = agent.run_checks(path)
            record = {"image": path, "findings": findings.__dict__}
            if args.skip_explain:
                print(json.dumps(record["findings"], indent=2))
            else:
                try:
                    explanation = agent.explain(findings)
                    record["explanation"] = explanation
                    print(json.dumps(record, indent=2))
                except Exception as e:
                    print(f"  [findings ok, explain() failed - is Ollama running? {e}]")
                    print(json.dumps(record["findings"], indent=2))
            results.append(record)
        except Exception as e:
            print(f"  [FAILED: {e}]")
        print(f"  ({time.time() - t0:.2f}s)\n")

    with open("semantic_agent_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} result(s) to semantic_agent_test_results.json")


if __name__ == "__main__":
    main()
