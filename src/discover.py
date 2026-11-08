"""EAP-IG discovery of the evidence circuit through CircuitKIT."""
import argparse
from pathlib import Path

from circuitkit.api import discover_circuit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--model", default="CohereLabs/tiny-aya-fire")
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--sparsity", type=float, default=0.7)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    discover_circuit({
        "model": {"name": args.model, "precision": args.precision},
        "data": {
            "type": "template",
            "path": args.pairs,
            "align_strategy": "filter",
            "template": {
                "clean_prompt": "{clean_prompt}",
                "corrupt_prompt": "{corrupt_prompt}",
                "clean_answer": "{clean_answer}",
                "corrupt_answer": "{corrupt_answer}",
            },
        },
        "discovery": {
            "algorithm": "eap-ig",
            "level": "node",
            "batch_size": 1,
            "ig_steps": 5,
            "data_params": {"num_examples": args.examples, "seed": 42, "cache_dir": "../runs/circuit_cache"},
            "cache_dir": "../runs/circuit_cache",
        },
        "pruning": {"target_sparsity": args.sparsity, "scope": "heads"},
        "output_path": args.out,
    })


if __name__ == "__main__":
    main()
