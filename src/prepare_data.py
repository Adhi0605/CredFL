"""Command-line entry point for the complete Week 1 data workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_and_validate, save_week1_outputs, summarize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/creditcard.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--num-clients", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-fraud", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_and_validate(args.input)
    summary = summarize_dataset(frame)
    manifest = save_week1_outputs(
        frame,
        args.output,
        num_clients=args.num_clients,
        alpha=args.alpha,
        test_size=args.test_size,
        seed=args.seed,
        min_fraud_per_client=args.min_fraud,
    )
    print(f"Validated {summary['rows']:,} rows; fraud rate={summary['fraud_rate']:.4%}")
    print("\nClient partition manifest:")
    print(manifest.to_string(index=False))
    print(f"\nSaved Week 1 outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
