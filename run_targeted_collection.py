#!/usr/bin/env python3
from __future__ import annotations

import argparse

from revamp.launch.config import load_config
from revamp.launch.pipeline import run_targeted_collection_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect targeted rollouts at the flagged interactions of a round.")
    parser.add_argument(
        "--config",
        default="configs/turn_on_sink_faucet.json",
        help="Path to the JSON experiment config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned command without execution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_targeted_collection_pipeline(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
