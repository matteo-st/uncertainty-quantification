"""Legacy experiment runner (deprecated)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Legacy runner (deprecated).")
    parser.add_argument("--help-only", action="store_true", help="Print this message and exit")
    parser.parse_args(argv)
    raise NotImplementedError(
        "run_clean.py was deprecated during the package refactor. "
        "Please use run_detection.py or run_ablation.py."
    )


if __name__ == "__main__":
    main()
