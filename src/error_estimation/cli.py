from __future__ import annotations

import sys

from error_estimation.experiments.run_ablation import main as run_ablation
from error_estimation.experiments.run_ablation_hyperparams import main as run_ablation_hyperparams
from error_estimation.experiments.run_detection import main as run_detection


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            """Usage:
  error-estimation run [args]
  error-estimation ablation [args]
  error-estimation ablation-hyperparams [args]

Use --help after a command for command-specific options."""
        )
        return

    command = args.pop(0)
    if command == "run":
        run_detection(args)
    elif command == "ablation":
        run_ablation(args)
    elif command in {"ablation-hyperparams", "ablation_hyperparams"}:
        run_ablation_hyperparams(args)
    else:
        raise SystemExit(f"Unknown command: {command}")
