from pathlib import Path

from error_estimation.experiments.run_detection import main


def test_run_detection_dry_run(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    args = [
        "--config-dataset",
        str(repo_root / "configs" / "datasets" / "cifar10" / "cifar10_ablation.yml"),
        "--config-model",
        str(repo_root / "configs" / "models" / "cifar10_resnet34.yml"),
        "--config-detection",
        str(repo_root / "configs" / "postprocessors" / "conformal" / "cifar10_resnet34.yml"),
        "--root-dir",
        str(tmp_path / "results"),
        "--dry-run",
        "--no-mlflow",
    ]
    main(args)
    assert (tmp_path / "results" / "run.json").exists()
