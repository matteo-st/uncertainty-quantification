from pathlib import Path

from error_estimation.cli import main


def test_cli_run_dry_run(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    args = [
        "run",
        "--config-dataset",
        str(repo_root / "configs" / "datasets" / "cifar10" / "cifar10_ablation.yml"),
        "--config-model",
        str(repo_root / "configs" / "models" / "cifar10_resnet34.yml"),
        "--config-detection",
        str(repo_root / "configs" / "postprocessors" / "conformal" / "cifar10_resnet34.yml"),
        "--root-dir",
        str(tmp_path / "run"),
        "--dry-run",
        "--no-mlflow",
    ]
    main(args)
    assert (tmp_path / "run" / "run.json").exists()
    assert (tmp_path / "run" / "configs" / "dataset.yml").exists()


def test_cli_ablation_dry_run(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    args = [
        "ablation",
        "--config-dataset",
        str(repo_root / "configs" / "datasets" / "cifar10" / "cifar10_ablation.yml"),
        "--config-model",
        str(repo_root / "configs" / "models" / "cifar10_resnet34.yml"),
        "--config-detection",
        str(repo_root / "configs" / "postprocessors" / "clustering" / "cifar10_resnet34_gmm.yml"),
        "--root-dir",
        str(tmp_path / "ablation"),
        "--dry-run",
        "--no-mlflow",
    ]
    main(args)
    assert (tmp_path / "ablation" / "run.json").exists()


def test_cli_ablation_hyperparams_dry_run(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    args = [
        "ablation-hyperparams",
        "--config-dataset",
        str(repo_root / "configs" / "datasets" / "cifar10" / "cifar10_ablation.yml"),
        "--config-model",
        str(repo_root / "configs" / "models" / "cifar10_resnet34.yml"),
        "--config-detection",
        str(
            repo_root
            / "configs"
            / "postprocessors"
            / "clustering"
            / "cifars_sklearn"
            / "clustering_cifar10_resnet34_ablation_hyperparams.yml"
        ),
        "--root-dir",
        str(tmp_path / "ablation-hyperparams"),
        "--dry-run",
        "--no-mlflow",
    ]
    main(args)
    assert (tmp_path / "ablation-hyperparams" / "run.json").exists()
