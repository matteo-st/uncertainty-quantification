from pathlib import Path

from error_estimation.utils.config import Config


def test_config_loads():
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "datasets" / "cifar10" / "cifar10_ablation.yml"
    cfg = Config(cfg_path)
    assert "name" in cfg
    assert "n_samples" in cfg
