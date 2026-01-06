import torch
from torch.utils.data import DataLoader, TensorDataset

from error_estimation.utils.eval import AblationDetector
from error_estimation.utils.postprocessors import get_postprocessor


class ConstantLogitsModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(inputs.shape[0], -1)


def test_gini_postprocessor_smoke(tmp_path):
    device = torch.device("cpu")
    num_classes = 2
    n_samples = 10

    inputs = torch.zeros((n_samples, 3, 4, 4))
    labels = torch.tensor([0, 1] * (n_samples // 2), dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

    model = ConstantLogitsModel(torch.tensor([[5.0, 0.0]]))
    cfg_dataset = {"num_classes": num_classes}

    detector = AblationDetector(
        model=model,
        dataloader=loader,
        device=device,
        suffix="test",
        latent_path=str(tmp_path / "latents.pt"),
        postprocessor_name="doctor",
        cfg_dataset=cfg_dataset,
        result_folder=str(tmp_path),
    )

    cfg = {"temperature": 1.0, "magnitude": 0.0, "normalize": True}
    postprocessor = get_postprocessor(
        postprocessor_name="doctor",
        model=model,
        cfg=cfg,
        result_folder=str(tmp_path),
        device=device,
    )

    results = detector.evaluate([cfg], detectors=[postprocessor], suffix="test")
    assert results
    assert "roc_auc_test" in results[0].columns
