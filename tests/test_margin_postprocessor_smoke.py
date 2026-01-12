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


def test_margin_postprocessor_smoke(tmp_path):
    device = torch.device("cpu")
    num_classes = 3
    n_samples = 12

    inputs = torch.zeros((n_samples, 3, 4, 4))
    labels = torch.tensor([0, 1, 2, 0, 1, 2] * 2, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=3)

    model = ConstantLogitsModel(torch.tensor([[3.0, 2.0, 1.0]]))
    cfg_dataset = {"num_classes": num_classes}

    detector = AblationDetector(
        model=model,
        dataloader=loader,
        device=device,
        suffix="test",
        latent_path=str(tmp_path / "latents.pt"),
        postprocessor_name="margin",
        cfg_dataset=cfg_dataset,
        result_folder=str(tmp_path),
    )

    cfg = {"temperature": 1.0, "magnitude": 0.0}
    postprocessor = get_postprocessor(
        postprocessor_name="margin",
        model=model,
        cfg=cfg,
        result_folder=str(tmp_path),
        device=device,
    )

    results = detector.evaluate([cfg], detectors=[postprocessor], suffix="test")
    assert results
    assert "roc_auc_test" in results[0].columns
