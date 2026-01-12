import torch

from .base_postprocessor import BasePostprocessor


class MarginPostprocessor(BasePostprocessor):
    def __init__(self, model, cfg, result_folder, device=torch.device("cpu")):
        super().__init__(model, cfg, result_folder, device)

        self.temperature = cfg["temperature"]
        self.magnitude = cfg["magnitude"]

    def __call__(self, inputs=None, logits=None):
        if logits is None:
            if inputs is None:
                raise ValueError("Either logits or inputs must be provided")
            logits = self.model(inputs)

        probs = torch.softmax(logits / self.temperature, dim=1)
        top2 = torch.topk(probs, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        return 1.0 - margin
