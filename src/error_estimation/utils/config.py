from pathlib import Path

import yaml


def require_keys(cfg: dict, keys: list[str], label: str) -> None:
    missing = [key for key in keys if key not in cfg]
    if missing:
        raise KeyError(f"Missing keys in {label} config: {', '.join(missing)}")

class Config(dict):
    def __init__(self, path):
        super().__init__()
        self.path = str(Path(path))  # keep the path if you want
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError(
                f"Top-level YAML must be a mapping; got {type(data).__name__}"
            )
        self.update(data)  # <-- initialize the dict in place

    def require(self, keys: list[str], label: str | None = None) -> None:
        require_keys(self, keys, label or self.path)
