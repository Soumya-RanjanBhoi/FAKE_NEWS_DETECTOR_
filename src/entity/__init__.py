from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfiguration:
    root_dir:Path
    tokenizer_path:Path
    model_path:Path