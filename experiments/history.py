"""Serializable metric histories for experiments."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class HistoryRecord:
    """Metrics for one minibatch, optionally one hardware realization/time."""

    loss: float
    accuracy: float
    count: int
    batch: int
    seconds: float = 0.
    step: int = 0
    epoch: float = 0.
    realization_id: str | None = None
    time_index: int | None = None
    t_inference: float | None = None


@dataclass
class ExperimentHistory:
    """Training histories and final inference results for one experiment."""

    config: dict
    digital_baseline: dict
    digital_history: list[dict]
    oi_history: list[dict]
    results: list[dict]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + '\n')
