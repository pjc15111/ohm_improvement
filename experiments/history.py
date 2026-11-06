"""Serializable metric histories for experiments."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class HistoryRecord:
    """Named metrics at one optimization step or evaluation.

    For example, metrics may contain classification loss/accuracy or WPO's
    objective. A step can be a minibatch update or an evolution generation.
    seconds measures elapsed time within the phase supplying this record.
    """

    metrics: dict[str, float]
    seconds: float = 0.
    step: int = 0
    count: int | None = None
    batch: int | None = None
    epoch: float | None = None
    realization_id: str | None = None
    time_index: int | None = None
    t_inference: float | None = None


@dataclass
class ExperimentHistory:
    """Method-independent optimization history and final inference results.

    Digital training is optional: separate optimization jobs can load the same
    checkpoint instead of training their own baseline.
    """

    method: str
    config: dict
    optimization_history: list[HistoryRecord | dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    digital_baseline: dict | None = None
    digital_history: list[dict] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + '\n')
