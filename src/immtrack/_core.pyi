import numpy as np
from numpy.typing import NDArray

class CovarianceNotPsd(RuntimeError): ...
class InvalidArgument(ValueError): ...
class NumericalError(RuntimeError): ...

class UkfPosVxyzYawCV:
    N: int
    M: int

    def __init__(
        self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0
    ) -> None: ...
    def init(self, state: NDArray[np.float64], cov: NDArray[np.float64]) -> None: ...
    def predict(self, dt: float) -> None: ...
    def update(self, measurement: NDArray[np.float64]) -> float: ...
    @property
    def state(self) -> NDArray[np.float64]: ...
    @property
    def covariance(self) -> NDArray[np.float64]: ...

class BoundingBox:
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    rot: float
    class_name: str
    score: float
    track_id: int

    def __init__(
        self,
        x: float = ...,
        y: float = ...,
        z: float = ...,
        l: float = ...,
        w: float = ...,
        h: float = ...,
        rot: float = ...,
        class_name: str = ...,
        score: float = ...,
        track_id: int = ...,
    ) -> None: ...

class TrackedObject:
    id: int
    class_name: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    rot: float
    l: float
    w: float
    h: float
    score: float
    age: int
    hit_count: int
    miss_count: int

class BBoxTracker:
    def __init__(
        self,
        n_init: int = ...,
        max_age: int = ...,
        min_hits: int = ...,
        size_ema_alpha: float = ...,
    ) -> None: ...
    def update(
        self,
        detections: list[BoundingBox],
        dt: float | None = ...,
    ) -> list[TrackedObject]: ...
    def reset(self) -> None: ...
    def track_count(self) -> int: ...

class metrics:
    class MatchMetric:
        CenterDistance: "metrics.MatchMetric"
        Iou3d: "metrics.MatchMetric"

    class AmotaConfig:
        recall_values: list[float]
        metric: "metrics.MatchMetric"
        match_threshold: float
        def __init__(self) -> None: ...

    class AmotaResult:
        overall: float
        per_class: dict[str, float]

    @staticmethod
    def amota(
        gt: list[list[BoundingBox]],
        pred: list[list[BoundingBox]],
        cfg: "metrics.AmotaConfig" = ...,
    ) -> "metrics.AmotaResult": ...
