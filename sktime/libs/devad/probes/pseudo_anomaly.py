from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PseudoAnomaly:
    injected: np.ndarray
    label: np.ndarray
    intervals: list[tuple[int, int]]


class PseudoAnomalyGenerator:
    """Generate same-length pseudo anomalies with template-specific strength.

    spike / level_shift: maximum offset in local-scale units.
    local_noise: noise standard deviation in local-scale units.
    amplitude_scale: multiplier around the segment median.
    quantization: quantization step in local-scale units, centered on the median.

    Labels and anomaly_rate describe whole event intervals, not only changed points.
    Local scale is estimated from the clean context around each event.
    """

    MIN_GAP = 50
    SCALE_CONTEXT = 50

    ANOMALY_METHODS = {
        "spike": "_add_spike",
        "level_shift": "_add_level_shift",
        "local_noise": "_add_local_noise",
        "amplitude_scale": "_scale_amplitude",
        "quantization": "_quantize",
    }

    def __init__(
        self,
        anomaly_type: str,
        length_range: tuple[int, int],
        strength_range: tuple[float, float],
        anomaly_rate: float,
        seed: int,
    ):
        if anomaly_type not in self.ANOMALY_METHODS:
            raise ValueError(f"Unsupported anomaly type: {anomaly_type}")

        length_min, length_max = length_range
        if length_min <= 0 or length_min > length_max:
            raise ValueError("length_range must contain two positive ordered values")

        strength_min, strength_max = strength_range
        if strength_min <= 0 or strength_min > strength_max:
            raise ValueError("strength_range must contain two positive ordered values")

        if anomaly_type == "amplitude_scale" and strength_min == strength_max == 1:
            raise ValueError("Amplitude factor 1 would leave the segment unchanged")

        if not 0 < anomaly_rate < 1:
            raise ValueError("anomaly_rate must be between 0 and 1")

        self.anomaly_type = anomaly_type
        self.length_range = length_range
        self.strength_range = strength_range
        self.anomaly_rate = anomaly_rate
        self.seed = seed

    def generate(self, x: np.ndarray) -> PseudoAnomaly:
        clean = np.asarray(x, dtype=np.float32)
        if clean.ndim != 1:
            raise ValueError(f"Time series must have shape (n,), got {clean.shape}")

        rng = np.random.default_rng(self.seed)
        injected = clean.copy()
        label = np.zeros(len(clean), dtype=np.int64)
        intervals: list[tuple[int, int]] = []

        remaining = round(len(clean) * self.anomaly_rate)
        length_min, _ = self.length_range
        if remaining < length_min:
            raise ValueError("Anomaly budget is smaller than the minimum anomaly length")

        inject_method = getattr(
            self,
            self.ANOMALY_METHODS[self.anomaly_type],
        )

        while remaining >= length_min:
            length = self._sample_length(rng, remaining)
            start, end, scale = self._sample_interval(
                clean=clean,
                label=label,
                length=length,
                rng=rng,
            )
            strength = float(rng.uniform(*self.strength_range))

            # 注入
            injected[start:end] = inject_method(
                segment=clean[start:end],
                strength=strength,
                scale=scale,
                rng=rng,
            )

            # 同步 Label
            label[start:end] = 1
            intervals.append((start, end))
            remaining -= length

        intervals.sort()
        return PseudoAnomaly(
            injected=injected,
            label=label,
            intervals=intervals,
        )

    def _sample_length(
        self,
        rng: np.random.Generator,
        remaining: int,
    ) -> int:
        length_min, length_max = self.length_range
        upper = min(length_max, remaining)

        if remaining <= length_max:
            return remaining

        safe_upper = min(upper, remaining - length_min)
        if safe_upper >= length_min:
            upper = safe_upper

        return int(rng.integers(length_min, upper + 1))

    @classmethod
    def _sample_interval(
        cls,
        clean: np.ndarray,
        label: np.ndarray,
        length: int,
        rng: np.random.Generator,
    ) -> tuple[int, int, float]:
        if length > len(clean):
            raise ValueError("Anomaly length exceeds the input length")

        for _ in range(1000):
            start = int(rng.integers(0, len(clean) - length + 1))
            end = start + length

            blocked_start = max(0, start - cls.MIN_GAP)
            blocked_end = min(len(clean), end + cls.MIN_GAP)
            if label[blocked_start:blocked_end].any():
                continue

            context_start = max(0, start - cls.SCALE_CONTEXT)
            context_end = min(len(clean), end + cls.SCALE_CONTEXT)
            context = clean[context_start:context_end]
            q25, q75 = np.quantile(context, [0.25, 0.75])
            scale = float((q75 - q25) / 1.349)
            if scale > 1e-8:
                return start, end, scale

        raise ValueError("Unable to sample a valid non-overlapping anomaly interval")

    @staticmethod
    def _add_spike(
        segment: np.ndarray,
        strength: float,
        scale: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        direction = float(rng.choice((-1.0, 1.0)))
        amplitude = direction * strength * scale
        triangle = np.bartlett(len(segment) + 2)[1:-1]
        return segment + amplitude * triangle

    @staticmethod
    def _add_level_shift(
        segment: np.ndarray,
        strength: float,
        scale: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        direction = float(rng.choice((-1.0, 1.0)))
        return segment + direction * strength * scale

    @staticmethod
    def _add_local_noise(
        segment: np.ndarray,
        strength: float,
        scale: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        noise = rng.normal(0.0, strength * scale, size=len(segment))
        return segment + noise

    @staticmethod
    def _scale_amplitude(
        segment: np.ndarray,
        strength: float,
        scale: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        baseline = np.median(segment)
        return baseline + strength * (segment - baseline)

    @staticmethod
    def _quantize(
        segment: np.ndarray,
        strength: float,
        scale: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        baseline = np.median(segment)
        step = strength * scale
        return baseline + step * np.round((segment - baseline) / step)


def mean_interval_percentile_lift(
    clean_scores: np.ndarray,
    injected_scores: np.ndarray,
    intervals: list[tuple[int, int]],
) -> float:
    """Return the event-weighted mean percentile increase after injection.

    Both score arrays must already be time-aligned, with higher scores meaning
    more anomalous. Intervals are [start, end) indices into these score arrays,
    not the original unaligned series. Each interval receives equal weight.

    Both arrays use F(v) = count(clean_scores <= v) / len(clean_scores).
    Negative increases are preserved. "Clean" means unmodified, not guaranteed
    anomaly-free. This function does not load models, align scores or save files.
    """
    clean_scores = np.asarray(clean_scores, dtype=np.float64)
    injected_scores = np.asarray(injected_scores, dtype=np.float64)
    if clean_scores.ndim != 1 or injected_scores.ndim != 1:
        raise ValueError("Scores must be one-dimensional")
    if clean_scores.size == 0 or clean_scores.shape != injected_scores.shape:
        raise ValueError("Scores must be non-empty and have the same length")
    if not np.isfinite(clean_scores).all() or not np.isfinite(injected_scores).all():
        raise ValueError("Scores must contain only finite values")
    if not intervals:
        raise ValueError("At least one anomaly interval is required")
    for start, end in intervals:
        if not 0 <= start < end <= len(clean_scores):
            raise ValueError(f"Invalid score interval: ({start}, {end})")

    reference = np.sort(clean_scores)
    clean_percentiles = (
        np.searchsorted(reference, clean_scores, side="right") / len(reference)
    )
    injected_percentiles = (
        np.searchsorted(reference, injected_scores, side="right") / len(reference)
    )
    lift = injected_percentiles - clean_percentiles

    interval_means = [lift[start:end].mean() for start, end in intervals]
    return float(np.mean(interval_means))
