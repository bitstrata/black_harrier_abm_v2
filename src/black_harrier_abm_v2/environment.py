from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List
import pandas as pd


@dataclass
class Turbine:
    lat: float
    lon: float
    rotor_radius_m: float = 60.0
    rotor_min_m: float = 30.0
    rotor_max_m: float = 130.0


@dataclass
class Environment:
    habitat_sampler: Optional[Callable[[float, float, pd.Timestamp], float]] = None
    wind_sampler: Optional[Callable[[float, float, pd.Timestamp], Tuple[float, float]]] = None
    slope_sampler: Optional[Callable[[float, float], float]] = None
    turbines: List[Turbine] = field(default_factory=list)

    def habitat(self, lat: float, lon: float, t: pd.Timestamp) -> float:
        if not self.habitat_sampler:
            return 0.5
        try:
            val = float(self.habitat_sampler(lat, lon, t))
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.5

    def wind(self, lat: float, lon: float, t: pd.Timestamp) -> Tuple[float, float]:
        if not self.wind_sampler:
            return (0.0, 0.0)
        try:
            u, v = self.wind_sampler(lat, lon, t)
            return float(u), float(v)
        except Exception:
            return (0.0, 0.0)

    def slope(self, lat: float, lon: float) -> float:
        if not self.slope_sampler:
            return 0.0
        try:
            s = float(self.slope_sampler(lat, lon))
            return max(-1.0, min(1.0, s))
        except Exception:
            return 0.0