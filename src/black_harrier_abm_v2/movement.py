from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import random
import numpy as np
import pandas as pd

from .utils.geo import destination_point, haversine_m


class State:
    BREEDING = "breeding"
    FORAGING = "foraging"
    COMMUTING = "commuting"
    DISPLAYING = "displaying"
    MIGRATING = "migrating"
    ROOSTING = "roosting"


@dataclass
class Weights:
    w_habitat: float = 1.0
    w_wind: float = 0.6
    w_target: float = 0.8
    w_slope_penalty: float = 0.4
    w_turbine_avoid: float = 0.6
    softmax_beta: float = 2.5


@dataclass
class AltitudePriors:
    means: Dict[str, float]
    sds: Dict[str, float]

    def sample(self, state: str, rng: random.Random) -> float:
        m = float(self.means.get(state, 10.0))
        s = float(self.sds.get(state, 3.0))
        return max(0.0, rng.gauss(m, s))


@dataclass
class StepSelectionPolicy:
    weights: Weights

    def choose_heading(
        self,
        lat: float,
        lon: float,
        speed_mps: float,
        t: pd.Timestamp,
        env,
        state: str,
        target: Tuple[float, float] | None,
        rng: random.Random,
    ) -> float:
        headings = np.arange(0, 360, 15)
        dist = max(0.1, speed_mps) * 60.0  # assumes 60s step; see Simulation for generalization

        u, v = env.wind(lat, lon, t)
        wind_speed = math.hypot(u, v)
        wind_dir = (math.degrees(math.atan2(u, v)) + 360) % 360 if wind_speed > 0 else None

        utilities: List[float] = []
        for hdg in headings:
            lat2, lon2 = destination_point(lat, lon, hdg, dist)
            h = env.habitat(lat2, lon2, t)

            wind_util = 0.0
            if wind_dir is not None:
                diff = abs(((hdg - wind_dir) + 180) % 360 - 180)
                wind_util = math.cos(math.radians(diff))

            target_util = 0.0
            if target is not None:
                d0 = haversine_m(lat, lon, target[0], target[1])
                d1 = haversine_m(lat2, lon2, target[0], target[1])
                target_util = (d0 - d1) / max(1.0, d0)

            slope_pen = -abs(env.slope(lat2, lon2))

            turbine_pen = 0.0
            for tb in env.turbines:
                from .utils.geo import haversine_m as hdist
                d = hdist(lat2, lon2, tb.lat, tb.lon)
                if d < tb.rotor_radius_m * 1.5:
                    turbine_pen -= (1.5 - d / (tb.rotor_radius_m * 1.5))

            U = (
                self.weights.w_habitat * h
                + self.weights.w_wind * wind_util
                + self.weights.w_target * target_util
                + self.weights.w_slope_penalty * slope_pen
                + self.weights.w_turbine_avoid * turbine_pen
            )
            utilities.append(U)

        beta = self.weights.softmax_beta
        utilities_np = np.array(utilities)
        weights = np.exp(beta * (utilities_np - utilities_np.max()))
        probs = weights / weights.sum()
        return float(rng.choices(list(headings), weights=list(probs), k=1)[0])


@dataclass
class AltitudeModel:
    priors: AltitudePriors

    def sample(self, state: str, rng: random.Random) -> float:
        return self.priors.sample(state, rng)