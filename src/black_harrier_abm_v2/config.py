from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import yaml
import pandas as pd


@dataclass
class InputsConfig:
    habitat_raster: str | None
    dem_raster: str | None
    wind_u_raster: str | None
    wind_v_raster: str | None
    crs_epsg: int | None = 4326


@dataclass
class WeightsConfig:
    w_habitat: float = 1.0
    w_wind: float = 0.6
    w_target: float = 0.8
    w_slope_penalty: float = 0.4
    w_turbine_avoid: float = 0.6
    softmax_beta: float = 2.5


@dataclass
class AltitudePriors:
    # mean/sd per state (m AGL)
    breeding_mean: float = 20.0
    breeding_sd: float = 10.0
    foraging_mean: float = 10.0
    foraging_sd: float = 6.0
    commuting_mean: float = 50.0
    commuting_sd: float = 20.0
    displaying_mean: float = 80.0
    displaying_sd: float = 20.0
    migrating_mean: float = 80.0
    migrating_sd: float = 20.0
    roosting_mean: float = 0.0
    roosting_sd: float = 1.0


@dataclass
class SimulationConfig:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    step_seconds: int = 60
    bsa_min_m: float = 30.0
    bsa_max_m: float = 130.0
    turbine_influence_m: float = 200.0


@dataclass
class ModelConfig:
    inputs: InputsConfig
    weights: WeightsConfig
    altitude_priors: AltitudePriors
    turbines: list[dict]
    agents: list[dict]


def load_config(path: str) -> tuple[SimulationConfig, ModelConfig]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    sim = cfg["simulation"]
    model = cfg["model"]

    sim_cfg = SimulationConfig(
        start_time=pd.to_datetime(sim["start_time"]),
        end_time=pd.to_datetime(sim["end_time"]),
        step_seconds=int(sim.get("step_seconds", 60)),
        bsa_min_m=float(sim.get("bsa_min_m", 30.0)),
        bsa_max_m=float(sim.get("bsa_max_m", 130.0)),
        turbine_influence_m=float(sim.get("turbine_influence_m", 200.0)),
    )

    inputs_cfg = InputsConfig(
        habitat_raster=model["inputs"].get("habitat_raster"),
        dem_raster=model["inputs"].get("dem_raster"),
        wind_u_raster=model["inputs"].get("wind_u_raster"),
        wind_v_raster=model["inputs"].get("wind_v_raster"),
        crs_epsg=int(model["inputs"].get("crs_epsg", 4326)),
    )

    weights_cfg = WeightsConfig(**model.get("weights", {}))
    altitude_cfg = AltitudePriors(**model.get("altitude_priors", {}))

    model_cfg = ModelConfig(
        inputs=inputs_cfg,
        weights=weights_cfg,
        altitude_priors=altitude_cfg,
        turbines=model.get("turbines", []),
        agents=model.get("agents", []),
    )

    return sim_cfg, model_cfg