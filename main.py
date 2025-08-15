from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd

from src.black_harrier_abm_v2.config import SimulationConfig, ModelConfig, WeightsConfig, load_config
from src.black_harrier_abm_v2.environment import Environment, Turbine
from src.black_harrier_abm_v2.samplers.raster import RasterSamplers
from src.black_harrier_abm_v2.agents import HarrierAgent
from src.black_harrier_abm_v2.movement import (
    StepSelectionPolicy,
    AltitudeModel,
    AltitudePriors as MovAltitudePriors,
    State,
    Weights as MovWeights,
)
from src.black_harrier_abm_v2.risk import ABMSimulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Black Harrier ABM runner")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--out", required=True, help="Output CSV path")
    return p.parse_args()


def build_env(cfg: ModelConfig) -> Environment:
    if cfg.inputs.crs_epsg is None:
        raise ValueError("crs_epsg must not be None")
    samplers = RasterSamplers(
        habitat_raster=cfg.inputs.habitat_raster,
        dem_raster=cfg.inputs.dem_raster,
        wind_u_raster=cfg.inputs.wind_u_raster,
        wind_v_raster=cfg.inputs.wind_v_raster,
        crs_epsg=cfg.inputs.crs_epsg,
    )

    env = Environment(
        habitat_sampler=samplers.habitat,
        wind_sampler=samplers.wind,
        slope_sampler=samplers.slope,  # simple slope from DEM
        turbines=[Turbine(**t) for t in cfg.turbines],
    )
    return env


def to_movement_weights(w: WeightsConfig) -> MovWeights:
    """Adapter: config.WeightsConfig -> movement.Weights"""
    return MovWeights(
        w_habitat=w.w_habitat,
        w_wind=w.w_wind,
        w_target=w.w_target,
        w_slope_penalty=w.w_slope_penalty,
        w_turbine_avoid=w.w_turbine_avoid,
        softmax_beta=w.softmax_beta,
    )


def build_agents(cfg: ModelConfig) -> list[HarrierAgent]:
    def _pair(lat: Optional[float], lon: Optional[float]) -> Optional[tuple[float, float]]:
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)

    agents: list[HarrierAgent] = []
    for a in cfg.agents:
        # Required keys via TypedDict
        start_lat: float = float(a["start_lat"])  # ensure float type
        start_lon: float = float(a["start_lon"])
        state_str: str = a.get("state", State.BREEDING)
        sex_str: str = a.get("sex", "unknown")

        nest = _pair(a.get("nest_lat"), a.get("nest_lon"))
        roost = _pair(a.get("roost_lat"), a.get("roost_lon"))

        agents.append(
            HarrierAgent(
                agent_id=int(a["id"]),
                lat=start_lat,
                lon=start_lon,
                state=state_str,
                sex=sex_str,
                nest=nest,
                roost=roost,
            )
        )
    return agents


def build_altitude_priors(model_cfg: ModelConfig) -> MovAltitudePriors:
    ap = model_cfg.altitude_priors
    return MovAltitudePriors(
        means={
            "breeding": ap.breeding_mean,
            "foraging": ap.foraging_mean,
            "commuting": ap.commuting_mean,
            "displaying": ap.displaying_mean,
            "migrating": ap.migrating_mean,
            "roosting": ap.roosting_mean,
        },
        sds={
            "breeding": ap.breeding_sd,
            "foraging": ap.foraging_sd,
            "commuting": ap.commuting_sd,
            "displaying": ap.displaying_sd,
            "migrating": ap.migrating_sd,
            "roosting": ap.roosting_sd,
        },
    )


def main() -> None:
    args = parse_args()
    sim_cfg, model_cfg = load_config(args.config)

    env = build_env(model_cfg)

    policy = StepSelectionPolicy(weights=to_movement_weights(model_cfg.weights))
    alt_priors = build_altitude_priors(model_cfg)
    alt_model = AltitudeModel(priors=alt_priors)

    agents = build_agents(model_cfg)

    sim = ABMSimulation(
        env=env,
        agents=agents,
        policy=policy,
        alt_model=alt_model,
        sim_cfg=sim_cfg,
    )

    df: pd.DataFrame = sim.run()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved tracks: {out_path}")


if __name__ == "__main__":
    main()