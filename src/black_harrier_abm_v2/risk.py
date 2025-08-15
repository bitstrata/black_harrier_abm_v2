from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd

from .environment import Environment
from .agents import HarrierAgent, month_to_season
from .movement import State, StepSelectionPolicy, AltitudeModel
from .utils.geo import destination_point
from .config import SimulationConfig


@dataclass
class ABMSimulation:
    env: Environment
    agents: List[HarrierAgent]
    policy: StepSelectionPolicy
    alt_model: AltitudeModel
    sim_cfg: SimulationConfig


    def run(self) -> pd.DataFrame:
        records: List[Dict] = []
        for ag in self.agents:
            ag.step_seconds = self.sim_cfg.step_seconds

        t = self.sim_cfg.start_time
        while t <= self.sim_cfg.end_time:
            for ag in self.agents:
                # Update state
                season = month_to_season(t)
                ag.state = ag.next_state(season)

                # Speed & target selection
                speed = max(0.1, ag.sample_speed())
                target = ag.nest if ag.state in (State.BREEDING, State.COMMUTING, State.DISPLAYING) and ag.nest else (
                    ag.roost if ag.state in (State.ROOSTING, State.FORAGING) and ag.roost else None
                )

                # Choose heading via policy
                hdg = self.policy.choose_heading(
                    lat=ag.lat, lon=ag.lon, speed_mps=speed, t=t, env=self.env, state=ag.state, target=target, rng=ag.rng
                )

                # Move
                dist = speed * self.sim_cfg.step_seconds
                lat2, lon2 = destination_point(ag.lat, ag.lon, hdg, dist)
                alt = self.alt_model.sample(ag.state, ag.rng)
                if ag.state in (State.DISPLAYING, State.MIGRATING, State.COMMUTING):
                    alt *= 1.1

                ag.lat, ag.lon = lat2, lon2

                # Risk proxy
                in_bsa = False
                nearest_id = None
                nearest_d = None
                for i, tb in enumerate(self.env.turbines):
                    from .utils.geo import haversine_m
                    d = haversine_m(lat2, lon2, tb.lat, tb.lon)
                    if d <= self.sim_cfg.turbine_influence_m and (self.sim_cfg.bsa_min_m <= alt <= self.sim_cfg.bsa_max_m):
                        in_bsa = True
                        if nearest_d is None or d < nearest_d:
                            nearest_id = i
                            nearest_d = d

                records.append(
                    {
                        "t": t,
                        "agent_id": ag.agent_id,
                        "lat": ag.lat,
                        "lon": ag.lon,
                        "alt_m": alt,
                        "state": ag.state,
                        "in_bsa": in_bsa,
                        "nearest_turbine_id": nearest_id,
                        "nearest_turbine_d_m": nearest_d,
                    }
                )

            t += pd.Timedelta(seconds=self.sim_cfg.step_seconds)

        return pd.DataFrame.from_records(records)