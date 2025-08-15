from black_harrier_abm_v2.config import SimulationConfig
from black_harrier_abm_v2.environment import Environment, Turbine
from black_harrier_abm_v2.agents import HarrierAgent
from black_harrier_abm_v2.movement import StepSelectionPolicy, AltitudeModel, Weights, AltitudePriors as AltP
from black_harrier_abm_v2.risk import ABMSimulation
import pandas as pd


def test_simulation_runs():
    sim_cfg = SimulationConfig(
        start_time=pd.Timestamp("2020-09-01 06:00:00"),
        end_time=pd.Timestamp("2020-09-01 06:10:00"),
        step_seconds=60,
    )
    env = Environment(
        habitat_sampler=lambda lat, lon, t: 0.7,
        wind_sampler=lambda lat, lon, t: (0.0, 0.0),
        slope_sampler=lambda lat, lon: 0.0,
        turbines=[Turbine(lat=-33.21, lon=18.11)],
    )
    agents = [HarrierAgent(agent_id=1, lat=-33.195, lon=18.096, state="breeding")]
    policy = StepSelectionPolicy(weights=Weights())
    alt_model = AltitudeModel(priors=AltP(
        means={"breeding": 20.0}, sds={"breeding": 5.0}
    ))

    sim = ABMSimulation(env=env, agents=agents, policy=policy, alt_model=alt_model, sim_cfg=sim_cfg)
    df = sim.run()
    assert not df.empty
    assert set(["t", "agent_id", "lat", "lon", "alt_m", "state", "in_bsa"]).issubset(df.columns)