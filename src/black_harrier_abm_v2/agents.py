from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import random
import math
import pandas as pd

from .movement import State


@dataclass
class StateParams:
    speed_mean: float
    speed_sd: float


DEFAULT_STATE_PARAMS: Dict[str, StateParams] = {
    State.BREEDING: StateParams(8.0, 2.5),
    State.FORAGING: StateParams(6.0, 2.0),
    State.COMMUTING: StateParams(11.0, 3.0),
    State.DISPLAYING: StateParams(7.0, 2.0),
    State.MIGRATING: StateParams(15.0, 4.0),
    State.ROOSTING: StateParams(0.5, 0.2),
}


Season = str  # "breeding", "nonbreeding", "migration_out", "migration_return"


def month_to_season(ts: pd.Timestamp) -> Season:
    m = ts.month
    if m in (12, 1):
        return "migration_out"
    if m in (5, 6, 7):
        return "migration_return"
    if 6 <= m <= 12:
        return "breeding"
    return "nonbreeding"


TRANSITIONS = {
    "breeding": {
        State.BREEDING: {State.BREEDING: 0.05, State.FORAGING: 0.45, State.COMMUTING: 0.35, State.DISPLAYING: 0.10, State.ROOSTING: 0.05},
        State.FORAGING: {State.FORAGING: 0.60, State.COMMUTING: 0.25, State.BREEDING: 0.05, State.ROOSTING: 0.10},
        State.COMMUTING: {State.FORAGING: 0.50, State.COMMUTING: 0.35, State.BREEDING: 0.05, State.ROOSTING: 0.10},
        State.DISPLAYING: {State.BREEDING: 0.40, State.COMMUTING: 0.30, State.FORAGING: 0.20, State.ROOSTING: 0.10},
        State.ROOSTING: {State.COMMUTING: 0.35, State.FORAGING: 0.35, State.BREEDING: 0.10, State.ROOSTING: 0.20},
        State.MIGRATING: {State.COMMUTING: 0.7, State.FORAGING: 0.2, State.ROOSTING: 0.1},
    },
    "nonbreeding": {
        State.ROOSTING: {State.FORAGING: 0.45, State.COMMUTING: 0.25, State.ROOSTING: 0.30},
        State.FORAGING: {State.FORAGING: 0.55, State.COMMUTING: 0.25, State.ROOSTING: 0.20},
        State.COMMUTING: {State.FORAGING: 0.55, State.COMMUTING: 0.25, State.ROOSTING: 0.20},
        State.MIGRATING: {State.COMMUTING: 0.6, State.FORAGING: 0.3, State.ROOSTING: 0.1},
        State.BREEDING: {State.ROOSTING: 1.0},
        State.DISPLAYING: {State.ROOSTING: 1.0},
    },
    "migration_out": {
        State.MIGRATING: {State.MIGRATING: 0.80, State.COMMUTING: 0.15, State.ROOSTING: 0.05},
        State.ROOSTING: {State.MIGRATING: 0.70, State.ROOSTING: 0.20, State.COMMUTING: 0.10},
        State.COMMUTING: {State.MIGRATING: 0.65, State.COMMUTING: 0.20, State.ROOSTING: 0.15},
        State.FORAGING: {State.COMMUTING: 0.40, State.MIGRATING: 0.40, State.ROOSTING: 0.20},
        State.BREEDING: {State.MIGRATING: 1.0},
        State.DISPLAYING: {State.MIGRATING: 1.0},
    },
    "migration_return": {
        State.MIGRATING: {State.MIGRATING: 0.65, State.COMMUTING: 0.20, State.FORAGING: 0.10, State.ROOSTING: 0.05},
        State.ROOSTING: {State.MIGRATING: 0.50, State.ROOSTING: 0.30, State.FORAGING: 0.20},
        State.COMMUTING: {State.MIGRATING: 0.55, State.COMMUTING: 0.25, State.FORAGING: 0.20},
        State.FORAGING: {State.COMMUTING: 0.40, State.MIGRATING: 0.40, State.ROOSTING: 0.20},
        State.BREEDING: {State.BREEDING: 1.0},
        State.DISPLAYING: {State.ROOSTING: 1.0},
    },
}


@dataclass
class HarrierAgent:
    agent_id: int
    lat: float
    lon: float
    state: str
    sex: str = "unknown"
    nest: Optional[tuple[float, float]] = None
    roost: Optional[tuple[float, float]] = None
    state_params: Dict[str, StateParams] = field(default_factory=lambda: DEFAULT_STATE_PARAMS)
    step_seconds: int = 60
    rng: random.Random = field(default_factory=random.Random)

    def sample_speed(self) -> float:
        p = self.state_params[self.state]
        return max(0.0, self.rng.gauss(p.speed_mean, p.speed_sd))

    def next_state(self, season: Season) -> str:
        row = TRANSITIONS.get(season, {}).get(self.state)
        if not row:
            row = {State.FORAGING: 0.5, State.COMMUTING: 0.3, State.ROOSTING: 0.2}
        states, probs = zip(*row.items())
        return self.rng.choices(states, weights=probs, k=1)[0]