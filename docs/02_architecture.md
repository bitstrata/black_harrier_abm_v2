# Architecture

- `agents.py` — `HarrierAgent` (state machine, speed sampling)
- `movement.py` — `StepSelectionPolicy` (step-selection), `AltitudeModel`
- `environment.py` — `Environment`, `Turbine` (samplers as callables)
- `samplers/raster.py` — `RasterSamplers` (raster-backed habitat, wind, slope)
- `risk.py` — `ABMSimulation` (orchestrates and computes risk proxies)
- `utils/geo.py` — geodesic helpers (haversine, destination)
- `config.py` — dataclasses + YAML loader

**Extensibility**
- Plug Bayesian updates for `AltitudeModel` & transition matrices.
- Add mitigation toggles: curtailment windows, blade painting coefficients, hub-height filters.