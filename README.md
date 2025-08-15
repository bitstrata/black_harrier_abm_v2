# black_harrier_abm_v2

Agent-based model (ABM) for **Black Harrier (Circus maurus)** flight prediction and wind-energy risk assessment. Modular OOP design with season-aware state transitions, step-selection movement, and turbine collision risk proxies.

## Quickstart

```bash
# From the project root
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run a demo
python main.py --config docs/example_config.yaml --out data/processed/tracks_demo.csv

# Run tests
pytest -q
```

## Layout
- `src/black_harrier_abm_v2`: core package (environment, agents, movement, risk, utils)
- `data`: input rasters (DEM, habitat, wind), turbines, and outputs
- `docs`: overview, architecture, and usage
- `tests`: pytest-based unit tests

## Features
- Season/state machine (breeding, commuting, foraging, displaying, migrating, roosting)
- Markov transitions by season (priors; swap for telemetry-derived matrices)
- Step-selection movement (habitat, wind, target pull, slope, turbine avoidance)
- Collision-risk proxy (time in blade-swept area near turbines)
- Ready for Bayesian parameter updates

## License
MIT (adjust as needed)