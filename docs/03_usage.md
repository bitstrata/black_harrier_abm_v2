# Usage

## 1) Install & activate venv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Prepare config
Create a YAML config (see `docs/example_config.yaml` below) pointing to rasters and agent setup.

## 3) Run
```bash
python main.py --config docs/example_config.yaml --out data/processed/tracks_demo.csv
```

## 4) Visualise
Load the CSV in your analysis stack to build heatmaps and BSA time summaries.
```