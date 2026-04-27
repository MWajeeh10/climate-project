# Lahore Smog Intelligence Studio

This repository contains a Lahore-focused PM2.5 prediction pipeline and a local demo UI for presenting the project to technical and non-technical audiences.

## What Is Here

- `fetch_data.py`: builds the merged training dataset from weather, PM2.5, fire, and traffic-related sources.
- `merge_pm25.py`: merges multiple PM2.5 sources across time ranges.
- `scrape_epd.py`: scrapes Punjab EPD AQI PDFs and extracts Lahore data.
- `train_and_evaluate.py`: trains Random Forest, XGBoost, LightGBM, and ensemble models and saves metrics/figures.
- `train_observed_daily_models.py`: trains a cleaner daily model suite from `lahore_air_quality_final_dataset.csv`.
- `demo_app.py`: launches the polished local demo UI with live predictions.
- `ui/`: frontend for the demo experience.
- `models/`: saved trained models, feature lists, and benchmark tables.
- `models_observed_daily/`: saved daily models trained on the newer observed dataset.
- `figures/`: saved evaluation and explainability plots.

## Demo UI

The current demo UI is designed for live presentation.

Features:

- scenario presets for high-smog and low-smog conditions
- weighted ensemble live prediction using the saved model artifacts
- selectable observed daily models for behavior comparison
- plain-language explanation of the result
- public-health style interpretation for non-technical viewers
- glossary of scientific terms
- saved benchmark cards and research figure gallery

## How To Run

Use Python 3.12+.

Install dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

Run the local demo:

```bash
python demo_app.py
```

Then open:

```text
http://127.0.0.1:8080
```

## Current Live Inference Logic

The demo headline uses the saved weighted ensemble because `models/evaluation_results.csv` shows it achieved the strongest held-out benchmark among the saved artifacts.

The app also supports a cleaner observed daily suite trained from `lahore_air_quality_final_dataset.csv`, which is useful when you want to compare model behavior on a more realistic seasonal target.

The live view also shows:

- Random Forest
- XGBoost
- LightGBM
- Weighted Average
- Stacking Ensemble

## Suggested Next Research Improvements

- replace any fallback/generated PM2.5 with fully observed end-to-end target data
- evaluate explicit forecasting horizons such as `t+1`, `t+6`, and `t+24`
- add ablation studies for weather-only vs inversion-aware vs fire-aware variants
- report uncertainty and calibration more formally
- validate on the latest smog season and, if possible, a second city

## Demo Tips

- start with `Winter Inversion Dawn` to show a strong smog case
- move wind speed or boundary layer height to show why the prediction changes
- use the figure gallery to explain accuracy, feature importance, and explainability
