# Automated FPL Analyser

This project fetches player data from the Fantasy Premier League (FPL) API, models expected points (xP) using Gaussian Process regression, and suggests optimal transfers for your FPL team.

## How it works

1. **Data ingestion** — Fetches current player stats and fixtures from the FPL API
2. **Training** — Trains per-position Gaussian Process models (gpytorch) on historical data (10 seasons, ~245K examples)
3. **Prediction** — Generates xP predictions for all current players
4. **Optimisation** — Fetches your squad and suggests transfers to maximise xP, respecting FPL constraints (budget, position limits, club limits, transfer hits)

## Prerequisites

- Python 3.14
- [uv](https://astral.sh/uv/)
- [pre-commit](https://pre-commit.com/#install) (for development)

## Setup

```bash
uv sync
uv pip install -e .
pre-commit install  # optional, for development
```

## Usage

### Weekly automation (GitHub Actions)

The repository includes a GitHub Actions workflow (`.github/workflows/run_automation.yaml`) that runs the full pipeline every Friday at 12:00 UTC and emails you the results.

To use it:

1. Fork this repository
2. Add the following secrets in **Settings → Secrets and variables → Actions**:
   - `FPL_USER_ID` — Your FPL user ID (find it in your team page URL, e.g. `fantasy.premierleague.com/entry/3846224`)
   - `RESEND_API_KEY` — API key from [Resend](https://resend.com) for email delivery
   - `EMAIL_ADDRESS` — The email address to send results to
3. The workflow will run automatically on schedule, or you can trigger it manually via **Actions → Run workflow**

### Local usage

**Optimise your team** (uses existing predictions):

```bash
uv run python bin/optimise.py <your_fpl_user_id>
```

**Retrain the model and optimise:**

```bash
uv run python bin/optimise.py <your_fpl_user_id> --train
```

**Run the full weekly pipeline locally:**

```bash
export FPL_USER_ID=your_fpl_user_id
uv run python bin/run.py
```

### Historical data

Training data from 10 seasons (2016-17 through 2025-26) is stored as JSONL files in `data/training/`. Historical seasons (2016-17 through 2024-25) were converted from the [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) repository using `historical_converter.py`. Current-season data is fetched and converted automatically by `load_player_data()`.

## Project structure

```
src/
├── api_client/          # FPL API client
├── models/              # Pydantic data models
│   ├── pre_processing.py    # API response models
│   ├── post_prediction.py   # Prediction output models
│   ├── player_features.py   # Position-specific feature models
│   └── squad.py             # Squad and transfer suggestion models
└── optimisation/
    ├── predictor.py             # Main pipeline orchestrator
    ├── gp_model.py              # Gaussian Process xP model (gpytorch)
    ├── historical_converter.py # Converts vaastav CSVs to training data
    ├── team_optimiser.py        # Constrained transfer optimisation
    └── player_feature_transformer.py  # RawPlayer → feature vectors
bin/
├── run.py              # Weekly automation entry point
├── optimise.py         # CLI tool for team optimisation
└── email_findings.py   # Email results via Resend API
data/
├── training/           # Training JSONL per season (10 seasons)
├── player_features/    # Current player feature vectors
└── predictions/        # xP predictions per position
```
