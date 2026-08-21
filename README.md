# Automated FPL Analyser

This project fetches player data from the Fantasy Premier League (FPL) API, models expected points (xP) using Gaussian Process regression, and suggests optimal transfers for your FPL team.

## How it works

1. **Data ingestion** — Fetches current player stats and fixtures from the FPL API, stores in MongoDB
2. **Training** — Trains per-position Gaussian Process models (gpytorch) on historical data (10 seasons, ~245K examples)
3. **Prediction** — Generates xP predictions for all current players
4. **Optimisation** — Fetches your squad and suggests transfers to maximise xP, respecting FPL constraints (budget, position limits, club limits, transfer hits)

## Prerequisites

- Python 3.14
- [uv](https://astral.sh/uv/)
- [pre-commit](https://pre-commit.com/#install) (for development)
- A MongoDB instance (e.g. [MongoDB Atlas](https://www.mongodb.com/atlas) free tier)

## Setup

```bash
uv sync
uv pip install -e .
pre-commit install  # optional, for development
```

Set the `MONGODB_URI` environment variable to your MongoDB connection string:

```bash
export MONGODB_URI="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/"
```

### One-time data migration

If you have existing JSONL training data (from `data/training/`), migrate it to MongoDB:

```bash
uv run python bin/migrate.py
```

## Usage

### Weekly automation (GitHub Actions)

The repository includes a GitHub Actions workflow (`.github/workflows/run_automation.yaml`) that runs the full pipeline every Friday at 12:00 UTC and emails you the results.

The scheduled run uses the repository owner's `FPL_USER_ID` and `EMAIL_ADDRESS` secrets automatically. Other users can trigger a manual run with their own FPL user ID and email — no fork or secrets required:

1. Go to **Actions → FPL Weekly Automation → Run workflow**
2. Enter your FPL user ID (find it in your team page URL, e.g. `fantasy.premierleague.com/entry/3846224`)
3. Enter your email address to receive results
4. Click **Run workflow**

The pipeline connects to a shared MongoDB database (configured via the `MONGODB_URI` secret), so all users share the same training data and predictions. Only the team optimisation step is personalised to the user ID you provide.

**Repository secrets** (set once by the repo owner):
- `FPL_USER_ID` — Default FPL user ID for scheduled runs
- `MONGODB_URI` — MongoDB Atlas connection string
- `RESEND_API_KEY` — API key from [Resend](https://resend.com) for email delivery
- `EMAIL_ADDRESS` — Default email address for scheduled run results

### Local usage

**Optimise your team** (uses existing predictions):

```bash
export MONGODB_URI="your_connection_string"
uv run python bin/optimise.py <your_fpl_user_id>
```

**Retrain the model and optimise:**

```bash
uv run python bin/optimise.py <your_fpl_user_id> --train
```

**Run the full weekly pipeline locally:**

```bash
export FPL_USER_ID=your_fpl_user_id
export MONGODB_URI=your_connection_string
uv run python bin/run.py
```

**Validate GP model hyperparameters:**

```bash
uv run python bin/validate.py --model svgp --kernel matern32
```

### Historical data

Training data from 10 seasons (2016-17 through 2025-26) is stored in MongoDB. Historical seasons (2016-17 through 2024-25) were converted from the [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) repository using `historical_converter.py`. Current-season data is fetched and converted automatically by `load_player_data()` on each weekly run, with new gameweeks added to MongoDB as the season progresses.

## Project structure

```
src/
├── api_client/          # FPL API client
├── models/              # Pydantic data models
│   ├── pre_processing.py    # API response models
│   ├── post_prediction.py   # Prediction output models
│   ├── player_features.py   # Position-specific feature models
│   └── squad.py             # Squad and transfer suggestion models
├── storage/
│   └── mongo_client.py      # MongoDB storage layer
└── optimisation/
    ├── predictor.py             # Main pipeline orchestrator
    ├── gp_model.py              # Gaussian Process xP model (gpytorch)
    ├── historical_converter.py # Converts vaastav CSVs to training data
    ├── team_optimiser.py        # Constrained transfer optimisation
    └── player_feature_transformer.py  # RawPlayer → feature vectors
bin/
├── run.py              # Weekly automation entry point
├── optimise.py         # CLI tool for team optimisation
├── validate.py         # GP model validation experiments
├── migrate.py          # One-time JSONL → MongoDB migration
└── email_findings.py   # Email results via Resend API
```
