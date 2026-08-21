"""Converts vaastav/Fantasy-Premier-League historical CSVs into TrainingExample JSONL files.

Reads merged_gw.csv (per-gameweek stats), players_raw.csv (player attributes),
and fixtures.csv (fixture difficulty) for a given season, then produces
position-specific JSONL training files matching the TrainingExample schema.
"""

import csv
import json
import statistics
from pathlib import Path

from src.models.post_prediction import Position
from src.models.training_examples import TrainingExample

BASE_DIR = Path(__file__).resolve().parents[2]
HISTORICAL_DIR = BASE_DIR / "data/historical"

POSITION_NAMES = {
    Position.GOALKEEPER: "goalkeepers",
    Position.DEFENDER: "defenders",
    Position.MIDFIELDER: "midfielders",
    Position.ATTACKER: "attackers",
}


def _load_csv(path: Path) -> list[dict]:
    for encoding in ("utf-8", "latin-1"):
        try:
            with path.open(newline="", encoding=encoding) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def _safe_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _safe_int(value, default=0):
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _rolling_mean(values: list[float], n: int) -> float | None:
    window = values[-n:]
    return statistics.mean(window) if window else None


def _fixture_difficulty(fixtures: list[dict], n: int) -> float | None:
    """Compute average fixture difficulty from upcoming fixtures.

    Uses the 'difficulty' field from fixtures.csv, adjusted for home/away.
    """
    upcoming = fixtures[:n]
    if not upcoming:
        return None

    adjusted = []
    for fx in upcoming:
        diff = _safe_float(fx.get("difficulty"), 3.0)
        is_home = fx.get("is_home", "").lower() in ("true", "1", "yes")
        adjusted.append(diff - 0.3 if is_home else diff + 0.3)

    return statistics.mean(adjusted)


def _home_ratio(fixtures: list[dict], n: int) -> float | None:
    upcoming = fixtures[:n]
    if not upcoming:
        return None

    return sum(1 for fx in upcoming if fx.get("is_home", "").lower() in ("true", "1", "yes")) / len(upcoming)


def _build_player_gw_rows(merged_gw: list[dict]) -> dict[int, list[dict]]:
    """Group merged_gw rows by player element id, sorted by gameweek."""
    players: dict[int, list[dict]] = {}

    for row in merged_gw:
        element = _safe_int(row.get("element"), -1)
        if element < 0:
            continue

        players.setdefault(element, []).append(row)

    for element in players:
        players[element].sort(key=lambda r: _safe_int(r.get("round") or r.get("GW"), 0))

    return players


def _build_player_attributes(players_raw: list[dict]) -> dict[int, dict]:
    """Index players_raw by element id."""
    return {_safe_int(p.get("id"), -1): p for p in players_raw if _safe_int(p.get("id"), -1) >= 0}


def _build_fixtures_by_team(fixtures_csv: list[dict]) -> dict[int, list[dict]]:
    """Group fixtures by team, sorted by gameweek.

    Each fixture is normalised to the perspective of the team:
    - is_home: whether the team is at home
    - difficulty: the team's difficulty rating for that fixture
    """
    by_team: dict[int, list[dict]] = {}

    for fx in fixtures_csv:
        team_h = _safe_int(fx.get("team_h"), -1)
        team_a = _safe_int(fx.get("team_a"), -1)
        gw = _safe_int(fx.get("event") or fx.get("round"), 0)
        diff_h = _safe_float(fx.get("team_h_difficulty"), 3.0)
        diff_a = _safe_float(fx.get("team_a_difficulty"), 3.0)

        if team_h >= 0:
            by_team.setdefault(team_h, []).append(
                {
                    "event": gw,
                    "is_home": "true",
                    "difficulty": diff_h,
                }
            )
        if team_a >= 0:
            by_team.setdefault(team_a, []).append(
                {
                    "event": gw,
                    "is_home": "false",
                    "difficulty": diff_a,
                }
            )

    for team in by_team:
        by_team[team].sort(key=lambda f: f["event"])

    return by_team


def _convert_season(season_dir: Path) -> dict[Position, list[dict]]:
    """Convert a single season's CSVs into TrainingExample dicts grouped by position."""
    merged_gw_path = season_dir / "merged_gw.csv"
    players_raw_path = season_dir / "players_raw.csv"
    fixtures_path = season_dir / "fixtures.csv"

    if not merged_gw_path.exists() or not players_raw_path.exists():
        print(f"  Skipping {season_dir.name}: missing required files")
        return {}

    merged_gw = _load_csv(merged_gw_path)
    players_raw = _load_csv(players_raw_path)
    fixtures_csv = _load_csv(fixtures_path) if fixtures_path.exists() else []

    player_gw_rows = _build_player_gw_rows(merged_gw)
    player_attrs = _build_player_attributes(players_raw)
    fixtures_by_team = _build_fixtures_by_team(fixtures_csv)

    examples: dict[Position, list[dict]] = {
        Position.GOALKEEPER: [],
        Position.DEFENDER: [],
        Position.MIDFIELDER: [],
        Position.ATTACKER: [],
    }

    for element, gw_rows in player_gw_rows.items():
        attrs = player_attrs.get(element)
        if attrs is None:
            continue

        element_type = _safe_int(attrs.get("element_type"), -1)
        if element_type < 1 or element_type > 4:
            continue

        position = Position(element_type)
        team = _safe_int(attrs.get("team"), -1)
        team_fixtures = fixtures_by_team.get(team, [])

        for gw_idx in range(len(gw_rows) - 1):
            available = gw_rows[: gw_idx + 1]
            target = gw_rows[gw_idx + 1]

            features = _compute_features(element, position, attrs, available, team_fixtures, gw_idx + 1)

            examples[position].append(
                {
                    "player_id": element,
                    "position": position,
                    "gameweek": gw_idx + 1,
                    "features": features,
                    "target_points": _safe_int(target.get("total_points")),
                }
            )

    return examples


def _compute_features(
    player_id: int,
    position: Position,
    attrs: dict,
    history: list[dict],
    team_fixtures: list[dict],
    current_gw: int,
) -> dict:
    """Compute the same feature set as PlayerFeatureTransformer, from CSV data."""
    # Rolling stats from gameweek history
    total_points = [_safe_float(r.get("total_points")) for r in history]
    minutes = [_safe_float(r.get("minutes")) for r in history]
    yellow_cards = [_safe_float(r.get("yellow_cards")) for r in history]
    red_cards = [_safe_float(r.get("red_cards")) for r in history]
    transfers_balance = [_safe_float(r.get("transfers_balance")) for r in history]

    # Upcoming fixtures (fixtures after current gameweek)
    upcoming = [f for f in team_fixtures if _safe_int(f.get("event"), 0) > current_gw]

    base = {
        "player_id": player_id,
        "playing_probability": _playing_probability(attrs),
        "next_fixture_difficulty": _fixture_difficulty(upcoming, 1),
        "avg_fixture_difficulty_3": _fixture_difficulty(upcoming, 3),
        "avg_fixture_difficulty_5": _fixture_difficulty(upcoming, 5),
        "home_fixture_ratio_next_5": _home_ratio(upcoming, 5),
        "avg_points_last_3": _rolling_mean(total_points, 3),
        "avg_points_last_5": _rolling_mean(total_points, 5),
        "avg_minutes_last_3": _rolling_mean(minutes, 3),
        "avg_minutes_last_5": _rolling_mean(minutes, 5),
        "yellow_cards_last_5": _rolling_mean(yellow_cards, 5),
        "red_cards_last_5": _rolling_mean(red_cards, 5),
        "selected_by_percent": _safe_float(attrs.get("selected_by_percent")),
        "transfers_balance_last_5": _rolling_mean(transfers_balance, 5),
        "now_cost": _safe_float(attrs.get("now_cost")),
        "avg_price_diff_historic": None,
        "avg_points_per_90_historic": None,
        "avg_minutes_per_season_historic": None,
    }

    # Position-specific features
    if position == Position.GOALKEEPER:
        base.update(_goalkeeper_features(attrs, history))
    elif position == Position.DEFENDER:
        base.update(_defender_features(attrs, history))
    elif position == Position.MIDFIELDER:
        base.update(_midfielder_features(attrs, history))
    elif position == Position.ATTACKER:
        base.update(_attacker_features(attrs, history))

    return base


def _playing_probability(attrs: dict) -> float:
    current = _safe_float(attrs.get("chance_of_playing_this_round"), 100.0)
    nxt = _safe_float(attrs.get("chance_of_playing_next_round"), current)
    if current == 0.0 and nxt == 0.0:
        current = 100.0
        nxt = 100.0
    return (0.65 * current + 0.35 * nxt) / 100


def _goalkeeper_features(attrs: dict, history: list[dict]) -> dict:
    saves_last_3 = None
    saves_last_5 = None
    for n in (3, 5):
        window = history[-n:]
        total_min = sum(_safe_float(r.get("minutes")) for r in window)
        if total_min > 0:
            total_saves = sum(_safe_float(r.get("saves")) for r in window)
            val = total_saves / (total_min / 90)
            if n == 3:
                saves_last_3 = val
            else:
                saves_last_5 = val

    return {
        "saves_per_90": _safe_float(attrs.get("saves_per_90")),
        "clean_sheets_per_90": _safe_float(attrs.get("clean_sheets_per_90")),
        "saves_per_90_last_3": saves_last_3,
        "saves_per_90_last_5": saves_last_5,
        "goals_conceded_per_90": _safe_float(attrs.get("goals_conceded_per_90")),
        "penalties_saved": _safe_int(attrs.get("penalties_saved")),
    }


def _defender_features(attrs: dict, history: list[dict]) -> dict:
    clean_sheets = [_safe_float(r.get("clean_sheets")) for r in history]
    xgi = [_safe_float(r.get("expected_goal_involvements")) for r in history]

    return {
        "clean_sheets_per_90": _safe_float(attrs.get("clean_sheets_per_90")),
        "expected_goals_conceded_per_90": _safe_float(attrs.get("expected_goals_conceded_per_90")),
        "defensive_contribution_per_90": _safe_float(attrs.get("defensive_contribution_per_90"), 0.0),
        "clean_sheet_rate_last_5": _rolling_mean(clean_sheets, 5),
        "expected_goal_involvements_per_90": _safe_float(attrs.get("expected_goal_involvements_per_90")),
        "avg_xgi_last_3": _rolling_mean(xgi, 3),
        "avg_xgi_last_5": _rolling_mean(xgi, 5),
    }


def _midfielder_features(attrs: dict, history: list[dict]) -> dict:
    xg = [_safe_float(r.get("expected_goals")) for r in history]
    xa = [_safe_float(r.get("expected_assists")) for r in history]

    return {
        "expected_goal_involvements_per_90": _safe_float(attrs.get("expected_goal_involvements_per_90")),
        "avg_xg_last_3": _rolling_mean(xg, 3),
        "avg_xg_last_5": _rolling_mean(xg, 5),
        "avg_xa_last_3": _rolling_mean(xa, 3),
        "avg_xa_last_5": _rolling_mean(xa, 5),
        "avg_set_piece_order": _set_piece_score(attrs),
        "clean_sheets_per_90": _safe_float(attrs.get("clean_sheets_per_90")),
    }


def _attacker_features(attrs: dict, history: list[dict]) -> dict:
    xg = [_safe_float(r.get("expected_goals")) for r in history]
    xa = [_safe_float(r.get("expected_assists")) for r in history]

    total_minutes = _safe_float(attrs.get("minutes"))
    goals = _safe_float(attrs.get("goals_scored"))
    assists = _safe_float(attrs.get("assists"))

    goals_per_90 = goals / (total_minutes / 90) if total_minutes > 0 else None
    assists_per_90 = assists / (total_minutes / 90) if total_minutes > 0 else None

    return {
        "expected_goal_involvements_per_90": _safe_float(attrs.get("expected_goal_involvements_per_90")),
        "avg_xg_last_3": _rolling_mean(xg, 3),
        "avg_xg_last_5": _rolling_mean(xg, 5),
        "avg_xa_last_3": _rolling_mean(xa, 3),
        "avg_xa_last_5": _rolling_mean(xa, 5),
        "goals_per_90": goals_per_90,
        "assists_per_90": assists_per_90,
        "avg_set_piece_order": _set_piece_score(attrs),
    }


def _set_piece_score(attrs: dict) -> float | None:
    scores = []
    for key in (
        "corners_and_indirect_freekicks_order",
        "direct_freekicks_order",
        "penalties_order",
    ):
        val = attrs.get(key)
        if val is not None and val != "":
            rank = _safe_int(val, -1)
            if rank > 0:
                scores.append(max(0, 10 - rank))

    return statistics.mean(scores) if scores else None


def convert_all_historical(store=None) -> None:
    """Convert all seasons in data/historical/ and store in MongoDB."""
    if not HISTORICAL_DIR.exists():
        print("No historical data directory found")
        return

    if store is None:
        from src.storage.mongo_client import MongoStore

        store = MongoStore()

    seasons = sorted(d.name for d in HISTORICAL_DIR.iterdir() if d.is_dir())

    for season in seasons:
        season_dir = HISTORICAL_DIR / season
        print(f"\nConverting {season}...")

        # Convert season name (e.g. "2024-25" -> "2024_25")
        season_folder = season.replace("-", "_")

        examples_by_position = _convert_season(season_dir)

        total = 0
        for position, examples in examples_by_position.items():
            # Build full training example dicts for MongoDB
            records = []
            for example in examples:
                training_example = TrainingExample(**example)
                records.append(json.loads(training_example.model_dump_json()))
            count = store.upsert_training_examples(position, season_folder, records)
            total += len(examples)
            print(f"  {POSITION_NAMES[position]}: {len(examples)} examples (upserted: {count})")

        print(f"  Total: {total} examples stored for {season_folder}")


if __name__ == "__main__":
    convert_all_historical()
