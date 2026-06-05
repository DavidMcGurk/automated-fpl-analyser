from typing import Optional
from pydantic import BaseModel


class BaseFeatures(BaseModel):
    player_id: int

    # Availability
    playing_probability: float

    # Fixture context
    next_fixture_difficulty: Optional[float]
    avg_fixture_difficulty_3: Optional[float]
    avg_fixture_difficulty_5: Optional[float]
    home_fixture_ratio_next_5: Optional[float]

    # Recent form
    avg_points_last_3: Optional[float]
    avg_points_last_5: Optional[float]

    avg_minutes_last_3: Optional[float]
    avg_minutes_last_5: Optional[float]

    # Discipline
    yellow_cards_last_5: Optional[float]
    red_cards_last_5: Optional[float]

    # Market signal
    selected_by_percent: float
    transfers_balance_last_5: Optional[float]

    # Cost
    now_cost: float

    # Historical baseline
    avg_price_diff_historic: Optional[float]
    avg_points_per_90_historic: Optional[float]
    avg_minutes_per_season_historic: Optional[float]


# -------------------------
# Goalkeepers
# -------------------------


class GoalkeeperFeatures(BaseFeatures):
    saves_per_90: float
    clean_sheets_per_90: float

    saves_per_90_last_3: Optional[float]
    saves_per_90_last_5: Optional[float]

    goals_conceded_per_90: float
    penalties_saved: int


# -------------------------
# Defenders
# -------------------------


class DefenderFeatures(BaseFeatures):
    clean_sheets_per_90: float

    expected_goals_conceded_per_90: float
    defensive_contribution_per_90: float

    clean_sheet_rate_last_5: Optional[float]

    expected_goal_involvements_per_90: float

    avg_xgi_last_3: Optional[float]
    avg_xgi_last_5: Optional[float]


# -------------------------
# Midfielders
# -------------------------


class MidfielderFeatures(BaseFeatures):
    expected_goal_involvements_per_90: float

    avg_xg_last_3: Optional[float]
    avg_xg_last_5: Optional[float]

    avg_xa_last_3: Optional[float]
    avg_xa_last_5: Optional[float]

    avg_set_piece_order: Optional[float]

    clean_sheets_per_90: float


# -------------------------
# Attackers
# -------------------------


class AttackerFeatures(BaseFeatures):
    expected_goal_involvements_per_90: float

    avg_xg_last_3: Optional[float]
    avg_xg_last_5: Optional[float]

    avg_xa_last_3: Optional[float]
    avg_xa_last_5: Optional[float]

    goals_per_90: Optional[float]
    assists_per_90: Optional[float]

    avg_set_piece_order: Optional[float]
