from typing import Optional

from pydantic import BaseModel


class PrePlayer(BaseModel):
    avg_fixture_score: int  # metric of avg fixture difficulty + h/a, next 5
    playing_probability: int  # 0.65 * chance playing this round + 0.35 * change playing the next
    points_per_game: str
    minutes: int
    selected_rank: int
    form_rank: int
    yellow_cards: int
    red_cards: int
    bps: int
    now_cost: float
    avg_price_diff_historic: float  # avg delta price in previous seasons
    avg_points_per_90_historic: Optional[float]  # avg ppg previous seasons


class Goalkeeper(PrePlayer):
    saves_per_90: float
    goals_conceeded_per_90: float
    clean_sheets_per_90: float
    penalties_saved: int


class Defender(PrePlayer):
    expected_goals_conceeded_per_90: float
    clean_sheets_per_90: float
    ict_index_rank: int
    expected_goal_involvements_per_90: float
    defensive_contribution_per_90: float


class Midfielder(PrePlayer):
    clean_sheets_per_90: float
    ict_index_rank: int
    expected_goal_involvements_per_90: float
    defensive_contribution_per_90: float
    avg_set_piece_order: int  # Composite metric to show avg set piece order, else 0


class Attacker(PrePlayer):
    ict_index_rank: int
    expected_goal_involvements_per_90: float
    avg_set_piece_order: int  # Composite metric to show avg set piece order, else 0
    goals_per_90: float  # need to compile
    assists_per_90: float  # need to compile
