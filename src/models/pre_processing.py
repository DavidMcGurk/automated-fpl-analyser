from __future__ import annotations

from typing import List, Optional

from src.models.post_prediction import Position
from pydantic import BaseModel, Field


class Fixture(BaseModel):
    is_home: bool
    difficulty: int


class History(BaseModel):
    total_points: int
    was_home: bool
    minutes: int

    goals_scored: int
    assists: int

    clean_sheets: int
    goals_conceded: int
    own_goals: int

    penalties_saved: int
    penalties_missed: int

    yellow_cards: int
    red_cards: int

    saves: int
    bonus: int
    bps: int
    influence: str
    creativity: str
    threat: str
    ict_index: str
    clearances_blocks_interceptions: int
    recoveries: int
    tackles: int
    defensive_contribution: int
    starts: int
    expected_goals: str
    expected_assists: str
    expected_goal_involvements: str
    expected_goals_conceded: str
    value: int
    transfers_balance: int
    selected: int
    transfers_in: int
    transfers_out: int


class SeasonHistory(BaseModel):
    season_name: str
    element_code: int
    start_cost: int
    end_cost: int
    total_points: int
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    own_goals: int
    penalties_saved: int
    penalties_missed: int
    yellow_cards: int
    red_cards: int
    saves: int
    bonus: int
    bps: int
    influence: str
    creativity: str
    threat: str
    ict_index: str
    clearances_blocks_interceptions: int
    recoveries: int
    tackles: int
    defensive_contribution: int
    starts: int
    expected_goals: str
    expected_assists: str
    expected_goal_involvements: str
    expected_goals_conceded: str


class PlayerAttributes(BaseModel):
    can_transact: bool
    can_select: bool

    chance_of_playing_next_round: Optional[int]
    chance_of_playing_this_round: Optional[int]

    code: int
    cost_change_event: int
    cost_change_event_fall: int
    cost_change_start: int
    cost_change_start_fall: int

    price_change_percent: str
    dreamteam_count: int

    ep_next: Optional[str]
    ep_this: str
    event_points: int

    first_name: str
    second_name: str
    web_name: str
    known_name: str

    form: str
    in_dreamteam: bool

    now_cost: float
    points_per_game: str

    removed: bool
    selected_by_percent: str
    special: bool

    team: int

    total_points: int

    transfers_in: int
    transfers_in_event: int
    transfers_out: int
    transfers_out_event: int

    value_form: str
    value_season: str

    # --- Performance stats ---
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    own_goals: int

    penalties_saved: int
    penalties_missed: int

    yellow_cards: int
    red_cards: int

    saves: int
    bonus: int
    bps: int

    influence: str
    creativity: str
    threat: str
    ict_index: str

    clearances_blocks_interceptions: int
    recoveries: int
    tackles: int
    defensive_contribution: int
    starts: int

    expected_goals: str
    expected_assists: str
    expected_goal_involvements: str
    expected_goals_conceded: str

    # --- Set pieces ---
    corners_and_indirect_freekicks_order: Optional[int]
    direct_freekicks_order: Optional[int]
    penalties_order: Optional[int]

    # --- Rankings ---
    influence_rank: int
    influence_rank_type: int
    creativity_rank: int
    creativity_rank_type: int
    threat_rank: int
    threat_rank_type: int
    ict_index_rank: int
    ict_index_rank_type: int

    # --- Per 90 stats ---
    expected_goals_per_90: float
    saves_per_90: float
    expected_assists_per_90: float
    expected_goal_involvements_per_90: float
    expected_goals_conceded_per_90: float
    goals_conceded_per_90: float

    # --- Value / selection ranks ---
    now_cost_rank: int
    now_cost_rank_type: int
    form_rank: int
    form_rank_type: int
    points_per_game_rank: int
    points_per_game_rank_type: int
    selected_rank: int
    selected_rank_type: int

    # --- Derived per 90 ---
    starts_per_90: float
    clean_sheets_per_90: float
    defensive_contribution_per_90: float


class RawPlayer(BaseModel):
    """Describes pre-prediction player model, to be instantiated from API call inputs"""

    player_id: int
    position: Position
    attributes: PlayerAttributes
    fixtures: List[Fixture]
    this_season_history: List[History] = Field(alias="history")
    previous_seasons_history: List[SeasonHistory] = Field(alias="history_past")
