from pydantic import BaseModel


class SquadPick(BaseModel):
    """A single player pick in a user's squad."""

    element: int
    position: int
    selling_price: float
    purchase_price: float
    is_captain: bool = False
    is_vice_captain: bool = False
    multiplier: int = 1


class Squad(BaseModel):
    """A user's current squad."""

    picks: list[SquadPick]
    bank: float = 0.0
    value: float = 0.0
    free_transfers: int = 1
    # FPL squad constraints
    SQUAD_SIZE: int = 15
    MAX_GOALKEEPERS: int = 2
    MAX_DEFENDERS: int = 5
    MAX_MIDFIELDERS: int = 5
    MAX_ATTACKERS: int = 3
    MAX_PER_CLUB: int = 3
    BUDGET: float = 100.0


class TransferSuggestion(BaseModel):
    """A suggested transfer."""

    player_out: int
    player_out_name: str = ""
    player_out_price: float = 0.0
    player_in: int
    player_in_name: str = ""
    player_in_price: float = 0.0
    xP_gain: float
    cost_change: float
    net_xp_improvement: float


class OptimisationResult(BaseModel):
    """Result of team optimisation."""

    current_squad: Squad
    suggestions: list[TransferSuggestion]
    current_squad_xp: float
    optimised_squad_xp: float
    transfers_used: int
    point_hit: int
