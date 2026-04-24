from enum import StrEnum

from pydantic import BaseModel


class Position(StrEnum):
    gk = "goalkeeper"
    dfr = "defender"
    mid = "midfielder"
    fwd = "forward"


class Player(BaseModel):
    """Describes post-prediction player model, for de/serialisation"""

    player_id: int
    position: Position
    team: int
    xp_series: list[int]
    current_price: float
