from enum import Enum

from pydantic import BaseModel


class Position(Enum):
    GOALKEEPER = 1
    DEFENDER = 2
    MIDFIELDER = 3
    ATTACKER = 4


class PostPlayer(BaseModel):
    """Describes post-prediction player model, for de/serialisation"""

    player_id: int
    position: Position
    team: int
    xp_series: list[float]
    current_price: float
