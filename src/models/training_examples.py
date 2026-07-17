from typing import Union

from pydantic import BaseModel

from src.models.player_features import (
    AttackerFeatures,
    DefenderFeatures,
    GoalkeeperFeatures,
    MidfielderFeatures,
)
from src.models.post_prediction import Position

FeatureModel = Union[
    GoalkeeperFeatures,
    DefenderFeatures,
    MidfielderFeatures,
    AttackerFeatures,
]


class TrainingExample(BaseModel):
    """
    A single supervised learning example.

    Represents:

    "Given everything known before this gameweek,
     predict the next gameweek points."
    """

    player_id: int
    position: Position
    gameweek: int
    features: FeatureModel
    target_points: int
