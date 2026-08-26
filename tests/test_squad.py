"""Tests for squad and transfer suggestion models."""

from src.models.squad import SquadPick, Squad, TransferSuggestion, OptimisationResult


class TestSquadPick:
    def test_basic_pick(self):
        pick = SquadPick(element=1, position=1, selling_price=5.0, purchase_price=4.5)
        assert pick.element == 1
        assert pick.selling_price == 5.0
        assert pick.is_captain is False
        assert pick.multiplier == 1

    def test_captain_pick(self):
        pick = SquadPick(
            element=10,
            position=3,
            selling_price=8.0,
            purchase_price=7.5,
            is_captain=True,
            multiplier=2,
        )
        assert pick.is_captain is True
        assert pick.multiplier == 2


class TestSquad:
    def test_empty_squad(self):
        squad = Squad(picks=[])
        assert squad.picks == []
        assert squad.bank == 0.0
        assert squad.value == 0.0
        assert squad.free_transfers == 1

    def test_squad_constraints(self):
        squad = Squad(picks=[])
        assert squad.SQUAD_SIZE == 15
        assert squad.MAX_GOALKEEPERS == 2
        assert squad.MAX_DEFENDERS == 5
        assert squad.MAX_MIDFIELDERS == 5
        assert squad.MAX_ATTACKERS == 3
        assert squad.MAX_PER_CLUB == 3
        assert squad.BUDGET == 100.0


class TestTransferSuggestion:
    def test_basic_suggestion(self):
        s = TransferSuggestion(
            player_out=1,
            player_in=2,
            xP_gain=2.5,
            cost_change=-0.5,
            net_xp_improvement=2.5,
        )
        assert s.player_out == 1
        assert s.player_in == 2
        assert s.xP_gain == 2.5
        assert s.player_out_name == ""
        assert s.player_in_name == ""
        assert s.player_out_price == 0.0
        assert s.player_in_price == 0.0

    def test_suggestion_with_prices(self):
        s = TransferSuggestion(
            player_out=1,
            player_in=2,
            xP_gain=2.5,
            cost_change=-0.5,
            net_xp_improvement=2.5,
            player_out_price=5.5,
            player_in_price=5.0,
        )
        assert s.player_out_price == 5.5
        assert s.player_in_price == 5.0


class TestOptimisationResult:
    def test_basic_result(self):
        squad = Squad(picks=[])
        result = OptimisationResult(
            current_squad=squad,
            suggestions=[],
            current_squad_xp=50.0,
            optimised_squad_xp=55.0,
            transfers_used=1,
            point_hit=0,
        )
        assert result.current_squad_xp == 50.0
        assert result.optimised_squad_xp == 55.0
        assert result.transfers_used == 1
        assert result.point_hit == 0
