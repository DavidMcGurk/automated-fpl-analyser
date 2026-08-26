"""Tests for TeamOptimiser."""

from unittest.mock import MagicMock, patch

from src.optimisation.team_optimiser import TeamOptimiser
from src.models.squad import Squad, SquadPick, TransferSuggestion
from src.models.post_prediction import Position


def _make_optimiser(predictions=None, player_info=None):
    """Create a TeamOptimiser with mocked API and pre-loaded data."""
    with patch.object(TeamOptimiser, "__init__", return_value=None):
        opt = TeamOptimiser.__new__(TeamOptimiser)
        opt.api_client = MagicMock()
        opt.predictions = predictions or {}
        opt.player_names = player_info.get("names", {}) if player_info else {}
        opt.player_positions = player_info.get("positions", {}) if player_info else {}
        opt.player_teams = player_info.get("teams", {}) if player_info else {}
        opt.player_prices = player_info.get("prices", {}) if player_info else {}
        return opt


class TestComputeSquadXp:
    def test_empty_squad(self):
        opt = _make_optimiser()
        squad = Squad(picks=[])
        assert opt.compute_squad_xp(squad) == 0.0

    def test_basic_squad(self):
        opt = _make_optimiser(predictions={1: {"xp": 5.0}, 2: {"xp": 3.0}})
        squad = Squad(
            picks=[
                SquadPick(element=1, position=1, selling_price=5.0, purchase_price=5.0, multiplier=1),
                SquadPick(element=2, position=2, selling_price=4.0, purchase_price=4.0, multiplier=1),
            ]
        )
        assert opt.compute_squad_xp(squad) == 8.0

    def test_captain_double(self):
        opt = _make_optimiser(predictions={1: {"xp": 5.0}})
        squad = Squad(
            picks=[
                SquadPick(element=1, position=1, selling_price=5.0, purchase_price=5.0, multiplier=2),
            ]
        )
        assert opt.compute_squad_xp(squad) == 10.0


class TestGetXp:
    def test_existing_player(self):
        opt = _make_optimiser(predictions={42: {"xp": 7.5}})
        assert opt.get_xp(42) == 7.5

    def test_missing_player(self):
        opt = _make_optimiser(predictions={})
        assert opt.get_xp(999) == 0.0


class TestApplyTransfers:
    def test_single_transfer(self):
        opt = _make_optimiser(
            predictions={1: {"xp": 5.0}, 2: {"xp": 8.0}},
            player_info={"prices": {2: 6.0}},
        )
        squad = Squad(
            picks=[
                SquadPick(element=1, position=1, selling_price=5.0, purchase_price=5.0),
            ],
            bank=0.0,
            value=5.0,
        )
        suggestions = [
            TransferSuggestion(
                player_out=1,
                player_in=2,
                xP_gain=3.0,
                cost_change=1.0,
                net_xp_improvement=3.0,
            )
        ]
        new_squad = opt._apply_transfers(squad, suggestions)
        assert new_squad.picks[0].element == 2
        assert new_squad.picks[0].selling_price == 6.0

    def test_no_transfers(self):
        opt = _make_optimiser()
        squad = Squad(
            picks=[
                SquadPick(element=1, position=1, selling_price=5.0, purchase_price=5.0),
            ],
            bank=0.0,
            value=5.0,
        )
        new_squad = opt._apply_transfers(squad, [])
        assert new_squad.picks[0].element == 1


class TestCheckSquadValid:
    def test_valid_squad(self):
        opt = _make_optimiser(
            player_info={
                "positions": {
                    i: pos
                    for i, pos in enumerate(
                        [
                            Position.GOALKEEPER,
                            Position.GOALKEEPER,
                            Position.DEFENDER,
                            Position.DEFENDER,
                            Position.DEFENDER,
                            Position.DEFENDER,
                            Position.DEFENDER,
                            Position.MIDFIELDER,
                            Position.MIDFIELDER,
                            Position.MIDFIELDER,
                            Position.MIDFIELDER,
                            Position.MIDFIELDER,
                            Position.ATTACKER,
                            Position.ATTACKER,
                            Position.ATTACKER,
                        ],
                        start=1,
                    )
                },
                "teams": {i: ((i - 1) // 3) + 1 for i in range(1, 16)},  # Spread across 5 clubs
            }
        )
        picks = [SquadPick(element=i, position=i, selling_price=5.0, purchase_price=5.0) for i in range(1, 16)]
        squad = Squad(picks=picks, bank=25.0, value=75.0)
        assert opt._check_squad_valid(squad) is True

    def test_wrong_squad_size(self):
        opt = _make_optimiser()
        squad = Squad(picks=[])
        assert opt._check_squad_valid(squad) is False

    def test_too_many_from_club(self):
        opt = _make_optimiser(
            player_info={
                "positions": {
                    i: pos
                    for i, pos in enumerate(
                        [
                            Position.GOALKEEPER,
                            Position.GOALKEEPER,
                            Position.DEFENDER,
                            Position.DEFENDER,
                            Position.DEFENDER,
                            Position.DEFENDER,
                            Position.DEFENDER,
                            Position.MIDFIELDER,
                            Position.MIDFIELDER,
                            Position.MIDFIELDER,
                            Position.MIDFIELDER,
                            Position.MIDFIELDER,
                            Position.ATTACKER,
                            Position.ATTACKER,
                            Position.ATTACKER,
                        ],
                        start=1,
                    )
                },
                "teams": {i: 1 for i in range(1, 16)},  # All from same team
            }
        )
        picks = [SquadPick(element=i, position=i, selling_price=5.0, purchase_price=5.0) for i in range(1, 16)]
        squad = Squad(picks=picks, bank=25.0, value=75.0)
        assert opt._check_squad_valid(squad) is False

    def test_over_budget(self):
        opt = _make_optimiser(
            player_info={
                "positions": {
                    i: pos
                    for i, pos in enumerate(
                        [
                            Position.GOALKEEPER,
                            Position.GOALKEEPER,
                            Position.DEFENDER,
                            Position.DEFENDER,
                            Position.DEFENDER,
                            Position.DEFENDER,
                            Position.DEFENDER,
                            Position.MIDFIELDER,
                            Position.MIDFIELDER,
                            Position.MIDFIELDER,
                            Position.MIDFIELDER,
                            Position.MIDFIELDER,
                            Position.ATTACKER,
                            Position.ATTACKER,
                            Position.ATTACKER,
                        ],
                        start=1,
                    )
                },
                "teams": {i: ((i - 1) // 3) + 1 for i in range(1, 16)},
            }
        )
        picks = [SquadPick(element=i, position=i, selling_price=8.0, purchase_price=8.0) for i in range(1, 16)]
        squad = Squad(picks=picks, bank=0.0, value=120.0)
        assert opt._check_squad_valid(squad) is False


class TestFindBestReplacements:
    def test_finds_better_player(self):
        opt = _make_optimiser(
            predictions={
                1: {"xp": 2.0},
                10: {"xp": 5.0},
                11: {"xp": 4.0},
            },
            player_info={
                "positions": {1: Position.GOALKEEPER, 10: Position.GOALKEEPER, 11: Position.GOALKEEPER},
                "prices": {1: 4.0, 10: 4.0, 11: 4.5},
                "teams": {1: 1, 10: 2, 11: 3},
            },
        )
        squad = Squad(
            picks=[
                SquadPick(element=1, position=1, selling_price=4.0, purchase_price=4.0),
            ],
            bank=1.0,
            value=4.0,
        )
        players_out = [squad.picks[0]]
        result = opt._find_best_replacements(squad, players_out)
        assert result is not None
        assert result[0].player_in == 10  # Higher xP, within budget
        assert result[0].xP_gain == 3.0

    def test_no_affordable_replacement(self):
        opt = _make_optimiser(
            predictions={1: {"xp": 2.0}, 10: {"xp": 10.0}},
            player_info={
                "positions": {1: Position.GOALKEEPER, 10: Position.GOALKEEPER},
                "prices": {1: 4.0, 10: 10.0},
                "teams": {1: 1, 10: 2},
            },
        )
        squad = Squad(
            picks=[
                SquadPick(element=1, position=1, selling_price=4.0, purchase_price=4.0),
            ],
            bank=0.0,
            value=4.0,
        )
        players_out = [squad.picks[0]]
        result = opt._find_best_replacements(squad, players_out)
        # Player 10 costs 10.0 but budget is only 4.0
        assert result is None


class TestGetSquadFallback:
    """Tests that get_squad falls back to the most recent GW with picks."""

    def test_falls_back_to_previous_gw(self):
        opt = _make_optimiser()
        opt.api_client.get_user_summary.return_value = {
            "last_deadline_bank": 500,
            "last_deadline_value": 5000,
            "started_event": 1,
        }
        opt.api_client.get_current_gw.return_value = 2
        opt.api_client.get_user_transfers.return_value = []
        # GW2 has no picks, GW1 has picks
        opt.api_client.get_user_picks.side_effect = [
            {"picks": []},  # GW2
            {"picks": [{"element": 1, "position": 1, "selling_price": 50, "purchase_price": 50}]},  # GW1
        ]

        squad = opt.get_squad(123)
        assert len(squad.picks) == 1
        assert squad.picks[0].element == 1

    def test_uses_current_gw_when_picks_exist(self):
        opt = _make_optimiser()
        opt.api_client.get_user_summary.return_value = {
            "last_deadline_bank": 500,
            "last_deadline_value": 5000,
            "started_event": 1,
        }
        opt.api_client.get_current_gw.return_value = 2
        opt.api_client.get_user_transfers.return_value = []
        opt.api_client.get_user_picks.return_value = {
            "picks": [{"element": 42, "position": 1, "selling_price": 60, "purchase_price": 60}],
        }

        squad = opt.get_squad(123)
        assert len(squad.picks) == 1
        assert squad.picks[0].element == 42
        # Should only call get_user_picks once (for current GW)
        assert opt.api_client.get_user_picks.call_count == 1


class TestGetSquadPriceFallback:
    """Tests that get_squad falls back to now_cost when API doesn't provide selling_price."""

    def test_uses_now_cost_when_no_selling_price(self):
        """FPL API doesn't include selling_price in picks for some gameweeks.

        The squad should fall back to the player's current now_cost from the API.
        """
        opt = _make_optimiser(
            player_info={
                "prices": {42: 6.5},
            }
        )
        opt.api_client.get_user_summary.return_value = {
            "last_deadline_bank": 0,
            "last_deadline_value": 650,
            "started_event": 1,
        }
        opt.api_client.get_current_gw.return_value = 1
        opt.api_client.get_user_transfers.return_value = []
        # No selling_price or purchase_price in picks data
        opt.api_client.get_user_picks.return_value = {
            "picks": [{"element": 42, "position": 1, "multiplier": 1}],
        }

        squad = opt.get_squad(123)
        assert squad.picks[0].selling_price == 6.5
        assert squad.picks[0].purchase_price == 6.5


class TestComputeFreeTransfers:
    """Tests for free transfer calculation from the FPL transfer history API."""

    def test_first_gw_no_transfers(self):
        opt = _make_optimiser()
        opt.api_client.get_user_transfers.return_value = []
        assert opt._compute_free_transfers(user_id=123, current_gw=1, used_gw=1) == 1

    def test_first_gw_one_transfer_used(self):
        opt = _make_optimiser()
        opt.api_client.get_user_transfers.return_value = [{"event": 1}]
        assert opt._compute_free_transfers(user_id=123, current_gw=1, used_gw=1) == 0

    def test_second_gw_no_transfers_prev_gw(self):
        """If no transfers were made in the previous GW, allowance is 2."""
        opt = _make_optimiser()
        opt.api_client.get_user_transfers.return_value = []
        assert opt._compute_free_transfers(user_id=123, current_gw=2, used_gw=2) == 2

    def test_second_gw_transfers_prev_gw(self):
        """If transfers were made in the previous GW, allowance is 1."""
        opt = _make_optimiser()
        opt.api_client.get_user_transfers.return_value = [{"event": 1}]
        assert opt._compute_free_transfers(user_id=123, current_gw=2, used_gw=2) == 1

    def test_second_gw_one_transfer_this_gw_with_carryover(self):
        """Used 1 of 2 free transfers (carryover from no transfers last GW)."""
        opt = _make_optimiser()
        opt.api_client.get_user_transfers.return_value = [{"event": 2}]
        assert opt._compute_free_transfers(user_id=123, current_gw=2, used_gw=2) == 1

    def test_api_error_defaults_to_one(self):
        """If the transfer API fails, default to 1 free transfer."""
        opt = _make_optimiser()
        opt.api_client.get_user_transfers.side_effect = Exception("API error")
        assert opt._compute_free_transfers(user_id=123, current_gw=3, used_gw=3) == 1
