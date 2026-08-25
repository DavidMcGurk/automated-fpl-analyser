"""Team optimisation: suggests transfers to maximise xP subject to FPL constraints."""

from itertools import combinations

from src.api_client.client import ApiClient
from src.models.post_prediction import Position
from src.models.squad import (
    OptimisationResult,
    Squad,
    SquadPick,
    TransferSuggestion,
)
from src.storage.mongo_client import MongoStore

POSITION_NAMES = {
    Position.GOALKEEPER: "goalkeepers",
    Position.DEFENDER: "defenders",
    Position.MIDFIELDER: "midfielders",
    Position.ATTACKER: "attackers",
}

# Point hit per extra transfer beyond free transfers
POINT_HIT_PER_TRANSFER = 4


class TeamOptimiser:
    """Optimises a user's FPL squad by suggesting transfers that maximise xP."""

    def __init__(self, store: MongoStore | None = None) -> None:
        self.api_client = ApiClient()
        self.store = store or MongoStore()
        self.predictions: dict[int, dict] = {}
        self.player_names: dict[int, str] = {}
        self.player_positions: dict[int, Position] = {}
        self.player_teams: dict[int, int] = {}
        self.player_prices: dict[int, float] = {}
        self._load_predictions()
        self._load_player_info()

    def _load_predictions(self) -> None:
        """Load xP predictions for all players from MongoDB."""
        self.predictions = self.store.load_all_predictions()

    def _load_player_info(self) -> None:
        """Load player names, positions, teams, and prices from the API."""
        general_info = self.api_client.get_general_info()

        for element in general_info["elements"]:
            player_id = element["id"]
            self.player_names[player_id] = element["web_name"]
            self.player_positions[player_id] = Position(element["element_type"])
            self.player_teams[player_id] = element["team"]
            self.player_prices[player_id] = element["now_cost"] / 10.0

    def get_squad(self, user_id: int) -> Squad:
        """Fetch a user's current squad from the FPL API.

        Falls back to the most recent gameweek that has saved picks if the
        current gameweek has none (e.g. user hasn't set their team yet).
        """
        user_summary = self.api_client.get_user_summary(user_id)

        # Determine the current gameweek
        try:
            current_gw: int | None = self.api_client.get_current_gw()
        except Exception:
            current_gw = user_summary.get("current_event") or user_summary.get("started_event")

        if current_gw is None:
            raise ValueError("Could not determine current gameweek")

        # Try the current GW first, then fall back to earlier GWs
        picks_data = self.api_client.get_user_picks(user_id, current_gw)
        used_gw = current_gw
        if not picks_data.get("picks"):
            started_gw = user_summary.get("started_event") or 1
            for gw in range(current_gw - 1, started_gw - 1, -1):
                picks_data = self.api_client.get_user_picks(user_id, gw)
                if picks_data.get("picks"):
                    used_gw = gw
                    break

        print(f"  Loaded squad from GW{used_gw} ({len(picks_data.get('picks', []))} picks)")

        picks = []
        for pick in picks_data.get("picks", []):
            picks.append(
                SquadPick(
                    element=pick["element"],
                    position=pick["position"],
                    selling_price=pick.get("selling_price", pick.get("purchase_price", 0)) / 10.0,
                    purchase_price=pick.get("purchase_price", 0) / 10.0,
                    is_captain=pick.get("is_captain", False),
                    is_vice_captain=pick.get("is_vice_captain", False),
                    multiplier=pick.get("multiplier", 1),
                )
            )

        bank = (user_summary.get("last_deadline_bank") or 0) / 10.0
        value = (user_summary.get("last_deadline_value") or 0) / 10.0

        return Squad(picks=picks, bank=bank, value=value)

    def get_xp(self, player_id: int) -> float:
        """Get xP for a player, defaulting to 0 if no prediction exists."""
        pred = self.predictions.get(player_id)
        return pred["xp"] if pred else 0.0

    def compute_squad_xp(self, squad: Squad) -> float:
        """Compute total xP for a squad (captain gets double)."""
        total = 0.0
        for pick in squad.picks:
            xp = self.get_xp(pick.element)
            total += xp * pick.multiplier
        return total

    def optimise(
        self,
        user_id: int,
        max_transfers: int = 2,
    ) -> OptimisationResult:
        """Suggest transfers to maximise squad xP.

        Args:
            user_id: FPL user ID
            max_transfers: Maximum number of transfers to consider (1 or 2)

        Returns:
            OptimisationResult with suggested transfers and xP improvement
        """
        squad = self.get_squad(user_id)
        current_xp = self.compute_squad_xp(squad)

        best_result = None
        best_xp = current_xp

        # Try 0, 1, and 2 transfers
        for n_transfers in range(1, max_transfers + 1):
            for players_out in combinations(squad.picks, n_transfers):
                suggestions = self._find_best_replacements(squad, list(players_out))

                if suggestions is None:
                    continue

                new_squad = self._apply_transfers(squad, suggestions)
                new_xp = self.compute_squad_xp(new_squad)

                free_transfers = max(0, squad.free_transfers)
                point_hit = max(0, n_transfers - free_transfers) * POINT_HIT_PER_TRANSFER
                net_improvement = new_xp - current_xp - point_hit

                if net_improvement > (best_xp - current_xp - (best_result.point_hit if best_result else 0)):
                    best_xp = new_xp
                    best_result = OptimisationResult(
                        current_squad=squad,
                        suggestions=suggestions,
                        current_squad_xp=round(current_xp, 2),
                        optimised_squad_xp=round(new_xp, 2),
                        transfers_used=n_transfers,
                        point_hit=point_hit,
                    )

        if best_result is None:
            best_result = OptimisationResult(
                current_squad=squad,
                suggestions=[],
                current_squad_xp=round(current_xp, 2),
                optimised_squad_xp=round(current_xp, 2),
                transfers_used=0,
                point_hit=0,
            )

        # Enrich suggestions with player names
        for suggestion in best_result.suggestions:
            suggestion.player_out_name = self.player_names.get(suggestion.player_out, "Unknown")
            suggestion.player_in_name = self.player_names.get(suggestion.player_in, "Unknown")

        return best_result

    def _find_best_replacements(self, squad: Squad, players_out: list[SquadPick]) -> list[TransferSuggestion] | None:
        """Find the best replacement players for the given outgoing players."""
        # Calculate available budget
        outgoing_value = sum(p.selling_price for p in players_out)
        available_budget = squad.bank + outgoing_value

        # Players remaining in squad (for constraint checking)
        outgoing_ids = {p.element for p in players_out}
        remaining_ids = {p.element for p in squad.picks if p not in players_out}

        # Find all valid replacement combinations
        candidates_by_position: dict[Position, list[int]] = {
            Position.GOALKEEPER: [],
            Position.DEFENDER: [],
            Position.MIDFIELDER: [],
            Position.ATTACKER: [],
        }

        for player_id, pred in self.predictions.items():
            if player_id in remaining_ids:
                continue
            if player_id in outgoing_ids:
                continue

            position = self.player_positions.get(player_id)
            if position is None:
                continue

            price = self.player_prices.get(player_id, 999.0)
            if price <= available_budget:
                candidates_by_position[position].append(player_id)

        # Match positions: outgoing players must be replaced by same-position players
        # (or we can do flexible substitution within squad constraints)
        outgoing_positions = [self.player_positions[p.element] for p in players_out]

        # For simplicity, find best single replacement per outgoing player
        # (same position, highest xP gain, within budget)
        suggestions = []
        used_incoming = set()
        remaining_budget = available_budget

        for i, player_out in enumerate(players_out):
            position = outgoing_positions[i]
            best_player_in = None
            best_xp_gain = -999.0

            for candidate_id in candidates_by_position[position]:
                if candidate_id in used_incoming:
                    continue
                if candidate_id in remaining_ids:
                    continue

                candidate_price = self.player_prices.get(candidate_id, 999.0)
                if candidate_price > remaining_budget:
                    continue

                xp_gain = self.get_xp(candidate_id) - self.get_xp(player_out.element)

                if xp_gain > best_xp_gain:
                    best_xp_gain = xp_gain
                    best_player_in = candidate_id

            if best_player_in is None:
                return None

            used_incoming.add(best_player_in)
            remaining_budget -= self.player_prices[best_player_in]

            suggestions.append(
                TransferSuggestion(
                    player_out=player_out.element,
                    player_in=best_player_in,
                    xP_gain=round(best_xp_gain, 2),
                    cost_change=round(self.player_prices[best_player_in] - player_out.selling_price, 2),
                    net_xp_improvement=0.0,  # Computed at the end
                )
            )

        return suggestions

    def _apply_transfers(self, squad: Squad, suggestions: list[TransferSuggestion]) -> Squad:
        """Create a new squad with the suggested transfers applied."""
        new_picks = []
        transfer_map = {s.player_out: s.player_in for s in suggestions}

        for pick in squad.picks:
            if pick.element in transfer_map:
                new_player_id = transfer_map[pick.element]
                new_price = self.player_prices.get(new_player_id, pick.selling_price)
                new_picks.append(
                    SquadPick(
                        element=new_player_id,
                        position=pick.position,
                        selling_price=new_price,
                        purchase_price=new_price,
                        is_captain=pick.is_captain,
                        is_vice_captain=pick.is_vice_captain,
                        multiplier=pick.multiplier,
                    )
                )
            else:
                new_picks.append(pick)

        # Recalculate bank
        total_cost = sum(p.selling_price for p in new_picks)
        new_bank = squad.BUDGET - total_cost

        return Squad(
            picks=new_picks,
            bank=new_bank,
            value=total_cost,
            free_transfers=squad.free_transfers,
        )

    def _check_squad_valid(self, squad: Squad) -> bool:
        """Check if a squad satisfies FPL constraints."""
        if len(squad.picks) != squad.SQUAD_SIZE:
            return False

        # Position limits
        position_counts = {Position.GOALKEEPER: 0, Position.DEFENDER: 0, Position.MIDFIELDER: 0, Position.ATTACKER: 0}
        club_counts: dict[int, int] = {}

        for pick in squad.picks:
            position = self.player_positions.get(pick.element)
            if position is None:
                return False
            position_counts[position] += 1

            team = self.player_teams.get(pick.element, -1)
            club_counts[team] = club_counts.get(team, 0) + 1

        if position_counts[Position.GOALKEEPER] != squad.MAX_GOALKEEPERS:
            return False
        if position_counts[Position.DEFENDER] != squad.MAX_DEFENDERS:
            return False
        if position_counts[Position.MIDFIELDER] != squad.MAX_MIDFIELDERS:
            return False
        if position_counts[Position.ATTACKER] != squad.MAX_ATTACKERS:
            return False

        # Club limit
        if any(count > squad.MAX_PER_CLUB for count in club_counts.values()):
            return False

        # Budget
        total_cost = sum(p.selling_price for p in squad.picks)
        if total_cost > squad.BUDGET:
            return False

        return True
