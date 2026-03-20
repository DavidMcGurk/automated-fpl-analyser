from src.api_client.client import ApiClient


class Predictor:
    def __init__(self) -> None:
        self.api_client = ApiClient()

    def load_player_data(self) -> None:
        num_players = self.api_client.get_number_of_players()
        for i in range(num_players):
            print(f"Loading data for player {i}")
            data = self.api_client.get_player_info(player_id=i)
            # write data to data folder

    def model_xp(self) -> None:
        # Produce ML model of expected points (xP) / player for next <5 fixtures, from player data and upcoming fixtures
        # Write to folder in data, describing player id, xP / player for remaining up to 5 fixtures, + other rel. info (e.g. position)
        pass

    def optimise_team(self) -> None:
        # Evaluate xP of users current team
        # For each non-selected player with > avg xp, see if they can be included in team + if that improves things
        # (Subsequently) consider pairs of transfers which can most improve xP
        # Relevant constraints: position rules, player prices, club max players, budget (sale prices / player), etc.
        pass
