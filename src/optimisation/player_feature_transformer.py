from statistics import mean

from src.models.player_features import (
    AttackerFeatures,
    DefenderFeatures,
    GoalkeeperFeatures,
    MidfielderFeatures,
)
from src.models.post_prediction import Position


class PlayerFeatureTransformer:

    @classmethod
    def transform(cls, player):
        base = cls._base_features(player)

        match player.position:

            case Position.GOALKEEPER:
                return cls._goalkeeper(player, base)

            case Position.DEFENDER:
                return cls._defender(player, base)

            case Position.MIDFIELDER:
                return cls._midfielder(player, base)

            case Position.ATTACKER:
                return cls._attacker(player, base)

            case _:
                raise ValueError(f"Unsupported position: {player.position}")

    @classmethod
    def _goalkeeper(cls, player, base):
        attrs = player.attributes
        history = player.this_season_history

        saves_last_3 = history[-3:]
        saves_last_5 = history[-5:]

        def saves_per_90(matches):
            minutes = sum(m.minutes for m in matches)

            if minutes == 0:
                return None

            saves = sum(m.saves for m in matches)
            return saves / (minutes / 90)

        return GoalkeeperFeatures(
            **base,
            saves_per_90=attrs.saves_per_90,
            clean_sheets_per_90=attrs.clean_sheets_per_90,
            saves_per_90_last_3=saves_per_90(saves_last_3),
            saves_per_90_last_5=saves_per_90(saves_last_5),
            goals_conceded_per_90=attrs.goals_conceded_per_90,
            penalties_saved=attrs.penalties_saved,
        )

    @classmethod
    def _defender(cls, player, base):
        attrs = player.attributes
        history = player.this_season_history

        matches = history[-5:]

        clean_sheet_rate = sum(m.clean_sheets for m in matches) / len(matches) if matches else None

        return DefenderFeatures(
            **base,
            clean_sheets_per_90=attrs.clean_sheets_per_90,
            expected_goals_conceded_per_90=attrs.expected_goals_conceded_per_90,
            defensive_contribution_per_90=attrs.defensive_contribution_per_90,
            clean_sheet_rate_last_5=clean_sheet_rate,
            expected_goal_involvements_per_90=attrs.expected_goal_involvements_per_90,
            avg_xgi_last_3=cls._rolling_mean(
                history,
                3,
                "expected_goal_involvements",
            ),
            avg_xgi_last_5=cls._rolling_mean(
                history,
                5,
                "expected_goal_involvements",
            ),
        )

    @classmethod
    def _midfielder(cls, player, base):
        attrs = player.attributes
        history = player.this_season_history

        return MidfielderFeatures(
            **base,
            expected_goal_involvements_per_90=attrs.expected_goal_involvements_per_90,
            avg_xg_last_3=cls._rolling_mean(
                history,
                3,
                "expected_goals",
            ),
            avg_xg_last_5=cls._rolling_mean(
                history,
                5,
                "expected_goals",
            ),
            avg_xa_last_3=cls._rolling_mean(
                history,
                3,
                "expected_assists",
            ),
            avg_xa_last_5=cls._rolling_mean(
                history,
                5,
                "expected_assists",
            ),
            avg_set_piece_order=cls._set_piece_score(attrs),
            clean_sheets_per_90=attrs.clean_sheets_per_90,
        )

    @classmethod
    def _attacker(cls, player, base):
        attrs = player.attributes
        history = player.this_season_history

        goals_per_90 = attrs.goals_scored / (attrs.minutes / 90) if attrs.minutes else None
        assists_per_90 = attrs.assists / (attrs.minutes / 90) if attrs.minutes else None

        return AttackerFeatures(
            **base,
            expected_goal_involvements_per_90=attrs.expected_goal_involvements_per_90,
            avg_xg_last_3=cls._rolling_mean(
                history,
                3,
                "expected_goals",
            ),
            avg_xg_last_5=cls._rolling_mean(
                history,
                5,
                "expected_goals",
            ),
            avg_xa_last_3=cls._rolling_mean(
                history,
                3,
                "expected_assists",
            ),
            avg_xa_last_5=cls._rolling_mean(
                history,
                5,
                "expected_assists",
            ),
            goals_per_90=goals_per_90,
            assists_per_90=assists_per_90,
            avg_set_piece_order=cls._set_piece_score(attrs),
        )

    @classmethod
    def _base_features(cls, player):
        attrs = player.attributes
        history = player.this_season_history

        return dict(
            player_id=player.player_id,
            playing_probability=cls._playing_probability(attrs),
            next_fixture_difficulty=cls._fixture_difficulty(
                player.fixtures,
                1,
            ),
            avg_fixture_difficulty_3=cls._fixture_difficulty(
                player.fixtures,
                3,
            ),
            avg_fixture_difficulty_5=cls._fixture_difficulty(
                player.fixtures,
                5,
            ),
            home_fixture_ratio_next_5=cls._home_ratio(
                player.fixtures,
                5,
            ),
            avg_points_last_3=cls._rolling_mean(
                history,
                3,
                "total_points",
                int,
            ),
            avg_points_last_5=cls._rolling_mean(
                history,
                5,
                "total_points",
                int,
            ),
            avg_minutes_last_3=cls._rolling_mean(
                history,
                3,
                "minutes",
                int,
            ),
            avg_minutes_last_5=cls._rolling_mean(
                history,
                5,
                "minutes",
                int,
            ),
            yellow_cards_last_5=cls._rolling_mean(
                history,
                5,
                "yellow_cards",
                int,
            ),
            red_cards_last_5=cls._rolling_mean(
                history,
                5,
                "red_cards",
                int,
            ),
            selected_by_percent=float(attrs.selected_by_percent),
            transfers_balance_last_5=cls._rolling_mean(
                history,
                5,
                "transfers_balance",
                int,
            ),
            now_cost=attrs.now_cost,
            avg_price_diff_historic=cls._historic_price_delta(player.previous_seasons_history),
            avg_points_per_90_historic=cls._historic_points_per_90(player.previous_seasons_history),
            avg_minutes_per_season_historic=cls._historic_minutes(player.previous_seasons_history),
        )

    @staticmethod
    def _rolling_mean(
        history,
        n: int,
        attr: str,
        cast=float,
    ):
        matches = history[-n:]

        if not matches:
            return None

        values = [cast(getattr(match, attr)) for match in matches]
        return mean(values)

    @staticmethod
    def _fixture_difficulty(
        fixtures,
        n: int,
    ):
        fixtures = fixtures[:n]

        if not fixtures:
            return None

        adjusted = [fixture.difficulty - 0.3 if fixture.is_home else fixture.difficulty + 0.3 for fixture in fixtures]
        return mean(adjusted)

    @staticmethod
    def _home_ratio(
        fixtures,
        n: int,
    ):
        fixtures = fixtures[:n]

        if not fixtures:
            return None

        return sum(fixture.is_home for fixture in fixtures) / len(fixtures)

    @staticmethod
    def _playing_probability(attrs):
        current = attrs.chance_of_playing_this_round or 0
        nxt = attrs.chance_of_playing_next_round or current

        return (0.65 * current + 0.35 * nxt) / 100

    @staticmethod
    def _historic_points_per_90(history):
        values = []

        for season in history:
            if season.minutes > 0:
                values.append(season.total_points / (season.minutes / 90))

        return mean(values) if values else None

    @staticmethod
    def _historic_price_delta(history):
        if not history:
            return None

        return mean(season.end_cost - season.start_cost for season in history)

    @staticmethod
    def _historic_minutes(history):
        if not history:
            return None

        return mean(season.minutes for season in history)

    @staticmethod
    def _set_piece_score(attrs):
        scores = []

        for rank in (
            attrs.corners_and_indirect_freekicks_order,
            attrs.direct_freekicks_order,
            attrs.penalties_order,
        ):
            if rank is not None:
                scores.append(max(0, 10 - rank))

        return mean(scores) if scores else None
