"""Check whether all fixtures in the current gameweek have finished.

Used by the Tuesday post-GW scheduled run. If any matches in the current
gameweek are not yet finished (e.g. midweek fixtures still to be played),
the script exits with a non-zero code so the GitHub Actions workflow skips
the subsequent run/email steps.
"""

import sys

from src.api_client.client import ApiClient


def main() -> None:
    api = ApiClient()

    try:
        current_gw = api.get_current_gw()
    except Exception as e:
        print(f"Error: could not determine current gameweek: {e}")
        print("Allowing run to proceed (no fixture check).")
        return

    print(f"Current gameweek: GW{current_gw}")

    fixtures = api.get_fixtures()
    gw_fixtures = [f for f in fixtures if f.get("event") == current_gw]

    if not gw_fixtures:
        print(f"No fixtures found for GW{current_gw}. Allowing run to proceed.")
        return

    unfinished = [f for f in gw_fixtures if not f.get("finished", False)]

    if unfinished:
        print(f"GW{current_gw} has {len(unfinished)} unfinished fixture(s):")
        for f in unfinished:
            print(f"  Fixture {f['id']}: team_h={f['team_h']} vs team_a={f['team_a']} "
                  f"(kickoff: {f.get('kickoff_time', 'unknown')})")
        print(f"\nSkipping this run — {len(unfinished)} match(es) still to be played.")
        print("The Friday pre-GW run will still proceed as normal.")
        sys.exit(1)
    else:
        print(f"All {len(gw_fixtures)} fixtures in GW{current_gw} are finished.")
        print("Proceeding with post-GW analysis...")


if __name__ == "__main__":
    main()
