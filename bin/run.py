"""Weekly automation script.

Runs the full FPL analysis pipeline:
1. Fetch current player data from the FPL API
2. Train GP model and generate xP predictions
3. Optimise the user's team using transfer suggestions

Training/model logs go to stdout (visible in GitHub Actions logs).
Only the team optimisation results are written to the email output file.

The FPL user ID is read from the FPL_USER_ID environment variable.
For the scheduled weekly run, this comes from GitHub Actions secrets.
For manual workflow_dispatch runs, a custom user ID can be passed via
the workflow input, which overrides the secret.

The MongoDB URI is read from MONGODB_URI (always from secrets — this
is the shared database that all users connect through).
"""

import os
import sys

from src.optimisation.predictor import Predictor


def main() -> None:
    user_id_str = os.environ.get("FPL_USER_ID")
    if not user_id_str:
        print("Error: FPL_USER_ID environment variable is not set.")
        print("Set it as a GitHub Actions secret or export it locally:")
        print("  export FPL_USER_ID=your_fpl_user_id")
        sys.exit(1)

    user_id = int(user_id_str)

    predictor = Predictor()

    print("=" * 60)
    print("Step 1: Loading player data from FPL API...")
    print("=" * 60)
    predictor.load_player_data()

    print("\n" + "=" * 60)
    print("Step 2: Training GP model and generating xP predictions...")
    print("=" * 60)
    predictor.model_xp()

    print("\n" + "=" * 60)
    print(f"Step 3: Optimising team for user {user_id}...")
    print("=" * 60)
    result = predictor.optimise_team(user_id, max_transfers=2)

    # Write only the optimisation result to the email output file
    email_path = os.environ.get("EMAIL_OUTPUT_PATH", "tmp/email_content.txt")
    _write_email_content(email_path, user_id, result)


def _write_email_content(path: str, user_id: int, result) -> None:
    """Write the team optimisation results as plain text for the email script."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    lines = []
    lines.append(f"Team Optimisation Results for User {user_id}")
    lines.append("=" * 60)
    lines.append(f"Current squad xP: {result.current_squad_xp}")
    lines.append(f"Optimised squad xP: {result.optimised_squad_xp}")
    lines.append(f"Transfers used: {result.transfers_used}")
    lines.append(f"Point hit: {result.point_hit}")
    lines.append(f"Net improvement: {result.optimised_squad_xp - result.current_squad_xp - result.point_hit:+.2f}")

    if result.suggestions:
        lines.append("")
        lines.append("Suggested transfers:")
        for s in result.suggestions:
            lines.append(f"  {s.player_out_name} (£{s.player_out_price:.1f}m) -> {s.player_in_name} (£{s.player_in_price:.1f}m)")
            lines.append(f"    xP gain: {s.xP_gain:+.2f}, cost change: {s.cost_change:+.2f}")
    else:
        lines.append("")
        lines.append("No beneficial transfers found.")

    lines.append("=" * 60)

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
