"""Weekly automation script.

Runs the full FPL analysis pipeline:
1. Fetch current player data from the FPL API
2. Train GP model and generate xP predictions
3. Optimise the user's team using transfer suggestions

The FPL user ID is read from the FPL_USER_ID environment variable.
Set it as a GitHub Actions secret in your repository settings.
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
    predictor.optimise_team(user_id, max_transfers=2)


if __name__ == "__main__":
    main()
