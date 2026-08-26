"""Report the exact parameter count for Model L (Nepali).

Spec 2.1 "Configuration and reporting": "report the exact parameter count
for each model". Asserts the target so a config edit cannot silently drift
the model off ~25M.

    python nepali/train/count_params.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.train.runner import print_parameter_report   # noqa: E402

EXPECTED_PARAMETERS = 25_321_472

if __name__ == "__main__":
    print_parameter_report(
        str(REPO / "nepali/configs/nepali_model.yaml"),
        REPO,
        expect=EXPECTED_PARAMETERS,
    )
