from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_oracle_diagnostics_cli_exposes_real_shard_inputs() -> None:
    # Given
    experiment_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(experiment_root)

    # When
    completed = subprocess.run(
        [sys.executable, "-m", "step_distill.oracle_diagnostics", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    # Then
    assert completed.returncode == 0
    assert "--input-root" in completed.stdout
    assert "--order-kind" in completed.stdout
