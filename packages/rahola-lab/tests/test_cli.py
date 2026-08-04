from __future__ import annotations

import subprocess
import sys


def test_importing_cli_does_not_import_optional_deep_learning_stack() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rahola_lab.cli; "
            "assert 'torch' not in sys.modules; assert 'chronos' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
