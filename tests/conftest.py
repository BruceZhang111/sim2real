"""Make the package importable when running tests from a source checkout
(without an editable install), and keep MuJoCo headless during tests."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
