"""Central application configuration."""

import os
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", _REPOSITORY_ROOT)).expanduser().resolve()
