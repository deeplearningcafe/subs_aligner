"""Configure test environment."""

from pathlib import Path
import sys

# Ensure the project root is on sys.path so the src package is importable.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
