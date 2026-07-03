"""Make `daemon/src` the import root for lifecycle tests."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
