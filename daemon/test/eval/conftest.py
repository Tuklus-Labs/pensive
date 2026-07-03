"""Make `daemon/src` and `daemon` import roots for eval tests."""
import sys
from pathlib import Path

DAEMON = Path(__file__).resolve().parent.parent.parent
SRC = DAEMON / "src"
for path in (DAEMON, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
