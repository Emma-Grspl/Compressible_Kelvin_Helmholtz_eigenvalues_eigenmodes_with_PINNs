from __future__ import annotations

from pathlib import Path
import sys


PART_II_ROOT = Path(__file__).resolve().parents[1]
CODE_SRC = PART_II_ROOT / "code" / "src"
if str(CODE_SRC) not in sys.path:
    sys.path.insert(0, str(CODE_SRC))
