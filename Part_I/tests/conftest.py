from __future__ import annotations

from pathlib import Path
import sys


PART_I_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PART_I_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
