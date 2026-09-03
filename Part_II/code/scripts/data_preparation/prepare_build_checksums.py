#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent
SHOWCASE = (
    REPO
    / "assets"
    / "classic_supersonic"
    / "reference_v2"
)
OUTPUT = (
    PACKAGE
    / "provenance"
    / "SHA256SUMS.txt"
)

EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    ".venv-release",
}

EXCLUDED_FILES = {
    "SHA256SUMS.txt",
    "supersonic_reference_v2_modal_raw.parquet",
    "supersonic_reference_v2_modal_tail_polished.parquet",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def include(path: Path) -> bool:
    if path.name in EXCLUDED_FILES:
        return False

    if any(
        part in EXCLUDED_PARTS
        for part in path.parts
    ):
        return False

    if any(
        part.endswith(".egg-info")
        for part in path.parts
    ):
        return False

    try:
        relative = path.relative_to(PACKAGE)
    except ValueError:
        relative = None

    if (
        relative is not None
        and relative.parts[:2]
        == ("reproducibility", "results")
    ):
        return False

    return path.is_file()


def main() -> None:
    files = []

    for root in [PACKAGE, SHOWCASE]:
        files.extend(
            path
            for path in root.rglob("*")
            if include(path)
        )

    files = sorted(set(files))

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w") as stream:
        for path in files:
            stream.write(
                f"{sha256(path)}  "
                f"{path.relative_to(REPO)}\n"
            )

    print(
        f"Wrote {OUTPUT.relative_to(REPO)}"
    )
    print(
        f"Files checksummed: {len(files)}"
    )


if __name__ == "__main__":
    main()
