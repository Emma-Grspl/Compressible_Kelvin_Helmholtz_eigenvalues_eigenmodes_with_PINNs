from __future__ import annotations

import csv
from pathlib import Path


PART_I_ROOT = Path(__file__).resolve().parents[1]


def test_public_routing_config_is_machine_independent_and_complete() -> None:
    route_file = PART_I_ROOT / "configs/atlas/N340_chart_routing.csv"
    with route_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 49
    assert len({row["chart_id"] for row in rows}) == 49
    assert all("/" not in value for row in rows for value in row.values())
    for row in rows:
        assert float(row["mach_min"]) < float(row["mach_max"])
        assert float(row["eta_min"]) < float(row["eta_max"])


def test_classical_configs_are_present() -> None:
    config_dir = PART_I_ROOT / "configs/classical"
    assert (config_dir / "subsonic_pointwise_accuracy_resolution_coupled.yaml").is_file()
    assert (config_dir / "subsonic_pointwise_box_resolution_coupled.yaml").is_file()
