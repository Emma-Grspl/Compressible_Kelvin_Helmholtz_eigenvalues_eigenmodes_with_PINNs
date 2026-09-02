from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[4]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))


from src.scripts.evaluation.benchmark_subsonic_local_atlas_core_ci_seeded_gep_v2 import (
    run_one_seed,
)

# Le fournisseur de ci est le même IDW 2D utilisé dans les charts
# p/q et p/Qscaled. Aucun réseau de champs n'est requis pour générer
# le seed spectral.
from src.scripts.training.atlas.direct_pinn.train_subsonic_pinn_seeded_gep_pq2d_continuous_M_alpha_etaaware import (
    CiGridIDW,
)


ATLAS_ROOT = Path(
    "assets/pinn_subsonic/local_atlas_v1"
)

ULTRALOW_PATH = Path(
    "assets/pinn_subsonic/"
    "local_atlas_ext_M002_006_eta002_006/"
    "seedGEP_pQscaled2d_ULTRALOW_M002_006_eta002_006_ciIDW"
)


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    return None


def load_chart_catalog() -> pd.DataFrame:
    sources = [
        (
            "core",
            ATLAS_ROOT / "atlas_manifest.csv",
        ),
        (
            "extension",
            ATLAS_ROOT
            / "atlas_extension_plan_M005_098_eta005_098.csv",
        ),
        (
            "fullrect",
            ATLAS_ROOT
            / "atlas_fullrect_plan_M002_098_eta002_098.csv",
        ),
    ]

    rows: list[dict] = []

    for source_name, path in sources:
        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)

        chart_column = find_column(
            frame,
            ["chart_id", "atlas_id", "name"],
        )

        path_column = find_column(
            frame,
            ["path", "output_dir", "run_dir"],
        )

        status_column = find_column(
            frame,
            ["status", "chart_status"],
        )

        if chart_column is None:
            raise RuntimeError(
                f"{path}: no chart identifier column; "
                f"columns={list(frame.columns)}"
            )

        if path_column is None:
            raise RuntimeError(
                f"{path}: no path/output_dir column; "
                f"columns={list(frame.columns)}"
            )

        for _, row in frame.iterrows():
            rows.append(
                {
                    "selected_source": source_name,
                    "chart_id": str(row[chart_column]),
                    "chart_path": str(row[path_column]),
                    "chart_status": (
                        str(row[status_column])
                        if status_column is not None
                        and pd.notna(row[status_column])
                        else "ci_only"
                    ),
                }
            )

    rows.append(
        {
            "selected_source": "ultralow_explicit",
            "chart_id": "ULTRALOW_M002_006_eta002_006",
            "chart_path": str(ULTRALOW_PATH),
            "chart_status": "ci_only",
        }
    )

    catalog = pd.DataFrame(rows)

    catalog = catalog.drop_duplicates(
        [
            "selected_source",
            "chart_id",
            "chart_path",
        ]
    ).reset_index(drop=True)

    return catalog


def resolve_chart(
    *,
    catalog: pd.DataFrame,
    chart_id: str,
    selected_source: str,
) -> pd.Series:
    exact = catalog[
        (catalog["chart_id"] == chart_id)
        & (
            catalog["selected_source"]
            == selected_source
        )
    ]

    if len(exact) == 1:
        return exact.iloc[0]

    fallback = catalog[
        catalog["chart_id"] == chart_id
    ]

    if len(fallback) == 1:
        return fallback.iloc[0]

    if len(exact) > 1:
        candidates = exact
    else:
        candidates = fallback

    raise RuntimeError(
        "Unable to uniquely resolve chart "
        f"{chart_id!r}, source={selected_source!r}.\n"
        f"Candidates:\n{candidates.to_string(index=False)}"
    )


class SeedProvider:
    def __init__(self, chart_path: Path):
        self.chart_path = chart_path
        checkpoint_path = chart_path / "model_best.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        if "anchor_df" not in checkpoint:
            raise RuntimeError(
                f"{checkpoint_path}: anchor_df missing"
            )

        if "args" not in checkpoint:
            raise RuntimeError(
                f"{checkpoint_path}: args missing"
            )

        anchor_df = pd.DataFrame(
            checkpoint["anchor_df"]
        )

        arguments = dict(checkpoint["args"])

        eta_scale = float(
            arguments.get(
                "ci_idw_eta_scale",
                0.25,
            )
        )

        mach_scale = float(
            arguments.get(
                "ci_idw_mach_scale",
                0.25,
            )
        )

        power = float(
            arguments.get(
                "ci_idw_power",
                4.0,
            )
        )

        self.provider = CiGridIDW(
            anchor_df=anchor_df,
            eta_scale=eta_scale,
            mach_scale=mach_scale,
            power=power,
            eps=1.0e-12,
        )

        self.provider = self.provider.to(
            device="cpu",
            dtype=torch.float64,
        )
        self.provider.eval()

        self.metadata = {
            "checkpoint_path": str(checkpoint_path),
            "n_ci_anchors": len(anchor_df),
            "ci_idw_eta_scale": eta_scale,
            "ci_idw_mach_scale": mach_scale,
            "ci_idw_power": power,
        }

    def predict(
        self,
        *,
        alpha: float,
        Mach: float,
    ) -> float:
        alpha_tensor = torch.tensor(
            [[alpha]],
            dtype=torch.float64,
        )

        mach_tensor = torch.tensor(
            [[Mach]],
            dtype=torch.float64,
        )

        with torch.no_grad():
            value = self.provider(
                alpha_tensor,
                mach_tensor,
            )

        return float(
            value.detach().cpu().reshape(-1)[0]
        )


def atomic_write(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(path)


def failure_row(
    point: pd.Series,
    *,
    ci_seed: float | None,
    chart_path: str | None,
    error: Exception,
) -> dict:
    return {
        "point_id": point["point_id"],
        "sample_group": point["sample_group"],
        "alpha": float(point["alpha"]),
        "eta": float(point["eta"]),
        "Mach": float(point["Mach"]),
        "chart_id": str(point["selected_chart"]),
        "selected_source": str(
            point["selected_source"]
        ),
        "chart_path": chart_path,
        "chart_status": "ci_only",
        "gep_regime": str(point["gep_regime"]),
        "N": int(point["N"]),
        "mapping_scale": float(
            point["mapping_scale"]
        ),
        "xi_max": float(point["xi_max"]),
        "continuation_required": bool(
            point["continuation_required"]
        ),
        "ci_classic": np.nan,
        "ci_seed": (
            float(ci_seed)
            if ci_seed is not None
            else np.nan
        ),
        "ci_seed_abs_err": np.nan,
        "ci_seed_rel_err": np.nan,
        "gep_cr": np.nan,
        "gep_ci": np.nan,
        "ci_gep_abs_err": np.nan,
        "ci_gep_rel_err": np.nan,
        "p_rel": np.nan,
        "rho_rel": np.nan,
        "u_rel": np.nan,
        "v_rel": np.nan,
        "p_overlap": np.nan,
        "selection_source": (
            f"ERROR: {type(error).__name__}: {error}"
        ),
        "n_finite_modes": 0,
        "success": False,
        "traceback": traceback.format_exc(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--points-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError(
            "--num-shards must be >= 1"
        )

    if not (
        0 <= args.shard_index < args.num_shards
    ):
        raise ValueError(
            "--shard-index must satisfy "
            "0 <= shard-index < num-shards"
        )

    points = pd.read_csv(args.points_csv)

    required = {
        "point_id",
        "sample_group",
        "Mach",
        "eta",
        "alpha",
        "selected_chart",
        "selected_source",
        "gep_regime",
        "N",
        "mapping_scale",
        "xi_max",
        "continuation_required",
    }

    missing = required - set(points.columns)

    if missing:
        raise RuntimeError(
            f"Points CSV missing columns: "
            f"{sorted(missing)}"
        )

    points = points.reset_index(drop=True)

    points = points[
        points.index % args.num_shards
        == args.shard_index
    ].copy()

    if args.max_points > 0:
        points = points.head(
            args.max_points
        ).copy()

    catalog = load_chart_catalog()

    previous_rows: list[dict] = []
    completed_ids: set[str] = set()

    if (
        args.resume
        and args.output_csv.exists()
    ):
        previous = pd.read_csv(
            args.output_csv
        )

        previous_rows = (
            previous.to_dict(orient="records")
        )

        completed_ids = set(
            previous["point_id"].astype(str)
        )

    points = points[
        ~points["point_id"]
        .astype(str)
        .isin(completed_ids)
    ].copy()

    print(
        "===== OFF-GRID SHARD ====="
    )
    print(
        f"shard={args.shard_index}/"
        f"{args.num_shards}"
    )
    print(
        f"remaining_points={len(points)}"
    )
    print(
        f"output={args.output_csv}"
    )

    provider_cache: dict[str, SeedProvider] = {}
    rows = list(previous_rows)

    for counter, (_, point) in enumerate(
        points.iterrows(),
        start=1,
    ):
        point_id = str(point["point_id"])
        chart_id = str(
            point["selected_chart"]
        )
        selected_source = str(
            point["selected_source"]
        )

        Mach = float(point["Mach"])
        eta = float(point["eta"])
        alpha = float(point["alpha"])

        chart_path: str | None = None
        ci_seed: float | None = None

        print(
            f"\n[{counter}/{len(points)}] "
            f"{point_id} "
            f"M={Mach:.8f} "
            f"eta={eta:.8f} "
            f"alpha={alpha:.8f} "
            f"chart={chart_id} "
            f"regime={point['gep_regime']}"
        )

        try:
            chart = resolve_chart(
                catalog=catalog,
                chart_id=chart_id,
                selected_source=selected_source,
            )

            chart_path = str(
                chart["chart_path"]
            )

            if chart_path not in provider_cache:
                provider_cache[chart_path] = (
                    SeedProvider(
                        Path(chart_path)
                    )
                )

                print(
                    "  loaded seed provider:",
                    chart_path,
                )

            provider = provider_cache[
                chart_path
            ]

            ci_seed = provider.predict(
                alpha=alpha,
                Mach=Mach,
            )

            result = run_one_seed(
                alpha=alpha,
                mach=Mach,
                eta=eta,
                ci_seed=ci_seed,
                chart_id=chart_id,
                chart_status=str(
                    chart["chart_status"]
                ),
                n_points=int(point["N"]),
                mapping_kind="pin",
                mapping_scale=float(
                    point["mapping_scale"]
                ),
                xi_max=float(
                    point["xi_max"]
                ),
            )

            result.update(
                {
                    "point_id": point_id,
                    "sample_group": str(
                        point["sample_group"]
                    ),
                    "selected_source": (
                        selected_source
                    ),
                    "chart_path": chart_path,
                    "gep_regime": str(
                        point["gep_regime"]
                    ),
                    "mapping_scale": float(
                        point["mapping_scale"]
                    ),
                    "xi_max": float(
                        point["xi_max"]
                    ),
                    "continuation_required": bool(
                        point[
                            "continuation_required"
                        ]
                    ),
                    "checkpoint_path": (
                        provider.metadata[
                            "checkpoint_path"
                        ]
                    ),
                    "n_ci_anchors": (
                        provider.metadata[
                            "n_ci_anchors"
                        ]
                    ),
                    "ci_idw_eta_scale": (
                        provider.metadata[
                            "ci_idw_eta_scale"
                        ]
                    ),
                    "ci_idw_mach_scale": (
                        provider.metadata[
                            "ci_idw_mach_scale"
                        ]
                    ),
                    "ci_idw_power": (
                        provider.metadata[
                            "ci_idw_power"
                        ]
                    ),
                    "traceback": "",
                }
            )

        except Exception as error:
            print(
                "  ERROR:",
                type(error).__name__,
                error,
            )

            result = failure_row(
                point,
                ci_seed=ci_seed,
                chart_path=chart_path,
                error=error,
            )

        rows.append(result)

        current = pd.DataFrame(rows)

        if "point_id" in current.columns:
            current = current.drop_duplicates(
                "point_id",
                keep="last",
            )

            current = current.sort_values(
                "point_id"
            ).reset_index(drop=True)

        atomic_write(
            current,
            args.output_csv,
        )

        print(
            "  success=",
            result.get("success"),
            "ci_seed_rel=",
            result.get("ci_seed_rel_err"),
            "ci_gep_rel=",
            result.get("ci_gep_rel_err"),
            "u_rel=",
            result.get("u_rel"),
            "overlap=",
            result.get("p_overlap"),
        )

    print()
    print("===== SHARD FINISHED =====")
    print("Output:", args.output_csv)


if __name__ == "__main__":
    main()
