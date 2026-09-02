from pathlib import Path
import math
import re

import pandas as pd
import torch

from src.scripts.training.atlas.direct_pinn.train_kh_subsonic_2d_pressure_pq_firstorder_mini import (
    FieldPQNet as LegacyFieldPQNet,
)
from src.scripts.training.atlas.direct_pinn.train_subsonic_pinn_seeded_gep_pq2d_continuous_M_alpha_etaaware import (
    FieldPQNet as EtaAwareFieldPQNet,
)

points = pd.read_csv(
    "assets/pinn_subsonic/csv/article/results_pinn/release_final/data/"
    "Table_validation_mode_points_20.csv"
)

results = pd.read_csv(
    "assets/pinn_subsonic/csv/release_v1/validation/"
    "Table_offgrid_validation_results_384_release.csv"
)

extra_columns = [
    "point_id",
    "N",
    "gep_regime",
    "mapping_scale",
    "xi_max",
    "ci_selected",
    "ci_selected_source",
    "ci_final_source",
    "success",
]
extra_columns = [
    c for c in extra_columns if c in results.columns
]

policy = points.merge(
    results[extra_columns].drop_duplicates(
        "point_id",
        keep="last",
    ),
    on="point_id",
    how="left",
)

policy["gep_target_ci"] = pd.to_numeric(
    policy["ci_final"],
    errors="coerce",
)

required = [
    "N",
    "mapping_scale",
    "xi_max",
    "gep_target_ci",
]

missing = {
    column: policy.loc[
        policy[column].isna(),
        "point_id",
    ].astype(str).tolist()
    for column in required
    if policy[column].isna().any()
}

if missing:
    raise RuntimeError(
        f"Missing extraction policy fields: {missing}"
    )

policy.to_csv(
    "mode_extraction_policy_20.csv",
    index=False,
)

inventory_rows = []

for chart_id in sorted(
    policy["chart_id"].astype(str).unique()
):
    checkpoint_path = (
        Path("pinn_subsonic/models")
        / chart_id
        / "model_state.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    args = dict(checkpoint["args"])
    state = checkpoint["field_state_dict"]

    linear_layers = []

    for key, tensor in state.items():
        match = re.fullmatch(
            r"net\.(\d+)\.weight",
            key,
        )
        if match:
            linear_layers.append(
                (
                    int(match.group(1)),
                    tuple(tensor.shape),
                )
            )

    linear_layers.sort()

    input_dimension = int(
        linear_layers[0][1][1]
    )
    output_dimension = int(
        linear_layers[-1][1][0]
    )

    n_freq = int(args["n_freq"])
    legacy_dimension = 3 + 2 * n_freq
    eta_aware_dimension = 7 + 2 * n_freq

    mach_min = float(args["mach_min"])
    mach_max = float(args["mach_max"])
    eta_min = float(args["eta_min"])
    eta_max = float(args["eta_max"])

    alpha_min = eta_min * math.sqrt(
        max(1.0 - mach_max**2, 1.0e-14)
    )
    alpha_max = eta_max * math.sqrt(
        max(1.0 - mach_min**2, 1.0e-14)
    )

    common = dict(
        ymax=float(args["ymax"]),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
        width=int(args["width"]),
        depth=int(args["depth"]),
        n_freq=n_freq,
    )

    if input_dimension == legacy_dimension:
        architecture = "legacy"
        model = LegacyFieldPQNet(
            **common
        ).double()

    elif input_dimension == eta_aware_dimension:
        architecture = "eta-aware"
        model = EtaAwareFieldPQNet(
            **common,
            eta_min=eta_min,
            eta_max=eta_max,
        ).double()

    else:
        raise RuntimeError(
            f"{chart_id}: input dimension "
            f"{input_dimension}; expected "
            f"{legacy_dimension} or "
            f"{eta_aware_dimension}"
        )

    load_result = model.load_state_dict(
        state,
        strict=True,
    )

    family = (
        "pQscaled"
        if "qscaled" in str(
            args.get("output_dir", "")
        ).lower()
        else "pq"
    )

    point = policy.loc[
        policy["chart_id"].astype(str)
        == chart_id
    ].iloc[0]

    y = torch.tensor(
        [[-1.0], [0.0], [1.0]],
        dtype=torch.float64,
    )
    alpha = torch.full_like(
        y,
        float(point["alpha"]),
    )
    mach = torch.full_like(
        y,
        float(point["Mach"]),
    )

    model.eval()

    with torch.no_grad():
        first, second = model(
            y,
            alpha,
            mach,
        )

    finite = bool(
        torch.isfinite(first.real).all()
        and torch.isfinite(first.imag).all()
        and torch.isfinite(second.real).all()
        and torch.isfinite(second.imag).all()
    )

    strict_ok = (
        not load_result.missing_keys
        and not load_result.unexpected_keys
    )

    inventory_rows.append(
        {
            "chart_id": chart_id,
            "architecture": architecture,
            "field_family": family,
            "input_dimension": input_dimension,
            "output_dimension": output_dimension,
            "n_freq": n_freq,
            "strict_load_ok": strict_ok,
            "finite_test": finite,
        }
    )

    print(
        chart_id,
        architecture,
        family,
        f"in={input_dimension}",
        f"out={output_dimension}",
        f"strict={strict_ok}",
        f"finite={finite}",
    )

inventory = pd.DataFrame(inventory_rows)

inventory.to_csv(
    "mode_model_loading_inventory.csv",
    index=False,
)

bad = inventory.loc[
    ~(
        inventory["strict_load_ok"]
        & inventory["finite_test"]
        & inventory["output_dimension"].eq(4)
    )
]

if not bad.empty:
    print("\nFAILED MODELS:")
    print(bad.to_string(index=False))
    raise SystemExit(2)

print("\nWrote: mode_extraction_policy_20.csv")
print("Wrote: mode_model_loading_inventory.csv")
print("ALL CHART MODELS: VALIDATED")
