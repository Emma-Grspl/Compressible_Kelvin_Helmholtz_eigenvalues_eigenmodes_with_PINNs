#!/usr/bin/env python3
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch

import src.scripts.training.atlas.direct_pinn.train_kh_subsonic_2d_pressure_pq_firstorder_mini as mini

# No PNG writes on Jean Zay.
mini.plot_complex_pair = lambda *args, **kwargs: None

RUNS = {
    "mini2d_pq_discrete": Path("assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete"),
    "mini2d_ucore_smooth005": Path("assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_ucore_smooth005_no_modal_anchor"),
    "mini2d_ucore_smooth02": Path("assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_ucore_smooth02_no_modal_anchor"),
}

EXPORT_ROOT = Path("assets/pinn_subsonic/u_reconstruction_diagnostics_M050/fields_export")
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

def parse_float_list_maybe(x):
    if isinstance(x, str):
        return mini.parse_float_list(x)
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]

def load_one(method, run, device):
    ckpt_path = run / "model_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_dict = dict(ckpt["args"])

    export_dir = EXPORT_ROOT / method
    export_dir.mkdir(parents=True, exist_ok=True)

    args_dict["output_dir"] = str(export_dir)
    args_dict.setdefault("n_y", 1500)
    args_dict.setdefault("amp_mask_frac", 0.0)

    args = SimpleNamespace(**args_dict)

    mach_values = parse_float_list_maybe(args_dict["mach_values"])
    mach_min, mach_max = min(mach_values), max(mach_values)

    anchor_df = pd.DataFrame(ckpt["anchor_df"])
    ci_init = float(anchor_df["ci"].mean())

    field = mini.FieldPQNet(
        ymax=float(args_dict["ymax"]),
        alpha_min=float(args_dict["alpha_min"]),
        alpha_max=float(args_dict["alpha_max"]),
        mach_min=mach_min,
        mach_max=mach_max,
        width=int(args_dict["width"]),
        depth=int(args_dict["depth"]),
        n_freq=int(args_dict["n_freq"]),
    ).to(device)

    ci_net = mini.CiNet(
        alpha_min=float(args_dict["alpha_min"]),
        alpha_max=float(args_dict["alpha_max"]),
        mach_min=mach_min,
        mach_max=mach_max,
        ci_init=ci_init,
    ).to(device)

    field.load_state_dict(ckpt["field_state_dict"])
    ci_net.load_state_dict(ckpt["ci_state_dict"])

    field = field.double()
    ci_net = ci_net.double()
    field.eval()
    ci_net.eval()

    return export_dir, args, field, ci_net, anchor_df

def main():
    device = torch.device("cpu")
    print("device =", device)

    for method, run in RUNS.items():
        print("\n" + "=" * 100)
        print("[METHOD]", method)
        print("[RUN]", run)

        export_dir, args, field, ci_net, anchor_df = load_one(method, run, device)

        before = sorted(export_dir.glob("fields_vs_classic_*.csv"))
        print("[before]", len(before))

        mini.run_diagnostics(args, field, ci_net, anchor_df, device)

        after = sorted(export_dir.glob("fields_vs_classic_*.csv"))
        print("[after]", len(after))
        for f in after:
            print(" ", f)

if __name__ == "__main__":
    main()
