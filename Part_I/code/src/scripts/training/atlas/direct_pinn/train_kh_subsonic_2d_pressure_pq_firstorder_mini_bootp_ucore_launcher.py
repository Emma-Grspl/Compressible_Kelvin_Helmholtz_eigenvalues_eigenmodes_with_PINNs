import argparse
import sys
import torch

import src.scripts.training.atlas.direct_pinn.train_kh_subsonic_2d_pressure_pq_firstorder_mini as pqmini

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--w-u-core-smooth", type=float, default=0.0)
ap.add_argument("--w-u-core-amp", type=float, default=0.0)
ap.add_argument("--u-core-ymax", type=float, default=2.0)

ucore_args, remaining = ap.parse_known_args()
sys.argv = [sys.argv[0]] + remaining

_base_physics_losses = pqmini.physics_losses

def physics_losses_ucore(*args, **kwargs):
    losses = _base_physics_losses(*args, **kwargs)

    if float(ucore_args.w_u_core_smooth) <= 0.0 and float(ucore_args.w_u_core_amp) <= 0.0:
        return losses

    if len(args) >= 5:
        field = args[0]
        y = args[2]
        alpha = args[3]
        mach = args[4]
    else:
        field = kwargs["field"]
        y = kwargs["y"]
        alpha = kwargs["alpha"]
        mach = kwargs["mach"]

    _, q = field(y, alpha, mach)
    core_mask = (torch.abs(y) <= float(ucore_args.u_core_ymax)).to(y.real.dtype)

    if float(ucore_args.w_u_core_smooth) > 0.0:
        qy = pqmini.grad_complex(q, y)
        losses["u_core_smooth"] = float(ucore_args.w_u_core_smooth) * torch.mean(
            core_mask * (qy.real**2 + qy.imag**2)
        )

    if float(ucore_args.w_u_core_amp) > 0.0:
        losses["u_core_amp"] = float(ucore_args.w_u_core_amp) * torch.mean(
            core_mask * (q.real**2 + q.imag**2)
        )

    return losses

pqmini.physics_losses = physics_losses_ucore

import src.scripts.training.atlas.direct_pinn.train_kh_subsonic_2d_pressure_pq_firstorder_mini_bootp as bootp

bootp.main()
