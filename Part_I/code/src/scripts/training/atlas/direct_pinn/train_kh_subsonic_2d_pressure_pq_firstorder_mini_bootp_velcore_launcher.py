import argparse
import sys
import torch

import src.scripts.training.atlas.direct_pinn.train_kh_subsonic_2d_pressure_pq_firstorder_mini as pqmini

ap = argparse.ArgumentParser(add_help=False)

ap.add_argument("--w-u-core-smooth", type=float, default=0.0)
ap.add_argument("--w-u-core-amp", type=float, default=0.0)
ap.add_argument("--u-core-ymax", type=float, default=2.0)

ap.add_argument("--w-vel-core-curv", type=float, default=0.0)
ap.add_argument("--w-vel-core-grad", type=float, default=0.0)
ap.add_argument("--vel-core-ymax", type=float, default=0.75)

extra_args, remaining = ap.parse_known_args()
sys.argv = [sys.argv[0]] + remaining

_base_physics_losses = pqmini.physics_losses

def abs2(z):
    return z.real**2 + z.imag**2

def get_arg(args, kwargs, pos, name):
    if len(args) > pos:
        return args[pos]
    return kwargs[name]

def smooth_core_mask(y, ymax):
    yr = y.real if torch.is_complex(y) else y
    return torch.exp(-((yr / float(ymax)) ** 4)).to(yr.dtype)

def physics_losses_velcore(*args, **kwargs):
    losses = _base_physics_losses(*args, **kwargs)

    field = get_arg(args, kwargs, 0, "field")
    ci_net = get_arg(args, kwargs, 1, "ci_net")
    y = get_arg(args, kwargs, 2, "y")
    alpha = get_arg(args, kwargs, 3, "alpha")
    mach = get_arg(args, kwargs, 4, "mach")

    p, q = field(y, alpha, mach)

    extra_penalty = torch.zeros((), device=y.device, dtype=y.real.dtype)

    if float(extra_args.w_u_core_smooth) > 0.0:
        mask_u = (torch.abs(y) <= float(extra_args.u_core_ymax)).to(y.real.dtype)
        qy = pqmini.grad_complex(q, y)
        pen = float(extra_args.w_u_core_smooth) * torch.mean(mask_u * abs2(qy))
        losses["u_core_smooth"] = pen
        extra_penalty = extra_penalty + pen

    if float(extra_args.w_u_core_amp) > 0.0:
        mask_u = (torch.abs(y) <= float(extra_args.u_core_ymax)).to(y.real.dtype)
        pen = float(extra_args.w_u_core_amp) * torch.mean(mask_u * abs2(q))
        losses["u_core_amp"] = pen
        extra_penalty = extra_penalty + pen

    if float(extra_args.w_vel_core_grad) > 0.0 or float(extra_args.w_vel_core_curv) > 0.0:
        ci = ci_net(alpha, mach)
        rho, u, v, gamma = pqmini.fields_from_pq(y, p, q, alpha, mach, ci)
        mask_v = smooth_core_mask(y, float(extra_args.vel_core_ymax))

        if float(extra_args.w_vel_core_grad) > 0.0:
            uy = pqmini.grad_complex(u, y)
            vy = pqmini.grad_complex(v, y)
            pen = float(extra_args.w_vel_core_grad) * torch.mean(mask_v * (abs2(uy) + abs2(vy)))
            losses["vel_core_grad"] = pen
            extra_penalty = extra_penalty + pen

        if float(extra_args.w_vel_core_curv) > 0.0:
            uy = pqmini.grad_complex(u, y)
            vy = pqmini.grad_complex(v, y)
            uyy = pqmini.grad_complex(uy, y)
            vyy = pqmini.grad_complex(vy, y)
            pen = float(extra_args.w_vel_core_curv) * torch.mean(mask_v * (abs2(uyy) + abs2(vyy)))
            losses["vel_core_curv"] = pen
            extra_penalty = extra_penalty + pen

    # Critical point:
    # bootp.py only sums known keys. Inject the new physics penalty into "ode",
    # which is already included in the Phase B total loss.
    if extra_penalty.detach().abs().item() > 0.0:
        losses["ode"] = losses["ode"] + extra_penalty

    return losses

pqmini.physics_losses = physics_losses_velcore

import src.scripts.training.atlas.direct_pinn.train_kh_subsonic_2d_pressure_pq_firstorder_mini_bootp as bootp

bootp.main()
