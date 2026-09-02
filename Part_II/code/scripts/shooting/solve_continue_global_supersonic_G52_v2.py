#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.training.train_global_supersonic_kappa_q_logamp import (
    build_model,
    load_dataset,
    build_training_metadata,
    load_spectral_targets,
    local_loss_config,
    spectral_audit,
)

from scripts.training.train_local_supersonic_kappa_q_logamp import (
    compute_losses,
)


def resolve_path(value):
    return Path(value).expanduser().resolve()


def save_checkpoint(
    path,
    *,
    model,
    config,
    stage,
    step,
    metrics,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "config":
                config,
            "stage":
                stage,
            "step":
                int(step),
            "metrics":
                metrics,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--source-checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--modal-steps",
        type=int,
        default=13000,
    )

    parser.add_argument(
        "--joint-steps",
        type=int,
        default=13000,
    )

    parser.add_argument(
        "--modal-lr",
        type=float,
        default=1.0e-4,
    )

    parser.add_argument(
        "--joint-spectral-lr",
        type=float,
        default=2.0e-5,
    )

    parser.add_argument(
        "--joint-modal-lr",
        type=float,
        default=5.0e-5,
    )

    parser.add_argument(
        "--n-interior",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--n-boundary",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--print-every-cycles",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    config_path = resolve_path(
        args.config
    )

    source_path = resolve_path(
        args.source_checkpoint
    )

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = json.loads(
        config_path.read_text()
    )

    device = torch.device(
        args.device
    )

    frame = load_dataset(
        config
    )

    train_machs, alpha_bounds = (
        build_training_metadata(
            frame
        )
    )

    print(
        "train Mach:",
        " ".join(
            f"{m:.2f}"
            for m in train_machs
        ),
        flush=True,
    )

    print(
        "n train Mach:",
        len(train_machs),
        flush=True,
    )

    assert len(train_machs) == 13

    model = (
        build_model(config)
        .to(device)
    )

    source = torch.load(
        source_path,
        map_location=device,
    )

    model.load_state_dict(
        source["model_state_dict"]
    )

    spectral_targets = (
        load_spectral_targets(
            frame,
            train_machs,
            model,
            device,
        )
    )

    weights = config[
        "loss_weights"
    ]

    try:
        generator = torch.Generator(
            device=device
        )
    except TypeError:
        generator = torch.Generator(
            device=device.type
        )

    generator.manual_seed(
        int(
            config.get(
                "seed",
                12345,
            )
        )
        + 1001
    )

    n_mach = len(
        train_machs
    )

    history = []

    def prepare_mach(mach):
        model.set_mach_context(
            float(mach)
        )

        alpha_min, alpha_max = (
            alpha_bounds[
                float(mach)
            ]
        )

        return local_loss_config(
            config,
            mach=float(mach),
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

    def global_spectral_loss():
        losses = []

        for mach in train_machs:
            target = spectral_targets[
                mach
            ]

            mach_tensor = (
                torch.full_like(
                    target["alpha"],
                    float(mach),
                )
            )

            cr_pred, ci_pred = (
                model.get_spectrum(
                    target["alpha"],
                    mach_tensor,
                )
            )

            loss_cr = (
                (
                    cr_pred
                    - target["cr"]
                )
                / target["cr_scale"]
            ).square().mean()

            loss_ci = (
                (
                    ci_pred
                    - target["ci"]
                )
                / target["ci_scale"]
            ).square().mean()

            eps = 1.0e-6

            loss_log_ci = (
                torch.log(
                    ci_pred + eps
                )
                - torch.log(
                    target["ci"]
                    + eps
                )
            ).square().mean()

            losses.append(
                loss_cr
                + loss_ci
                + 0.25
                * loss_log_ci
            )

        return torch.stack(
            losses
        ).mean()

    def validation_metrics():
        predictions, metrics = (
            spectral_audit(
                model,
                frame,
                device=device,
                include_test=False,
            )
        )

        return (
            predictions,
            metrics,
        )

    # ============================================================
    # Initial spectral audit
    # ============================================================

    pred0, metrics0 = (
        validation_metrics()
    )

    print()
    print(
        "INITIAL VALIDATION SPECTRAL MEAN =",
        metrics0[
            "by_split"
        ][
            "validation"
        ][
            "spectral_error_mean"
        ],
        flush=True,
    )

    # ============================================================
    # STAGE 1: modal physics with frozen validated spectrum
    # ============================================================

    for p in model.spectral_parameters():
        p.requires_grad_(False)

    for p in model.modal_parameters():
        p.requires_grad_(True)

    modal_optimizer = (
        torch.optim.Adam(
            model.modal_parameters(),
            lr=args.modal_lr,
        )
    )

    best_modal_cycle = float(
        "inf"
    )

    cycle_losses = []

    print()
    print("=" * 80)
    print("MODAL FROZEN-SPECTRUM")
    print("=" * 80)

    for step in range(
        1,
        args.modal_steps + 1,
    ):
        mach = train_machs[
            (step - 1) % n_mach
        ]

        local_cfg = prepare_mach(
            mach
        )

        modal_optimizer.zero_grad(
            set_to_none=True
        )

        loss, values = compute_losses(
            model,
            local_cfg,
            spectral_targets=None,
            modal_targets=None,
            generator=generator,
            n_interior=
                args.n_interior,
            n_boundary=
                args.n_boundary,
            include_spectral=False,
            include_modal=False,
            device=device,
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite modal loss "
                f"at step {step}"
            )

        loss.backward()
        modal_optimizer.step()

        cycle_losses.append(
            float(
                values["loss"]
            )
        )

        history.append(
            {
                "stage":
                    "modal_frozen",
                "step":
                    step,
                "Mach":
                    mach,
                **values,
            }
        )

        if (
            step % n_mach == 0
        ):
            cycle = (
                step // n_mach
            )

            cycle_mean = float(
                np.mean(
                    cycle_losses
                )
            )

            cycle_losses = []

            if (
                cycle_mean
                < best_modal_cycle
            ):
                best_modal_cycle = (
                    cycle_mean
                )

                save_checkpoint(
                    output_dir
                    / "best_modal_cycle_checkpoint.pt",
                    model=model,
                    config=config,
                    stage=
                        "modal_frozen",
                    step=step,
                    metrics={
                        "cycle_mean":
                            cycle_mean
                    },
                )

            if (
                cycle == 1
                or cycle
                % args.print_every_cycles
                == 0
            ):
                print(
                    f"[modal] "
                    f"cycle={cycle:4d} "
                    f"step={step:6d} "
                    f"mean="
                    f"{cycle_mean:.6e} "
                    f"best="
                    f"{best_modal_cycle:.6e}",
                    flush=True,
                )

    # Reload best balanced modal checkpoint.
    modal_best = torch.load(
        output_dir
        / "best_modal_cycle_checkpoint.pt",
        map_location=device,
    )

    model.load_state_dict(
        modal_best[
            "model_state_dict"
        ]
    )

    # ============================================================
    # STAGE 2: joint
    # physics at one Mach + GLOBAL spectral loss on all 52 anchors
    # ============================================================

    for p in model.spectral_parameters():
        p.requires_grad_(True)

    for p in model.modal_parameters():
        p.requires_grad_(True)

    joint_optimizer = (
        torch.optim.Adam(
            [
                {
                    "params":
                        list(
                            model
                            .spectral_parameters()
                        ),
                    "lr":
                        args
                        .joint_spectral_lr,
                },
                {
                    "params":
                        list(
                            model
                            .modal_parameters()
                        ),
                    "lr":
                        args
                        .joint_modal_lr,
                },
            ]
        )
    )

    best_validation = float(
        "inf"
    )

    best_validation_step = None

    best_joint_cycle = float(
        "inf"
    )

    cycle_losses = []

    print()
    print("=" * 80)
    print(
        "JOINT: PHYSICS + "
        "GLOBAL 52-ANCHOR LOSS"
    )
    print("=" * 80)

    for step in range(
        1,
        args.joint_steps + 1,
    ):
        mach = train_machs[
            (step - 1) % n_mach
        ]

        local_cfg = prepare_mach(
            mach
        )

        joint_optimizer.zero_grad(
            set_to_none=True
        )

        physics_loss, values = (
            compute_losses(
                model,
                local_cfg,
                spectral_targets=None,
                modal_targets=None,
                generator=generator,
                n_interior=
                    args.n_interior,
                n_boundary=
                    args.n_boundary,
                include_spectral=False,
                include_modal=False,
                device=device,
            )
        )

        spec_loss = (
            global_spectral_loss()
        )

        total = (
            physics_loss
            + float(
                weights[
                    "spectral"
                ]
            )
            * spec_loss
        )

        if not torch.isfinite(
            total
        ):
            raise RuntimeError(
                "Non-finite joint loss "
                f"at step {step}"
            )

        total.backward()
        joint_optimizer.step()

        total_value = float(
            total.detach()
        )

        spec_value = float(
            spec_loss.detach()
        )

        cycle_losses.append(
            total_value
        )

        history.append(
            {
                "stage":
                    "joint",
                "step":
                    step,
                "Mach":
                    mach,
                "loss":
                    total_value,
                "spectral_global":
                    spec_value,
                "physics":
                    float(
                        physics_loss
                        .detach()
                    ),
                **{
                    k: v
                    for k, v
                    in values.items()
                    if k != "loss"
                },
            }
        )

        if (
            step % n_mach == 0
        ):
            cycle = (
                step // n_mach
            )

            cycle_mean = float(
                np.mean(
                    cycle_losses
                )
            )

            cycle_losses = []

            predictions, metrics = (
                validation_metrics()
            )

            val = float(
                metrics[
                    "by_split"
                ][
                    "validation"
                ][
                    "spectral_error_mean"
                ]
            )

            train_spec = float(
                metrics[
                    "by_split"
                ][
                    "train"
                ][
                    "spectral_error_mean"
                ]
            )

            if (
                cycle_mean
                < best_joint_cycle
            ):
                best_joint_cycle = (
                    cycle_mean
                )

                save_checkpoint(
                    output_dir
                    / "best_joint_cycle_checkpoint.pt",
                    model=model,
                    config=config,
                    stage=
                        "joint_cycle",
                    step=step,
                    metrics={
                        "cycle_mean":
                            cycle_mean,
                        "validation_spectral_mean":
                            val,
                        "train_spectral_mean":
                            train_spec,
                    },
                )

            if (
                val
                < best_validation
            ):
                best_validation = (
                    val
                )

                best_validation_step = (
                    step
                )

                save_checkpoint(
                    output_dir
                    / "best_validation_checkpoint.pt",
                    model=model,
                    config=config,
                    stage=
                        "joint_validation",
                    step=step,
                    metrics={
                        "cycle_mean":
                            cycle_mean,
                        "validation_spectral_mean":
                            val,
                        "train_spectral_mean":
                            train_spec,
                        "global_anchor_loss":
                            spec_value,
                    },
                )

            if (
                cycle == 1
                or cycle
                % args.print_every_cycles
                == 0
            ):
                print(
                    f"[joint] "
                    f"cycle={cycle:4d} "
                    f"step={step:6d} "
                    f"total="
                    f"{cycle_mean:.6e} "
                    f"spec52="
                    f"{spec_value:.6e} "
                    f"train_spec="
                    f"{train_spec:.6e} "
                    f"val_spec="
                    f"{val:.6e} "
                    f"best_val="
                    f"{best_validation:.6e}",
                    flush=True,
                )

    # ============================================================
    # Final audit of best-validation model
    # ============================================================

    best = torch.load(
        output_dir
        / "best_validation_checkpoint.pt",
        map_location=device,
    )

    model.load_state_dict(
        best[
            "model_state_dict"
        ]
    )

    predictions, metrics = (
        validation_metrics()
    )

    predictions.to_csv(
        output_dir
        / "spectral_audit_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        history
    ).to_csv(
        output_dir
        / "training_history.csv",
        index=False,
    )

    (
        output_dir
        / "spectral_metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        )
        + "\n"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "config":
                config,
            "source_checkpoint":
                str(source_path),
            "best_validation_step":
                best_validation_step,
            "best_validation":
                best_validation,
            "metrics":
                metrics,
        },
        output_dir
        / "final_checkpoint.pt",
    )

    print()
    print("=" * 100)
    print("G52-v2 COMPLETE")
    print("=" * 100)

    print(
        "best validation step =",
        best_validation_step,
    )

    print(
        "best validation      =",
        f"{best_validation:.8e}",
    )

    print()
    print("BY SPLIT")

    for split in [
        "train",
        "validation",
    ]:
        print()
        print(split.upper())

        print(
            json.dumps(
                metrics[
                    "by_split"
                ][split],
                indent=2,
            )
        )

    print()
    print(
        "VALIDATION BY MACH"
    )

    for mach in [
        "1.15",
        "1.45",
        "1.75",
    ]:
        print()
        print(
            f"M={mach}"
        )

        print(
            json.dumps(
                metrics[
                    "by_mach"
                ][mach],
                indent=2,
            )
        )

    print()
    print(
        "TEST SPLIT REMAINS SEALED."
    )


if __name__ == "__main__":
    main()
