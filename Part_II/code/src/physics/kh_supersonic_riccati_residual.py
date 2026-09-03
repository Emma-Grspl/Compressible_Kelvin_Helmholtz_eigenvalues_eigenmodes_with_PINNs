from __future__ import annotations

import torch


def xi_to_y(
    xi: torch.Tensor,
    mapping_scale: torch.Tensor | float,
) -> torch.Tensor:
    scale = torch.as_tensor(
        mapping_scale,
        dtype=xi.dtype,
        device=xi.device,
    )

    return (
        scale
        * xi
        / (1.0 - xi.square())
    )


def dy_dxi(
    xi: torch.Tensor,
    mapping_scale: torch.Tensor | float,
) -> torch.Tensor:
    scale = torch.as_tensor(
        mapping_scale,
        dtype=xi.dtype,
        device=xi.device,
    )

    return (
        scale
        * (1.0 + xi.square())
        / (1.0 - xi.square()).square()
    )


def y_to_xi(
    y: torch.Tensor,
    mapping_scale: torch.Tensor | float,
) -> torch.Tensor:
    scale = torch.as_tensor(
        mapping_scale,
        dtype=y.dtype,
        device=y.device,
    )

    # Stable inverse of:
    # y = scale * xi / (1 - xi^2)
    return (
        2.0 * y
        / (
            scale
            + torch.sqrt(
                scale.square()
                + 4.0 * y.square()
            )
        )
    )


def base_velocity(
    y: torch.Tensor,
) -> torch.Tensor:
    return torch.tanh(y)


def base_velocity_derivative(
    y: torch.Tensor,
) -> torch.Tensor:
    # Stable sech^2(y).
    exp_term = torch.exp(
        -2.0 * torch.abs(y)
    )

    return (
        4.0
        * exp_term
        / (1.0 + exp_term).square()
    )


def differentiate(
    values: torch.Tensor,
    coordinate: torch.Tensor,
) -> torch.Tensor:
    gradient = torch.autograd.grad(
        values,
        coordinate,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]

    return gradient


def principal_sqrt_positive_real(
    value: torch.Tensor,
) -> torch.Tensor:
    root = torch.sqrt(value)

    flip = (
        (root.real < 0.0)
        |
        (
            root.real.abs() < 1.0e-12
        )
        & (root.imag < 0.0)
    )

    return torch.where(
        flip,
        -root,
        root,
    )


def asymptotic_riccati_gammas(
    alpha: torch.Tensor,
    mach: float,
    cr: torch.Tensor,
    ci: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mach_tensor = torch.as_tensor(
        float(mach),
        dtype=alpha.dtype,
        device=alpha.device,
    )

    alpha_complex = torch.complex(
        alpha,
        torch.zeros_like(alpha),
    )

    c = torch.complex(
        cr,
        ci,
    )

    one = torch.ones_like(alpha)

    left_difference = torch.complex(
        -one,
        torch.zeros_like(one),
    ) - c

    right_difference = torch.complex(
        one,
        torch.zeros_like(one),
    ) - c

    r_left = (
        torch.ones_like(left_difference)
        - mach_tensor.square()
        * left_difference.square()
    )

    r_right = (
        torch.ones_like(right_difference)
        - mach_tensor.square()
        * right_difference.square()
    )

    root_left = principal_sqrt_positive_real(
        r_left
    )

    root_right = principal_sqrt_positive_real(
        r_right
    )

    gamma_left = (
        alpha_complex
        * root_left
    )

    gamma_right = (
        -alpha_complex
        * root_right
    )

    return gamma_left, gamma_right


def riccati_regularized_residuals(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    *,
    mach: float,
) -> dict[str, torch.Tensor]:
    if not xi.requires_grad:
        xi = xi.clone().detach().requires_grad_(
            True
        )

    prediction = model(
        xi,
        alpha,
    )

    kappa = prediction[:, 0:1]
    phase_gradient = prediction[:, 1:2]
    log_amp = prediction[:, 2:3]

    kappa_xi = differentiate(
        kappa,
        xi,
    )

    phase_gradient_xi = differentiate(
        phase_gradient,
        xi,
    )

    log_amp_xi = differentiate(
        log_amp,
        xi,
    )

    mapping_scale = (
        model.get_mapping_scale()
    )

    y = xi_to_y(
        xi,
        mapping_scale,
    )

    jacobian = dy_dxi(
        xi,
        mapping_scale,
    )

    kappa_y = (
        kappa_xi
        / jacobian
    )

    phase_gradient_y = (
        phase_gradient_xi
        / jacobian
    )

    log_amp_y = (
        log_amp_xi
        / jacobian
    )

    velocity = base_velocity(y)

    velocity_y = (
        base_velocity_derivative(y)
    )

    cr, ci = model.get_spectrum(alpha)

    d = velocity - cr

    mach_tensor = torch.as_tensor(
        float(mach),
        dtype=alpha.dtype,
        device=alpha.device,
    )

    # U-c = d - i ci
    u_minus_c = torch.complex(
        d,
        -ci,
    )

    r_complex = (
        torch.ones_like(u_minus_c)
        - mach_tensor.square()
        * u_minus_c.square()
    )

    r_real = r_complex.real
    r_imag = r_complex.imag

    t_real = (
        kappa_y
        + kappa.square()
        - phase_gradient.square()
        - alpha.square() * r_real
    )

    t_imag = (
        phase_gradient_y
        + 2.0
        * kappa
        * phase_gradient
        - alpha.square() * r_imag
    )

    # Regularized Riccati equation:
    #
    # (U-c) [gamma_y + gamma^2 - alpha^2 R]
    # - 2 U_y gamma = 0.
    residual_kappa = (
        d * t_real
        + ci * t_imag
        - 2.0
        * velocity_y
        * kappa
    )

    residual_phase_gradient = (
        d * t_imag
        - ci * t_real
        - 2.0
        * velocity_y
        * phase_gradient
    )

    residual_log_amp = (
        log_amp_y
        - kappa
    )

    riccati_scale = (
        1.0
        + torch.abs(d)
        * torch.sqrt(
            t_real.square()
            + t_imag.square()
            + 1.0e-14
        )
        + 2.0
        * torch.abs(velocity_y)
        * torch.sqrt(
            kappa.square()
            + phase_gradient.square()
            + 1.0e-14
        )
    ).detach()

    amplitude_scale = (
        1.0
        + torch.abs(kappa)
    ).detach()

    return {
        "residual_kappa": residual_kappa,
        "residual_phase_gradient": (
            residual_phase_gradient
        ),
        "residual_log_amp": (
            residual_log_amp
        ),
        "loss_kappa": (
            residual_kappa
            / riccati_scale
        ).square().mean(),
        "loss_phase_gradient": (
            residual_phase_gradient
            / riccati_scale
        ).square().mean(),
        "loss_log_amp": (
            residual_log_amp
            / amplitude_scale
        ).square().mean(),
        "cr": cr,
        "ci": ci,
        "y": y,
        "kappa": kappa,
        "phase_gradient": phase_gradient,
        "log_amp": log_amp,
    }


def riccati_boundary_losses(
    model,
    alpha: torch.Tensor,
    *,
    mach: float,
    xi_boundary: float,
) -> dict[str, torch.Tensor]:
    xi_left = torch.full_like(
        alpha,
        -float(xi_boundary),
    )

    xi_right = torch.full_like(
        alpha,
        float(xi_boundary),
    )

    prediction_left = model(
        xi_left,
        alpha,
    )

    prediction_right = model(
        xi_right,
        alpha,
    )

    cr, ci = model.get_spectrum(alpha)

    gamma_left, gamma_right = (
        asymptotic_riccati_gammas(
            alpha,
            mach,
            cr,
            ci,
        )
    )

    loss_kappa = (
        (
            prediction_left[:, 0:1]
            - gamma_left.real
        )
        .square()
        .mean()
        +
        (
            prediction_right[:, 0:1]
            - gamma_right.real
        )
        .square()
        .mean()
    )

    loss_phase_gradient = (
        (
            prediction_left[:, 1:2]
            - gamma_left.imag
        )
        .square()
        .mean()
        +
        (
            prediction_right[:, 1:2]
            - gamma_right.imag
        )
        .square()
        .mean()
    )

    return {
        "loss_bc_kappa": loss_kappa,
        "loss_bc_phase_gradient": (
            loss_phase_gradient
        ),
        "gamma_left": gamma_left,
        "gamma_right": gamma_right,
    }


def spectral_smoothness_loss(
    model,
    alpha: torch.Tensor,
) -> torch.Tensor:
    if not alpha.requires_grad:
        alpha = (
            alpha.clone()
            .detach()
            .requires_grad_(True)
        )

    cr, ci = model.get_spectrum(alpha)

    cr_alpha = differentiate(
        cr,
        alpha,
    )

    ci_alpha = differentiate(
        ci,
        alpha,
    )

    cr_alpha2 = differentiate(
        cr_alpha,
        alpha,
    )

    ci_alpha2 = differentiate(
        ci_alpha,
        alpha,
    )

    return (
        cr_alpha2.square().mean()
        + ci_alpha2.square().mean()
    )
