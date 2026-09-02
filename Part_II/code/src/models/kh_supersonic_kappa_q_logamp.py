from __future__ import annotations

import math

import torch
from torch import nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    *,
    width: int,
    depth: int,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be >= 1")

    layers: list[nn.Module] = []

    current_dim = input_dim

    for _ in range(depth):
        linear = nn.Linear(
            current_dim,
            width,
        )

        nn.init.xavier_normal_(
            linear.weight,
            gain=nn.init.calculate_gain("tanh"),
        )

        nn.init.zeros_(linear.bias)

        layers.extend(
            [
                linear,
                nn.Tanh(),
            ]
        )

        current_dim = width

    output = nn.Linear(
        current_dim,
        output_dim,
    )

    nn.init.xavier_normal_(
        output.weight,
        gain=0.25,
    )

    nn.init.zeros_(output.bias)

    layers.append(output)

    return nn.Sequential(*layers)


class SpectralNet(nn.Module):
    """
    alpha -> (cr, ci)

    cr is bounded through tanh.
    ci is positive and bounded through sigmoid.
    """

    def __init__(
        self,
        *,
        alpha_min: float,
        alpha_max: float,
        width: int = 96,
        depth: int = 3,
        cr_min: float = -0.10,
        cr_max: float = 0.30,
        ci_floor: float = 1.0e-6,
        ci_max: float = 0.20,
    ) -> None:
        super().__init__()

        if alpha_max <= alpha_min:
            raise ValueError(
                "alpha_max must exceed alpha_min"
            )

        if cr_max <= cr_min:
            raise ValueError(
                "cr_max must exceed cr_min"
            )

        if ci_max <= 0.0:
            raise ValueError(
                "ci_max must be positive"
            )

        self.register_buffer(
            "alpha_min",
            torch.tensor(float(alpha_min)),
        )

        self.register_buffer(
            "alpha_max",
            torch.tensor(float(alpha_max)),
        )

        self.register_buffer(
            "cr_mid",
            torch.tensor(
                0.5 * (cr_min + cr_max)
            ),
        )

        self.register_buffer(
            "cr_half_range",
            torch.tensor(
                0.5 * (cr_max - cr_min)
            ),
        )

        self.register_buffer(
            "ci_floor",
            torch.tensor(float(ci_floor)),
        )

        self.register_buffer(
            "ci_max",
            torch.tensor(float(ci_max)),
        )

        # alpha_norm, alpha_norm^2, alpha_norm^3
        self.net = build_mlp(
            3,
            2,
            width=width,
            depth=depth,
        )

    def normalize_alpha(
        self,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        return (
            2.0
            * (alpha - self.alpha_min)
            / (self.alpha_max - self.alpha_min)
            - 1.0
        )

    def forward(
        self,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_norm = self.normalize_alpha(
            alpha
        )

        features = torch.cat(
            [
                alpha_norm,
                alpha_norm.square(),
                alpha_norm.pow(3),
            ],
            dim=-1,
        )

        raw = self.net(features)

        cr = (
            self.cr_mid
            + self.cr_half_range
            * torch.tanh(raw[:, 0:1])
        )

        ci = (
            self.ci_floor
            + self.ci_max
            * torch.sigmoid(raw[:, 1:2])
        )

        return cr, ci


class SpatialFourierEncoding(nn.Module):
    def __init__(
        self,
        *,
        n_frequencies: int,
    ) -> None:
        super().__init__()

        frequencies = (
            math.pi
            * torch.arange(
                1,
                n_frequencies + 1,
                dtype=torch.get_default_dtype(),
            )
        )

        self.register_buffer(
            "frequencies",
            frequencies.view(1, -1),
        )

    @property
    def output_dim(self) -> int:
        return 2 * int(
            self.frequencies.shape[1]
        )

    def forward(
        self,
        xi_normalized: torch.Tensor,
    ) -> torch.Tensor:
        angles = (
            xi_normalized
            * self.frequencies
        )

        return torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )


class ModalNet(nn.Module):
    """
    (xi, alpha) -> (kappa, q, raw_log_amp)

    The log-amplitude gauge is imposed later by subtracting
    raw_log_amp(xi=0, alpha).
    """

    def __init__(
        self,
        *,
        alpha_min: float,
        alpha_max: float,
        xi_max: float,
        width: int = 256,
        depth: int = 7,
        n_frequencies: int = 12,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "alpha_min",
            torch.tensor(float(alpha_min)),
        )

        self.register_buffer(
            "alpha_max",
            torch.tensor(float(alpha_max)),
        )

        self.register_buffer(
            "xi_max",
            torch.tensor(float(xi_max)),
        )

        self.encoding = SpatialFourierEncoding(
            n_frequencies=n_frequencies,
        )

        # xi_norm, alpha_norm, Fourier(xi)
        input_dim = (
            2
            + self.encoding.output_dim
        )

        self.net = build_mlp(
            input_dim,
            3,
            width=width,
            depth=depth,
        )

    def normalize_alpha(
        self,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        return (
            2.0
            * (alpha - self.alpha_min)
            / (self.alpha_max - self.alpha_min)
            - 1.0
        )

    def encode(
        self,
        xi: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        xi_normalized = xi / self.xi_max

        alpha_normalized = (
            self.normalize_alpha(alpha)
        )

        return torch.cat(
            [
                xi_normalized,
                alpha_normalized,
                self.encoding(xi_normalized),
            ],
            dim=-1,
        )

    def forward_raw(
        self,
        xi: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            self.encode(
                xi,
                alpha,
            )
        )


class KHSupersonicLocalPINN(nn.Module):
    """
    Fixed-Mach local PINN.

    Spectral outputs:
        cr(alpha), ci(alpha)

    Modal outputs:
        kappa(xi, alpha)
        q(xi, alpha) = Im(p_y / p)
        log_amp(xi, alpha)

    The phase is not predicted.
    """

    def __init__(
        self,
        *,
        mach: float,
        alpha_min: float,
        alpha_max: float,
        xi_max: float = 0.985,
        mapping_scale: float = 3.0,
        spectral_width: int = 96,
        spectral_depth: int = 3,
        modal_width: int = 256,
        modal_depth: int = 7,
        n_frequencies: int = 12,
        mode_experts: int = 2,
        alpha_split: float = 0.20,
        alpha_gate_width: float = 0.02,
        cr_min: float = -0.10,
        cr_max: float = 0.30,
        ci_floor: float = 1.0e-6,
        ci_max: float = 0.20,
    ) -> None:
        super().__init__()

        if mode_experts not in {1, 2}:
            raise ValueError(
                "mode_experts must be 1 or 2"
            )

        self.mach = float(mach)
        self.mode_experts = int(mode_experts)

        self.register_buffer(
            "mapping_scale",
            torch.tensor(
                float(mapping_scale)
            ),
        )

        self.register_buffer(
            "xi_max",
            torch.tensor(float(xi_max)),
        )

        self.register_buffer(
            "alpha_split",
            torch.tensor(float(alpha_split)),
        )

        self.register_buffer(
            "alpha_gate_width",
            torch.tensor(
                float(alpha_gate_width)
            ),
        )

        self.spectral_net = SpectralNet(
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            width=spectral_width,
            depth=spectral_depth,
            cr_min=cr_min,
            cr_max=cr_max,
            ci_floor=ci_floor,
            ci_max=ci_max,
        )

        self.modal_low = ModalNet(
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            xi_max=xi_max,
            width=modal_width,
            depth=modal_depth,
            n_frequencies=n_frequencies,
        )

        if self.mode_experts == 2:
            self.modal_high = ModalNet(
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                xi_max=xi_max,
                width=modal_width,
                depth=modal_depth,
                n_frequencies=n_frequencies,
            )
        else:
            self.modal_high = None

    def get_spectrum(
        self,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.spectral_net(alpha)

    def get_cr(
        self,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        return self.get_spectrum(alpha)[0]

    def get_ci(
        self,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        return self.get_spectrum(alpha)[1]

    def get_mapping_scale(
        self,
    ) -> torch.Tensor:
        return self.mapping_scale

    def modal_raw(
        self,
        xi: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        low = self.modal_low.forward_raw(
            xi,
            alpha,
        )

        if self.modal_high is None:
            return low

        high = self.modal_high.forward_raw(
            xi,
            alpha,
        )

        gate = torch.sigmoid(
            (
                alpha
                - self.alpha_split
            )
            / self.alpha_gate_width
        )

        return (
            (1.0 - gate) * low
            + gate * high
        )

    def forward(
        self,
        xi: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.modal_raw(
            xi,
            alpha,
        )

        xi_center = torch.zeros_like(xi)

        raw_center = self.modal_raw(
            xi_center,
            alpha,
        )

        kappa = raw[:, 0:1]
        phase_gradient = raw[:, 1:2]

        # Hard amplitude gauge:
        # log|p|(y=0, alpha) = 0.
        log_amp = (
            raw[:, 2:3]
            - raw_center[:, 2:3]
        )

        return torch.cat(
            [
                kappa,
                phase_gradient,
                log_amp,
            ],
            dim=-1,
        )

    def spectral_parameters(
        self,
    ):
        return self.spectral_net.parameters()

    def modal_parameters(
        self,
    ):
        modules = [self.modal_low]

        if self.modal_high is not None:
            modules.append(self.modal_high)

        for module in modules:
            yield from module.parameters()
