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
        linear = nn.Linear(current_dim, width)

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

    output = nn.Linear(current_dim, output_dim)

    nn.init.xavier_normal_(
        output.weight,
        gain=0.25,
    )
    nn.init.zeros_(output.bias)

    layers.append(output)

    return nn.Sequential(*layers)


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


class SpectralNet2D(nn.Module):
    """
    (alpha, Mach) -> (cr, ci)

    No neutral-boundary coordinate is used here:
    held-out Mach values therefore require no
    classical alpha_neutral information.
    """

    def __init__(
        self,
        *,
        alpha_min: float,
        alpha_max: float,
        mach_min: float,
        mach_max: float,
        width: int = 96,
        depth: int = 3,
        cr_min: float = -0.05,
        cr_max: float = 0.60,
        ci_floor: float = 1.0e-6,
        ci_max: float = 0.20,
    ) -> None:
        super().__init__()

        if alpha_max <= alpha_min:
            raise ValueError(
                "alpha_max must exceed alpha_min"
            )

        if mach_max <= mach_min:
            raise ValueError(
                "mach_max must exceed mach_min"
            )

        if cr_max <= cr_min:
            raise ValueError(
                "cr_max must exceed cr_min"
            )

        if ci_max <= 0.0:
            raise ValueError(
                "ci_max must be positive"
            )

        for name, value in [
            ("alpha_min", alpha_min),
            ("alpha_max", alpha_max),
            ("mach_min", mach_min),
            ("mach_max", mach_max),
        ]:
            self.register_buffer(
                name,
                torch.tensor(float(value)),
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

        # a, a^2, a^3, m, m^2, a*m
        self.net = build_mlp(
            6,
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

    def normalize_mach(
        self,
        mach: torch.Tensor,
    ) -> torch.Tensor:
        return (
            2.0
            * (mach - self.mach_min)
            / (self.mach_max - self.mach_min)
            - 1.0
        )

    def forward(
        self,
        alpha: torch.Tensor,
        mach: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.normalize_alpha(alpha)
        m = self.normalize_mach(mach)

        features = torch.cat(
            [
                a,
                a.square(),
                a.pow(3),
                m,
                m.square(),
                a * m,
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


class ModalNet2D(nn.Module):
    """
    (xi, alpha, Mach)
        -> (kappa, q, raw_log_amp)
    """

    def __init__(
        self,
        *,
        alpha_min: float,
        alpha_max: float,
        mach_min: float,
        mach_max: float,
        xi_max: float,
        width: int = 256,
        depth: int = 7,
        n_frequencies: int = 12,
    ) -> None:
        super().__init__()

        for name, value in [
            ("alpha_min", alpha_min),
            ("alpha_max", alpha_max),
            ("mach_min", mach_min),
            ("mach_max", mach_max),
            ("xi_max", xi_max),
        ]:
            self.register_buffer(
                name,
                torch.tensor(float(value)),
            )

        self.encoding = SpatialFourierEncoding(
            n_frequencies=n_frequencies,
        )

        # xi, alpha, Mach, alpha*Mach, Fourier(xi)
        input_dim = (
            4
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

    def normalize_mach(
        self,
        mach: torch.Tensor,
    ) -> torch.Tensor:
        return (
            2.0
            * (mach - self.mach_min)
            / (self.mach_max - self.mach_min)
            - 1.0
        )

    def encode(
        self,
        xi: torch.Tensor,
        alpha: torch.Tensor,
        mach: torch.Tensor,
    ) -> torch.Tensor:
        x = xi / self.xi_max
        a = self.normalize_alpha(alpha)
        m = self.normalize_mach(mach)

        return torch.cat(
            [
                x,
                a,
                m,
                a * m,
                self.encoding(x),
            ],
            dim=-1,
        )

    def forward_raw(
        self,
        xi: torch.Tensor,
        alpha: torch.Tensor,
        mach: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            self.encode(
                xi,
                alpha,
                mach,
            )
        )


class KHSupersonicGlobalPINN(nn.Module):
    """
    One shared PINN over (Mach, alpha).

    Spectrum:
        cr(alpha, Mach)
        ci(alpha, Mach)

    Modes:
        kappa(xi, alpha, Mach)
        q(xi, alpha, Mach)
        log_amp(xi, alpha, Mach)
    """

    def __init__(
        self,
        *,
        alpha_min: float = 0.05,
        alpha_max: float = 0.36,
        mach_min: float = 1.00,
        mach_max: float = 1.90,
        xi_max: float = 0.985,
        mapping_scale: float = 3.0,
        spectral_width: int = 96,
        spectral_depth: int = 3,
        modal_width: int = 256,
        modal_depth: int = 7,
        n_frequencies: int = 12,
        mode_experts: int = 2,
        cr_min: float = -0.05,
        cr_max: float = 0.60,
        ci_floor: float = 1.0e-6,
        ci_max: float = 0.20,
    ) -> None:
        super().__init__()

        if mode_experts not in {1, 2}:
            raise ValueError(
                "mode_experts must be 1 or 2"
            )

        self.mode_experts = int(
            mode_experts
        )

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
            "mach_context",
            torch.tensor(float("nan")),
        )

        self.spectral_net = SpectralNet2D(
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            mach_min=mach_min,
            mach_max=mach_max,
            width=spectral_width,
            depth=spectral_depth,
            cr_min=cr_min,
            cr_max=cr_max,
            ci_floor=ci_floor,
            ci_max=ci_max,
        )

        modal_kwargs = dict(
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            mach_min=mach_min,
            mach_max=mach_max,
            xi_max=xi_max,
            width=modal_width,
            depth=modal_depth,
            n_frequencies=n_frequencies,
        )

        self.modal_low = ModalNet2D(
            **modal_kwargs
        )

        if self.mode_experts == 2:
            self.modal_high = ModalNet2D(
                **modal_kwargs
            )

            # learned smooth routing in (alpha, Mach)
            self.gate_net = build_mlp(
                2,
                1,
                width=32,
                depth=2,
            )
        else:
            self.modal_high = None
            self.gate_net = None

    def set_mach_context(
        self,
        mach: float,
    ) -> None:
        self.mach_context.fill_(
            float(mach)
        )

    def _resolve_mach(
        self,
        alpha: torch.Tensor,
        mach: torch.Tensor | None,
    ) -> torch.Tensor:
        if mach is not None:
            return mach

        if not torch.isfinite(
            self.mach_context
        ):
            raise RuntimeError(
                "Mach context is not set."
            )

        return torch.full_like(
            alpha,
            self.mach_context.item(),
        )

    def get_spectrum(
        self,
        alpha: torch.Tensor,
        mach: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mach = self._resolve_mach(
            alpha,
            mach,
        )

        return self.spectral_net(
            alpha,
            mach,
        )

    def get_cr(
        self,
        alpha: torch.Tensor,
        mach: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.get_spectrum(
            alpha,
            mach,
        )[0]

    def get_ci(
        self,
        alpha: torch.Tensor,
        mach: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.get_spectrum(
            alpha,
            mach,
        )[1]

    def get_mapping_scale(
        self,
    ) -> torch.Tensor:
        return self.mapping_scale

    def modal_raw(
        self,
        xi: torch.Tensor,
        alpha: torch.Tensor,
        mach: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mach = self._resolve_mach(
            alpha,
            mach,
        )

        low = self.modal_low.forward_raw(
            xi,
            alpha,
            mach,
        )

        if self.modal_high is None:
            return low

        high = self.modal_high.forward_raw(
            xi,
            alpha,
            mach,
        )

        a = (
            self.spectral_net
            .normalize_alpha(alpha)
        )
        m = (
            self.spectral_net
            .normalize_mach(mach)
        )

        gate = torch.sigmoid(
            self.gate_net(
                torch.cat(
                    [a, m],
                    dim=-1,
                )
            )
        )

        return (
            (1.0 - gate) * low
            + gate * high
        )

    def forward(
        self,
        xi: torch.Tensor,
        alpha: torch.Tensor,
        mach: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mach = self._resolve_mach(
            alpha,
            mach,
        )

        raw = self.modal_raw(
            xi,
            alpha,
            mach,
        )

        raw_center = self.modal_raw(
            torch.zeros_like(xi),
            alpha,
            mach,
        )

        kappa = raw[:, 0:1]
        q = raw[:, 1:2]

        log_amp = (
            raw[:, 2:3]
            - raw_center[:, 2:3]
        )

        return torch.cat(
            [
                kappa,
                q,
                log_amp,
            ],
            dim=-1,
        )

    def spectral_parameters(self):
        return self.spectral_net.parameters()

    def modal_parameters(self):
        modules = [
            self.modal_low,
        ]

        if self.modal_high is not None:
            modules.extend(
                [
                    self.modal_high,
                    self.gate_net,
                ]
            )

        for module in modules:
            yield from module.parameters()
