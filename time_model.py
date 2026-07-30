import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical


class BernsteinBetaTimeModel(nn.Module):
    """Continuous density on a biological waiting-time CDF coordinate.

    Component ``k`` of an ``M``-component basis is
    ``Beta(k + 1, M - k)``. Equal mixture weights therefore sum exactly to the
    uniform density, which is the constant-Ne CWR prior in conditional-CDF
    coordinates.
    """

    CONTEXT_FEATURE_DIM = 4

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout,
        basis_components,
        layers=3,
    ):
        super().__init__()
        layers = int(layers)
        basis_components = int(basis_components)
        if layers < 0:
            raise ValueError(f"layers must be non-negative, got {layers}")
        if basis_components < 2:
            raise ValueError("time basis must contain at least two components")

        self.basis_components = basis_components
        if layers > 0:
            modules = [
                nn.Linear(input_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            ]
            for _ in range(layers - 1):
                modules.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Dropout(dropout),
                        nn.ReLU(),
                    ]
                )
            self.feature = nn.Sequential(*modules)
            self.output_layer = nn.Linear(hidden_dim, basis_components)
        else:
            self.feature = None
            self.output_layer = nn.Linear(input_dim, basis_components)

        alpha = torch.arange(1, basis_components + 1, dtype=torch.float32)
        beta = torch.arange(
            basis_components,
            0,
            -1,
            dtype=torch.float32,
        )
        # These are fixed analytical basis constants, not learned checkpoint
        # state. Float32 storage keeps the module movable to Apple MPS; density
        # evaluation promotes them to float64 below.
        self.register_buffer("basis_alpha", alpha, persistent=False)
        self.register_buffer("basis_beta", beta, persistent=False)
        self.apply(self._init_weights)
        # Start at the exact uniform CWR prior instead of an arbitrary tilt.
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def context_features(rates, max_deltas, *, device, dtype):
        """Create stable direct features for the waiting-time policy."""

        rows = []
        for rate, max_delta in zip(rates, max_deltas):
            rate = float(rate)
            if not math.isfinite(rate) or rate <= 0.0:
                raise ValueError(
                    "continuous generated transition requires a positive rate"
                )
            raw_log_rate = math.log(rate)
            log_rate = min(max(raw_log_rate, -30.0), 30.0)
            if max_delta is None:
                rows.append((log_rate, 0.0, 0.0, 0.0))
                continue
            max_delta = float(max_delta)
            if not math.isfinite(max_delta) or max_delta <= 0.0:
                raise ValueError(
                    "continuous generated transition requires a positive horizon"
                )
            raw_log_horizon = math.log(max_delta)
            log_horizon = min(max(raw_log_horizon, -30.0), 30.0)
            log_rate_horizon = min(
                max(raw_log_rate + raw_log_horizon, -30.0),
                30.0,
            )
            rows.append(
                (log_rate, 1.0, log_horizon, log_rate_horizon)
            )
        return torch.as_tensor(rows, dtype=dtype, device=device)

    @staticmethod
    def _temperature(random_spec):
        if random_spec is None:
            return 1.0
        temperature = float(random_spec.get("T", 1.0))
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("time-policy temperature must be positive")
        return temperature

    def _tempered_logits(self, mixture_logits, random_spec=None):
        return mixture_logits / self._temperature(random_spec)

    def sample(self, mixture_logits, random_spec=None):
        tempered = self._tempered_logits(mixture_logits, random_spec)
        component = Categorical(logits=tempered).sample()
        sample_device = mixture_logits.device
        if sample_device.type == "mps":
            # MPS has no float64 distributions. Samples do not require a
            # gradient, so draw the exact fixed-basis distribution on CPU.
            component = component.detach().cpu().to(torch.float64)
            sample_device = torch.device("cpu")
            alpha = component + 1.0
            beta = float(self.basis_components) - component
        else:
            alpha = self.basis_alpha.index_select(
                0,
                component,
            ).to(device=sample_device, dtype=torch.float64)
            beta = self.basis_beta.index_select(
                0,
                component,
            ).to(device=sample_device, dtype=torch.float64)
        quantiles = Beta(alpha, beta).sample()
        epsilon = torch.finfo(quantiles.dtype).eps
        return quantiles.clamp(min=epsilon, max=1.0 - epsilon)

    def _log_quantile_density_float64(
        self,
        mixture_logits,
        quantiles,
        random_spec=None,
    ):
        tempered = self._tempered_logits(mixture_logits, random_spec)
        density_device = mixture_logits.device
        if density_device.type == "mps":
            # Keep exact density arithmetic in float64 without relying on
            # unsupported MPS float64 tensors.
            density_device = torch.device("cpu")
            density_logits = tempered.detach().cpu().to(torch.float64)
            alpha_values = self.basis_alpha.cpu().to(torch.float64)
            beta_values = self.basis_beta.cpu().to(torch.float64)
        else:
            density_logits = tempered.to(torch.float64)
            alpha_values = self.basis_alpha.to(torch.float64)
            beta_values = self.basis_beta.to(torch.float64)
        log_weights = F.log_softmax(density_logits, dim=-1)
        quantiles = torch.as_tensor(
            quantiles,
            dtype=torch.float64,
            device=density_device,
        )
        epsilon = torch.finfo(quantiles.dtype).eps
        quantiles = quantiles.clamp(min=epsilon, max=1.0 - epsilon)
        u = quantiles[:, None]
        alpha = alpha_values[None, :]
        beta = beta_values[None, :]
        log_basis = (
            (alpha - 1.0) * torch.log(u)
            + (beta - 1.0) * torch.log1p(-u)
            + torch.lgamma(alpha + beta)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
        )
        return torch.logsumexp(log_weights + log_basis, dim=-1)

    def _log_quantile_density_native(
        self,
        mixture_logits,
        quantiles,
        random_spec=None,
    ):
        """Differentiable native-device density used by the MPS surrogate."""

        tempered = self._tempered_logits(mixture_logits, random_spec)
        log_weights = F.log_softmax(tempered, dim=-1)
        quantiles = torch.as_tensor(
            quantiles,
            dtype=mixture_logits.dtype,
            device=mixture_logits.device,
        )
        epsilon = torch.finfo(quantiles.dtype).eps
        u = quantiles.clamp(
            min=epsilon,
            max=1.0 - epsilon,
        )[:, None]
        alpha = self.basis_alpha[None, :].to(mixture_logits.dtype)
        beta = self.basis_beta[None, :].to(mixture_logits.dtype)
        log_basis = (
            (alpha - 1.0) * torch.log(u)
            + (beta - 1.0) * torch.log1p(-u)
            + torch.lgamma(alpha + beta)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
        )
        return torch.logsumexp(log_weights + log_basis, dim=-1)

    def log_quantile_density(
        self,
        mixture_logits,
        quantiles,
        random_spec=None,
    ):
        log_density = self._log_quantile_density_float64(
            mixture_logits,
            quantiles,
            random_spec=random_spec,
        )
        if mixture_logits.device.type == "mps":
            exact_value = log_density.to(
                device=mixture_logits.device,
                dtype=mixture_logits.dtype,
            )
            native_density = self._log_quantile_density_native(
                mixture_logits,
                quantiles,
                random_spec=random_spec,
            )
            # Forward values come from float64 CPU arithmetic. The analytically
            # identical float32 expression supplies gradients on MPS, whose
            # autograd engine cannot receive float64 gradients.
            return exact_value + (
                native_density - native_density.detach()
            )
        return log_density

    def log_time_density(
        self,
        mixture_logits,
        quantiles,
        delta_times,
        rates,
        generated_masses,
        random_spec=None,
    ):
        """Return ``log q(delta_t)`` with the exact CDF Jacobian."""

        log_q_u = self._log_quantile_density_float64(
            mixture_logits,
            quantiles,
            random_spec=random_spec,
        )
        density_device = log_q_u.device
        delta_times = torch.as_tensor(
            delta_times,
            dtype=torch.float64,
            device=density_device,
        )
        rates = torch.as_tensor(
            rates,
            dtype=torch.float64,
            device=density_device,
        )
        generated_masses = torch.as_tensor(
            generated_masses,
            dtype=torch.float64,
            device=density_device,
        )
        if not bool(
            (
                torch.isfinite(delta_times)
                & torch.isfinite(rates)
                & torch.isfinite(generated_masses)
                & (delta_times >= 0.0)
                & (rates > 0.0)
                & (generated_masses > 0.0)
            )
            .all()
            .detach()
            .cpu()
            .item()
        ):
            raise ValueError("continuous time-density inputs must be finite")
        log_density = (
            log_q_u
            + torch.log(rates)
            - rates * delta_times
            - torch.log(generated_masses)
        )
        if mixture_logits.device.type == "mps":
            exact_value = log_density.to(
                device=mixture_logits.device,
                dtype=mixture_logits.dtype,
            )
            native_density = self._log_quantile_density_native(
                mixture_logits,
                quantiles,
                random_spec=random_spec,
            )
            return exact_value + (
                native_density - native_density.detach()
            )
        return log_density

    def forward(self, action_and_time_features):
        features = action_and_time_features
        if self.feature is not None:
            features = self.feature(features)
        return self.output_layer(features)


# The public name remains concise while the checkpoint version identifies the
# incompatible continuous implementation.
TimeModel = BernsteinBetaTimeModel
