import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from linear_attention_transformer import LinearAttentionTransformer
from fourier import dft_unitary, idft_unitary
from fourier import dct, idct
from fourier import dwt_haar, idwt_haar

from utils import cosine_beta_scheduler
from utils import set_noise_scaling_identity
from utils import compute_alphas

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_linear_trans(heads=8, layers=1, channels=64, localheads=0, localwindow=0):
    return LinearAttentionTransformer(
        dim=channels,
        depth=layers,
        heads=heads,
        max_seq_len=256,
        n_local_attn_heads=0,
        local_attn_window_size=0,
    )


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step: torch.LongTensor) -> torch.Tensor:
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class FreqThresholdSelectiveDiffusionEmbedding(nn.Module):
    def __init__(
        self,
        num_steps: int = 128,
        d_model: int = 128,
        projection_dim: int = None,
        max_freq: float = 100.0,
        t2f: str = "dft",
        gamma_thr: float = 1.0,
        tau: float = 0.7,
        eps: float = 1e-6,
        gate_floor: float = 0.3,
        detach_gate: bool = True,
        normalize_gate: bool = True,
        logit_clamp: float = 12.0,
        threshold_calib: str = "global_ema",
        ema_decay: float = 0.99,
        power_reduce: str = "mean",
        kappa_miss: float = 0.5,
        use_missing_reliability: bool = True,
        use_stage_schedule: bool = True,
        schedule_sharpness: float = 2.0,
        cutoff_min: float = 1.0,
        cutoff_max: float = 100.0,
        cutoff_power: float = 1.0,
        schedule_mode: str = "low2high", 
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        if projection_dim is None:
            projection_dim = d_model

        self.num_steps = num_steps
        self.d_model = d_model
        self.dim = d_model // 2

        self.gamma_thr = gamma_thr
        self.tau = tau
        self.eps = eps
        self.gate_floor = gate_floor
        self.detach_gate = detach_gate
        self.normalize_gate = normalize_gate
        self.logit_clamp = logit_clamp
        self.threshold_calib = threshold_calib
        self.ema_decay = ema_decay
        self.power_reduce = power_reduce

        self.kappa_miss = kappa_miss
        self.use_missing_reliability = use_missing_reliability

        self.use_stage_schedule = use_stage_schedule
        self.schedule_sharpness = float(schedule_sharpness)
        self.cutoff_min = float(cutoff_min)
        self.cutoff_max = float(cutoff_max)
        self.cutoff_power = float(cutoff_power)
        self.schedule_mode = schedule_mode

        if t2f == "dft":
            self.t2f = dft_unitary
        elif t2f == "dct":
            self.t2f = dct
        elif t2f == "dwt":
            self.t2f = dwt_haar
        else:
            raise NotImplementedError(f"Unsupported transformation: {t2f}")

        self.register_buffer(
            "freq_bands_dim",
            torch.linspace(0.0, max_freq, self.dim),
            persistent=False,
        )

        if self.threshold_calib == "global_ema":
            self.register_buffer("global_scale_ema", torch.tensor(1.0), persistent=True)

        self.proj1 = nn.Linear(d_model, projection_dim)
        self.proj2 = nn.Linear(projection_dim, projection_dim)

    def _power_spectrum(self, z: torch.Tensor) -> torch.Tensor:
        if self.t2f == dft_unitary:
            B, K, L = z.shape
            Nr = L // 2 + 1

            re = z[:, :, :Nr]
            im = z[:, :, Nr:]

            P = z.new_zeros(B, K, Nr)
            P[:, :, 0] = re[:, :, 0].pow(2)

            if L % 2 == 0:
                P[:, :, -1] = re[:, :, -1].pow(2)
                if Nr - 2 > 0:
                    P[:, :, 1:Nr-1] = 0.5 * (re[:, :, 1:Nr-1].pow(2) + im[:, :, :Nr-2].pow(2))
            else:
                if Nr - 1 > 0:
                    P[:, :, 1:Nr] = 0.5 * (re[:, :, 1:Nr].pow(2) + im[:, :, :Nr-1].pow(2))
            return P
        elif self.t2f == dwt_haar:
            B, K, L = z.shape
            assert L % 2 == 0, "Haar DWT output length should be even"
            L2 = L // 2
            cA = z[:, :, :L2]
            cD = z[:, :, L2:]

            P_low  = (cA ** 2).mean(dim=-1, keepdim=True)  # (B,K,1)
            P_high = (cD ** 2).mean(dim=-1, keepdim=True)  # (B,K,1)
            P = torch.cat([P_low, P_high], dim=-1)         # (B,K,2)
            return P
        else:
            return z.pow(2)

    def _reduce_channels(self, P: torch.Tensor) -> torch.Tensor:
        if self.power_reduce == "mean":
            return P.mean(dim=1)
        if self.power_reduce == "median":
            return P.median(dim=1).values
        raise ValueError(f"power_reduce must be mean/median, got {self.power_reduce}")

    @torch.no_grad()
    def _update_global_scale_ema(self, P_bin: torch.Tensor):
        batch_med = P_bin.detach().median().clamp_min(self.eps)
        self.global_scale_ema.mul_(self.ema_decay).add_(batch_med * (1.0 - self.ema_decay))

    def _get_scale(self, P_bin: torch.Tensor) -> torch.Tensor:
        if self.threshold_calib == "none":
            return torch.ones(1, 1, device=P_bin.device, dtype=P_bin.dtype)
        if self.threshold_calib == "global_ema":
            if self.training:
                self._update_global_scale_ema(P_bin)
            s = self.global_scale_ema.to(dtype=P_bin.dtype, device=P_bin.device).clamp_min(self.eps)
            return s.view(1, 1)
        raise ValueError(f"threshold_calib must be none/global_ema, got {self.threshold_calib}")

    def _interp_1d(self, x: torch.Tensor, out_len: int) -> torch.Tensor:
        x_ = x.unsqueeze(1)  # (B,1,N)
        y_ = F.interpolate(x_, size=out_len, mode="linear", align_corners=False)
        return y_.squeeze(1)

    def _missing_reliability(
        self, 
        cond_mask: torch.Tensor,
        signal_proxy: torch.Tensor
        ) -> torch.Tensor:
        """
        cond_mask: (B,K,L) in {0,1}
        reliability R(ω) = 1 / (1 + kappa * P_missing(ω))  (bin-wise, reduced over K)
        """
        z_s = self.t2f(signal_proxy)
        Ps = self._power_spectrum(z_s)
        Ps_bin = self._reduce_channels(Ps)
        missing = (1.0 - cond_mask.float()).to(torch.float32)     # missing indicator
        z_m = self.t2f(missing)                                   # (B,K,L*)
        Pm = self._power_spectrum(z_m)                             # (B,K,Nbin)
        Pm_bin = self._reduce_channels(Pm)
        R = Ps_bin / (Ps_bin + self.kappa_miss * Pm_bin + self.eps)
        return R.clamp(0.0, 1.0)                                   # (B,Nbin)

    def _schedule_weight(self, diffusion_step: torch.LongTensor, Nbin: int, device, dtype):
        """
        Return w_sched: (B, Nbin), encourages different ω bands at different t.

        We define a cutoff c(t) in [cutoff_min, cutoff_max] over reverse time.
        - schedule_mode="low2high": early reverse steps (tt large) -> small c (low-pass),
                                   late reverse steps (tt small)  -> large c (allow high freq)
        Weight: w = exp( - (ω/c)^p )
        """
        if not self.use_stage_schedule:
            return torch.ones(diffusion_step.size(0), Nbin, device=device, dtype=dtype)

        tt = diffusion_step.to(dtype=torch.float32)
        T = float(self.num_steps - 1)
        u = tt / max(T, 1.0)  # u in [0,1]

        if self.schedule_mode == "low2high":
            s = (1.0 - u).clamp(0.0, 1.0)  # late larger
        elif self.schedule_mode == "high2low":
            s = u.clamp(0.0, 1.0)
        else:
            raise ValueError(f"unknown schedule_mode={self.schedule_mode}")

        s = s.pow(self.cutoff_power)

        c = self.cutoff_min + (self.cutoff_max - self.cutoff_min) * s  # (B,)
        c = c.to(device=device, dtype=dtype).unsqueeze(-1)             # (B,1)

        w_grid = torch.linspace(0.0, 1.0, Nbin, device=device, dtype=dtype).unsqueeze(0)  # (1,Nbin)

        w_grid = w_grid * self.cutoff_max

        p = self.schedule_sharpness
        w = torch.exp(-((w_grid / (c + self.eps)).clamp_min(0.0) ** p))
        return w  # (B,Nbin)

    def forward(
        self,
        diffusion_step: torch.LongTensor,
        N_t: torch.Tensor,
        signal_proxy: torch.Tensor,
        cond_mask: torch.Tensor, 
    ) -> torch.Tensor:
        """
        diffusion_step: (B,)
        N_t: (B,) noise level proxy for threshold
        signal_proxy: (B,K,L) proxy signal for power spectrum
        cond_mask: (B,K,L) observed mask for missing-aware reliability
        """
        B = diffusion_step.shape[0]
        device = diffusion_step.device

        z = self.t2f(signal_proxy.to(torch.float32))
        P = self._power_spectrum(z)              # (B,K,Nbin)
        P_bin = self._reduce_channels(P)         # (B,Nbin)
        Nbin = P_bin.size(-1)

        Nt = N_t.view(B, 1).to(device=device, dtype=P_bin.dtype).clamp_min(self.eps)
        scale = self._get_scale(P_bin).to(P_bin.dtype)
        T = (self.gamma_thr * Nt * scale).clamp_min(self.eps)

        logP = torch.log(P_bin + self.eps)
        logT = torch.log(T + self.eps)
        x = (logP - logT) / max(self.tau, self.eps)
        if self.logit_clamp is not None:
            x = x.clamp(-self.logit_clamp, self.logit_clamp)
        gate_bin = torch.sigmoid(x)  
        gate_bin = self.gate_floor + (1.0 - self.gate_floor) * gate_bin

        if self.use_missing_reliability:
            R = self._missing_reliability(cond_mask, signal_proxy).to(dtype=gate_bin.dtype, device=device)  # (B,Nbin)
        else:
            R = torch.ones_like(gate_bin)

        S = self._schedule_weight(diffusion_step, Nbin, device=device, dtype=gate_bin.dtype)  # (B,Nbin)

        gate_bin = gate_bin * R * S
        gate_bin = gate_bin.clamp(0.0, 1.0)

        gate = self._interp_1d(gate_bin, self.dim)  # (B,dim)

        if self.detach_gate:
            gate = gate.detach()
        if self.normalize_gate:
            gate = gate / (gate.mean(dim=-1, keepdim=True) + self.eps)

        fb = self.freq_bands_dim.to(device=device, dtype=gate.dtype)  # (dim,)
        t = diffusion_step.to(dtype=gate.dtype).unsqueeze(-1)         # (B,1)
        phase = t * fb.unsqueeze(0)                                   # (B,dim)
        emb = torch.cat([torch.sin(phase) * gate, torch.cos(phase) * gate], dim=-1)  # (B,d_model)

        y = self.proj1(emb)
        y = F.silu(y)
        y = self.proj2(y)
        y = F.silu(y)
        return y

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)
        y = y + cond_info

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class diff_CD2(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        trans = config["trans"]
        assert trans in ["dft", "dct", "dwt"]
        if trans == "dft":
            self.t2f = dft_unitary
        elif trans == "dct":
            self.t2f = dct
        elif trans == "dwt":
            self.t2f = dwt_haar

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["time_diffusion_embedding_dim"],
        )

        # freq embedding (patched)
        self.freq_diffusion_embedding = FreqThresholdSelectiveDiffusionEmbedding(
            num_steps=config["num_steps"],
            d_model=config["freq_diffusion_embedding_dim"],
            projection_dim=config["freq_diffusion_embedding_dim"],
            t2f=config["trans"],
            kappa_miss=float(config.get("kappa_miss", 0.5)),
            use_missing_reliability=bool(config.get("use_missing_reliability", True)),
            use_stage_schedule=bool(config.get("use_stage_schedule", True)),
            schedule_sharpness=float(config.get("schedule_sharpness", 2.0)),
            cutoff_min=float(config.get("cutoff_min", 1.0)),
            cutoff_max=float(config.get("cutoff_max", 100.0)),
            cutoff_power=float(config.get("cutoff_power", 1.0)),
            schedule_mode=str(config.get("schedule_mode", "low2high")),
        )
        # self.freq_diffusion_embedding = DiffusionEmbedding(
        #     num_steps = config["num_steps"],
        #     embedding_dim = config["freq_diffusion_embedding_dim"]
        # )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)

        self.input_projection_f = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1_f = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_f = Conv1d_with_init(self.channels, 1, 1)

        nn.init.zeros_(self.output_projection2.weight)
        nn.init.zeros_(self.output_projection2_f.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim_time"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["time_diffusion_embedding_dim"],
                    nheads=config["nheads_time"],
                    is_linear=config["is_linear_time"],
                )
                for _ in range(config["layers_t"])
            ]
        )

        self.residual_layers_f = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim_freq"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["freq_diffusion_embedding_dim"],
                    nheads=config["nheads_freq"],
                    is_linear=config["is_linear_freq"],
                )
                for _ in range(config["layers_f"])
            ]
        )

    def forward_time(self, x, cond_info_t, diffusion_step):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, s = layer(x, cond_info_t, diffusion_emb)
            skip.append(s)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, K, L)
        return x

    def forward_freq(self, x, cond_info_f, diffusion_step, N_t, signal_proxy, cond_mask):
        B, inputdim, K, L = x.shape
        x = x.reshape(B * inputdim, K, L)
        x = self.t2f(x)
        x = x.reshape(B, inputdim, K * L)

        x = self.input_projection_f(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        freq_diffusion_emb = self.freq_diffusion_embedding(
            diffusion_step=diffusion_step,
            N_t=N_t,
            signal_proxy=signal_proxy,
            cond_mask=cond_mask,
        )
        # freq_diffusion_emb = self.diffusion_embedding(
        #     diffusion_step
        # )

        skip = []
        for layer in self.residual_layers_f:
            x, s = layer(x, cond_info_f, freq_diffusion_emb)
            skip.append(s)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers_f))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1_f(x)
        x = F.relu(x)
        x = self.output_projection2_f(x)
        x = x.reshape(B, K, L)
        return x
