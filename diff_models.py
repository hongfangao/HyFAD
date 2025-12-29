import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
from fourier import dft, idft
from fourier import dft_unitary, idft_unitary
from fourier import dct, idct
from fourier import dwt_haar, idwt_haar

# class EnergyAdaptiveDiffusionEmbedding(nn.Module):
#     def __init__(
#         self,
#         num_steps: int = 128,
#         d_model: int = 128,
#         projection_dim: int = None,
#         sigma: float = 0.1,
#         gamma: float = 0.1,
#         n: float = 1.0,
#         max_freq: float = 100.0,
#         kappa: float = 0.5,
#         eps: float = 1e-6,
#         gate_floor: float = 0.3,
#         t2f: str = 'dft',
#         pool: str = "mean",
#         normalize_gate: bool = False,
#         detach_gate: bool = True,
#         use_missingness: bool = True, # gate modifications
#     ):
#         super().__init__()
#         assert d_model % 2 == 0, "d_model must be even for frequency domain transformation"
#         self.num_steps = num_steps
#         self.d_model = d_model
#         self.dim = d_model // 2
#         self.kappa = kappa
#         self.eps = eps
#         self.gate_floor = gate_floor
#         if t2f == 'dft':
#             self.t2f = dft_unitary
#         elif t2f == 'dwt':
#             self.t2f = dwt_haar
#         elif t2f == 'dct':
#             self.t2f = dct
#         else:
#             raise NotImplementedError("Not Implemented Frequency Domain Transformation {t2f}")
#         self.pool = pool
#         self.normalize_gate = normalize_gate
#         self.detach_gate = detach_gate
#         self.use_missingness = use_missingness

#         if projection_dim is None:
#             projection_dim = d_model

#         self.proj1 = nn.Linear(d_model, projection_dim)
#         self.proj2 = nn.Linear(projection_dim, projection_dim)

#         self.register_buffer(
#             "freq_bands",
#             torch.linspace(1.0, max_freq, self.dim),
#             persistent=False
#         )
        
#         # pre-store cut-off frequency w.r.t to t
#         t = torch.arange(num_steps).float()
#         t_scaled = t * (sigma**2) * gamma + 1e-7
#         cutoff = t_scaled ** (-1.0/n)
#         self.register_buffer(
#             "cutoff",
#             cutoff,
#             persistent=False
#         )

#     # def _band_pool(
#     #     self,
#     #     power: torch.Tensor
#     # ) -> torch.Tensor:
#     #     B, K, L = power.shape
#     #     power = power.mean(dim=1)
#     #     dim = self.dim
#     #     if L % dim != 0:
#     #         pad = dim - (L%dim)
#     #         power = F.pad(power, (0, pad), mode='constant', value=0.0)
#     #         L2 = power.size(-1)
#     #     else:
#     #         L2 = L

#     #     g = L2 // dim
#     #     power = power.view(B, dim, g)
        
#     #     if self.pool == "sum":
#     #         band = power.sum(dim=-1)
#     #     elif self.pool == "mean":
#     #         band = power.mean(dim=-1)
#     #     else:
#     #         raise NotImplementedError("band pooling should be mean or sum, got {}".format(self.pool)) 
#     #     return band
    
#     def _band_pool_1d(
#         self,
#         p: torch.Tensor,
#         out_dim: int,
#     ) -> torch.Tensor:
#         B, N = p.shape
#         if N % out_dim != 0:
#             pad = out_dim - (N%out_dim)
#             p = F.pad(p, (0, pad), mode="constant", value=0.0)
#             N2 = p.size(-1)
#         else:
#             N2 = N
        
#         g = N2 // out_dim
#         p = p.view(B, out_dim, g)
#         if self.pool == "sum":
#             band = p.sum(dim=-1)
#         elif self.pool == "mean":
#             band = p.mean(dim=-1)
#         else:
#             raise NotImplementedError("band pooling should be mean or sum, got {}".format(self.pool)) 
#         return band
    
#     def _band_pool(
#         self, 
#         power: torch.Tensor,
#     ) -> torch.Tensor:
#         power = power.mean(dim=1)
#         return self._band_pool_1d(power, self.dim)

#     def _power_spectrum(
#         self,
#         z: torch.Tensor
#     ) -> torch.Tensor:
#         '''
#         calculate the power p after Frequency domain transformation
#         '''
#         if self.t2f == dft_unitary:
#             B, K, L = z.shape
#             Nr = L // 2 + 1

#             re = z[:, :, :Nr]     # (B,K,Nr)
#             im = z[:, :, Nr:]     # packed imag, scaled by sqrt(2)

#             P = z.new_zeros(B, K, Nr)

#             # DC
#             P[:, :, 0] = re[:, :, 0].pow(2)

#             if L % 2 == 0:
#                 # Nyquist-only bin (pure real)
#                 P[:, :, -1] = re[:, :, -1].pow(2)
#                 # k = 1..Nr-2 (scaled by sqrt2 in both re/im => divide by 2)
#                 if Nr - 2 > 0:
#                     P[:, :, 1:Nr-1] = 0.5 * (re[:, :, 1:Nr-1].pow(2) + im[:, :, :Nr-2].pow(2))
#             else:
#                 # k = 1..Nr-1
#                 if Nr - 1 > 0:
#                     P[:, :, 1:Nr] = 0.5 * (re[:, :, 1:Nr].pow(2) + im[:, :, :Nr-1].pow(2))

#             return P
#         else:
#             return z.pow(2)

#     def _schedule_weight(
#         self,
#         diffusion_step: torch.LongTensor
#     ) -> torch.Tensor:
#         c = self.cutoff[diffusion_step].unsqueeze(-1)
#         fb = self.freq_bands.unsqueeze(0)
#         w = torch.exp(-(fb**2)/(c**2+1e-12))
#         return w
    
#     def _reliability_weight(
#         self,
#         cond_obs: torch.Tensor,
#         cond_mask: torch.Tensor,
#     ) -> torch.Tensor:

#         obs_f = self.t2f(cond_obs.float())
#         missing = 1.0 - cond_mask.float()
#         missing_f = self.t2f(missing)
#         pm = self._power_spectrum(missing_f)
#         po = self._power_spectrum(obs_f)
#         O_band = self._band_pool(po)
#         M_band = self._band_pool(pm)
#         w = O_band / (O_band + self.kappa * M_band + self.eps)
#         return w.clamp(0.0, 1.0)
                
#     def forward(
#         self,
#         diffusion_step: torch.LongTensor,
#         cond_obs: torch.Tensor = None,
#         cond_mask: torch.Tensor = None
#     ):
#         B = diffusion_step.shape[0]
#         fb = self.freq_bands.to(diffusion_step.device)
#         schw = self._schedule_weight(diffusion_step)
        
#         if cond_obs is not None and cond_mask is not None:
#             relw = self._reliability_weight(cond_obs, cond_mask)
#         else:
#             relw = torch.ones(B, self.dim, device=diffusion_step.device)
        
#         relw = self.gate_floor + (1.0 - self.gate_floor) * relw
#         gate = schw * relw

#         # detach gate gradient
#         if self.detach_gate:
#             gate = gate.detach()
#         if self.normalize_gate:
#             gate = gate / (gate.mean(dim=-1, keepdim=True) + self.eps)

#         t = diffusion_step.float().unsqueeze(-1)
#         phase = t * fb.unsqueeze(0)
#         emb = torch.cat([torch.sin(phase)*gate, torch.cos(phase)*gate],dim=-1)

#         x = self.proj1(emb)
#         x = F.silu(x)
#         x = self.proj2(x)
#         x = F.silu(x)

#         return x

class FreqThresholdSelectiveDiffusionEmbedding(nn.Module):
    def __init__(
        self,
        num_steps: int = 128,
        d_model: int = 128,
        projection_dim: int = None,
        max_freq: float = 100.0,
        t2f: str = 'dft',
        gamma_thr: float = 1.0,
        tau: float = 0.7,
        eps: float = 1e-6,
        gate_floor: float = 0.3,
        detach_gate: bool = True,
        normalize_gate: bool = True,
        logit_clamp: float = 12.0,
        threshold_calib: str = 'global_ema',
        ema_decay: float = 0.99,
        power_reduce: str = 'mean',
    ):
        super().__init__()
        assert d_model%2 == 0, "d_model must be even"
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

        if t2f == 'dft':
            self.t2f = dft_unitary
        elif t2f == 'dct':
            self.t2f = dct
        elif t2f == 'dwt':
            self.t2f = dwt_haar
        else:
            raise NotImplementedError("Unsupported transformation:{t2f}")
        
        self.register_buffer(
            "freq_u_dim", 
            torch.linspace(0.0, 1.0, self.dim),
            persistent=False
        )
        self.register_buffer(
            "freq_bands_dim",
            torch.linspace(0.0, max_freq, self.dim),
            persistent=False
        )

        if self.threshold_calib == 'global_ema':
            self.register_buffer("global_scale_ema", torch.tensor(1.0), persistent=True)

        self.proj1 = nn.Linear(d_model, projection_dim)
        self.proj2 = nn.Linear(projection_dim, projection_dim)

    # def _compute_scale(
    #     self,
    #     P_band: torch.Tensor
    # ) -> torch.Tensor:
    #     if self.threshold_calib == "none":
    #         return torch.ones(P_band.size(0), 1, device=P_band.device, dtype=P_band.dtype)

    #     if self.threshold_calib == "median":
    #         s = P_band.detach().median(dim=-1, keepdim=True).values  # (B,1)
    #         return s.clamp_min(self.eps)

    #     if self.threshold_calib == "mean":
    #         s = P_band.detach().mean(dim=-1, keepdim=True)  # (B,1)
    #         return s.clamp_min(self.eps)

    #     if self.threshold_calib == "global_ema":
    #         batch_med = P_band.detach().median().clamp_min(self.eps)
    #         self.global_scale_ema = self.global_scale_ema * self.ema_decay + batch_med * (1.0 - self.ema_decay)
    #         s = self.global_scale_ema.to(P_band.dtype)
    #         return s.view(1, 1).expand(P_band.size(0), 1)

    def _power_spectrum(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        '''
        calculate the power p after Frequency domain transformation
        '''
        if self.t2f == dft_unitary:
            B, K, L = z.shape
            Nr = L // 2 + 1

            re = z[:, :, :Nr]     # (B,K,Nr)
            im = z[:, :, Nr:]     # packed imag, scaled by sqrt(2)

            P = z.new_zeros(B, K, Nr)

            # DC
            P[:, :, 0] = re[:, :, 0].pow(2)

            if L % 2 == 0:
                # Nyquist-only bin (pure real)
                P[:, :, -1] = re[:, :, -1].pow(2)
                # k = 1..Nr-2 (scaled by sqrt2 in both re/im => divide by 2)
                if Nr - 2 > 0:
                    P[:, :, 1:Nr-1] = 0.5 * (re[:, :, 1:Nr-1].pow(2) + im[:, :, :Nr-2].pow(2))
            else:
                # k = 1..Nr-1
                if Nr - 1 > 0:
                    P[:, :, 1:Nr] = 0.5 * (re[:, :, 1:Nr].pow(2) + im[:, :, :Nr-1].pow(2))

            return P
        else:
            return z.pow(2)
    def _reduce_channels(self, P: torch.Tensor) -> torch.Tensor:
        """
        P: (B,K,N) -> (B,N)
        """
        if self.power_reduce == "mean":
            return P.mean(dim=1)
        if self.power_reduce == "median":
            return P.median(dim=1).values
        raise ValueError(f"power_reduce must be mean/median, got {self.power_reduce}")

    @torch.no_grad()
    def _update_global_scale_ema(self, P_bin: torch.Tensor):
        """
        P_bin: (B,N) power on bins, update scalar EMA using global median.
        """
        # scalar median over all (B,N)
        batch_med = P_bin.detach().median().clamp_min(self.eps)
        self.global_scale_ema.mul_(self.ema_decay).add_(batch_med * (1.0 - self.ema_decay))

    def _get_scale(self, P_bin: torch.Tensor) -> torch.Tensor:
        """
        return scale: (1,1) or (B,1) depending on calibration strategy
        """
        if self.threshold_calib == "none":
            return torch.ones(1, 1, device=P_bin.device, dtype=P_bin.dtype)
        if self.threshold_calib == "global_ema":
            # update only during training to avoid eval-time drift
            if self.training:
                self._update_global_scale_ema(P_bin)
            s = self.global_scale_ema.to(dtype=P_bin.dtype, device=P_bin.device).clamp_min(self.eps)
            return s.view(1, 1)
        raise ValueError(f"threshold_calib must be none/global_ema, got {self.threshold_calib}")

    def _interp_1d(self, x: torch.Tensor, out_len: int) -> torch.Tensor:
        """
        x: (B,N) -> (B,out_len) via linear interpolation
        """
        x_ = x.unsqueeze(1)  # (B,1,N)
        y_ = F.interpolate(x_, size=out_len, mode="linear", align_corners=False)
        return y_.squeeze(1)

    
    # def _band_pool_1d(
    #     self,
    #     p: torch.Tensor,
    #     out_dim: int,
    # ) -> torch.Tensor:
    #     B, N = p.shape
    #     if N % out_dim != 0:
    #         pad = out_dim - (N%out_dim)
    #         p = F.pad(p, (0, pad), mode="constant", value=0.0)
    #         N2 = p.size(-1)
    #     else:
    #         N2 = N
        
    #     g = N2 // out_dim
    #     p = p.view(B, out_dim, g)
    #     if self.pool == "sum":
    #         band = p.sum(dim=-1)
    #     elif self.pool == "mean":
    #         band = p.mean(dim=-1)
    #     else:
    #         raise NotImplementedError("band pooling should be mean or sum, got {}".format(self.pool)) 
    #     return band
    
    # def _band_pool(
    #     self, 
    #     power: torch.Tensor,
    # ) -> torch.Tensor:
    #     power = power.mean(dim=1)
    #     return self._band_pool_1d(power, self.dim)

    def forward(
        self,
        diffusion_step: torch.LongTensor,
        N_t: torch.Tensor,
        signal_proxy: torch.Tensor
    ) -> torch.Tensor:
        B = diffusion_step.shape[0]
        device = diffusion_step.device

        # evaluating P_t(\omega)
        z = self.t2f(signal_proxy.to(torch.float32))   # (B,K,L*)
        P = self._power_spectrum(z)          # (B,K,Nfreq)

        P_bin = self._reduce_channels(P)
        Nbin = P_bin.size(-1)

        Nt = N_t.view(B, 1).to(device=device, dtype=P_bin.dtype).clamp_min(self.eps)
        scale = self._get_scale(P_bin).to(P_bin.dtype)
        T = (self.gamma_thr * Nt * scale).clamp_min(self.eps)

        logP = torch.log(P_bin + self.eps)
        logT = torch.log(T + self.eps)
        x = (logP - logT) / max(self.tau, self.eps)

        if self.logit_clamp is not None:
            x = x.clamp(-self.logit_clamp, self.logit_clamp)

        gate_bin = torch.sigmoid(x)                    # (B,Nbin)
        gate_bin = self.gate_floor + (1.0 - self.gate_floor) * gate_bin

        # ===== map gate from bins -> dim =====
        gate = self._interp_1d(gate_bin, self.dim)     # (B,dim)

        if self.detach_gate:
            gate = gate.detach()
        if self.normalize_gate:
            gate = gate / (gate.mean(dim=-1, keepdim=True) + self.eps)

        # ===== build basis aligned with interpolation grid =====
        fb = self.freq_bands_dim.to(device=device, dtype=gate.dtype)  # (dim,)

        t = diffusion_step.to(dtype=gate.dtype).unsqueeze(-1)         # (B,1)
        phase = t * fb.unsqueeze(0)                                   # (B,dim)

        emb = torch.cat(
            [torch.sin(phase) * gate, torch.cos(phase) * gate],
            dim=-1
        )

        x = self.proj1(emb)
        x = F.silu(x)
        x = self.proj2(x)
        x = F.silu(x)
        return x

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
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
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class HighPassDiffusionEmbedding(nn.Module):
    def __init__(
        self,
        num_steps: int,
        embedding_dim: int = 128,
        projection_dim: int = None,
        sigma: float = 0.1,
        gamma: float = 0.1,
        n: float = 1.0,
        max_freq: float = 100.0,
    ):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim

        self.embedding_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(
                num_steps=num_steps,
                d_model=embedding_dim,
                sigma=sigma,
                gamma=gamma,
                n=n,
                max_freq=max_freq,
            ),
            persistent=False,
        )

        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            diffusion_step: (B,) int64 tensor
        Returns:
            (B, projection_dim)
        """
        x = self.embedding[diffusion_step]  # (B, d_model)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(
        self,
        num_steps: int,
        d_model: int,
        sigma: float,
        gamma: float,
        n: float,
        max_freq: float,
    ) -> torch.Tensor:
        """
        Returns:
            embedding_table: (num_steps, d_model)
        """
        assert d_model % 2 == 0, "embedding_dim must be even"

        dim = d_model // 2
        freq_bands = torch.linspace(1.0, max_freq, dim)  # (dim,)
        t = torch.arange(num_steps).float()  # (T,)
        t_scaled = t * sigma**2 * gamma + 1e-7
        cutoff = t_scaled ** (-1.0 / n)  # (T,)

        # Compute frequency masking weights (T, dim)
        weights = torch.exp(-freq_bands[None, :] ** 2 / (cutoff[:, None] ** 2))

        # Create gated sinusoidal embeddings (T, dim*2)
        phase = t[:, None] * freq_bands[None, :]  # (T, dim)
        sin_part = torch.sin(phase) * weights
        cos_part = torch.cos(phase) * weights
        embedding = torch.cat([sin_part, cos_part], dim=1)  # (T, d_model)
        return embedding

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
        # self.freq_diffusion_embedding = HighPassDiffusionEmbedding(
        #     num_steps=config["num_steps"],
        #     embedding_dim=config["freq_diffusion_embedding_dim"],
        # )
        # self.freq_diffusion_embedding = EnergyAdaptiveDiffusionEmbedding(
        #     num_steps=config["num_steps"],
        #     d_model=config["freq_diffusion_embedding_dim"],
        #     t2f=config["trans"],
        #     kappa=0.5
        # )
        self.freq_diffusion_embedding = FreqThresholdSelectiveDiffusionEmbedding(
            num_steps=config["num_steps"],
            d_model=config["freq_diffusion_embedding_dim"],
            projection_dim=config["freq_diffusion_embedding_dim"],
            t2f=config["trans"],
        )

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
                    side_dim = config["side_dim_freq"],
                    channels = self.channels,
                    diffusion_embedding_dim = config['freq_diffusion_embedding_dim'],
                    nheads = config['nheads_freq'],
                    is_linear = config['is_linear_freq']
                )
                for _ in range(config["layers_f"])
            ]
        )
        
    def forward_time(self, x, cond_info_t, diffusion_step):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K*L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info_t, diffusion_emb)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K*L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, K, L)
        return x
    
    def forward_freq(self, x, cond_info_f, diffusion_step, N_t, signal_proxy):
        B, inputdim, K, L = x.shape
        x = x.reshape(B*inputdim, K, L)
        x = self.t2f(x)
        x = x.reshape(B, inputdim, K*L)
        x = self.input_projection_f(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        # freq_diffusion_emb = self.freq_diffusion_embedding(diffusion_step)
        freq_diffusion_emb = self.freq_diffusion_embedding(
            diffusion_step = diffusion_step,
            N_t = N_t,
            signal_proxy = signal_proxy
        )
        skip_f = []
        for layer in self.residual_layers_f:
            x, skip_connection_f = layer(x, cond_info_f, freq_diffusion_emb)
            skip_f.append(skip_connection_f)
        x = torch.sum(torch.stack(skip_f), dim=0) / math.sqrt(len(self.residual_layers_f))
        x = x.reshape(B, self.channels, K*L)
        x = self.output_projection1_f(x)
        x = F.relu(x)
        x = self.output_projection2_f(x)
        x = x.reshape(B, K, L)
        return x

    def forward(self, x, cond_info_t, cond_info_f, diffusion_step):
        pred_t = self.forward_time(x, cond_info_t, diffusion_step)
        pred_f = self.forward_freq(x, cond_info_f, diffusion_step)
        return pred_t, pred_f

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
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
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip

