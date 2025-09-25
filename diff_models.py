import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
from fourier import dft, idft
from fourier import dft_unitary, idft_unitary
from fourier import dct, idct
from fourier import dwt_haar, idwt_haar



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
        self.freq_diffusion_embedding = HighPassDiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["freq_diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        self.input_projection_f = Conv1d_with_init(1, self.channels, 1)
        self.output_projection1_f = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_f = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        nn.init.zeros_(self.output_projection2_f.weight)


        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
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
                    side_dim = config["side_dim"],
                    channels = self.channels,
                    diffusion_embedding_dim = config['freq_diffusion_embedding_dim'],
                    nheads = config['nheads_freq'],
                    is_linear = config['is_linear_freq']
                )
                for _ in range(config["layers_f"])
            ]
        )
        

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        freq_diffusion_emb = self.freq_diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x_t = x.reshape(B, K, L)
        x_f = self.t2f(x_t)
        x_f = x_f.reshape(B, 1, K*L)
        x = self.input_projection_f(x_f)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        freq_diffusion_emb = self.freq_diffusion_embedding(diffusion_step)
        skip_f = []
        for layer in self.residual_layers_f:
            x, skip_connection_f = layer(x, cond_info, freq_diffusion_emb)
            skip_f.append(skip_connection_f)
        
        x = torch.sum(torch.stack(skip_f), dim=0) / math.sqrt(len(self.residual_layers_f))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1_f(x)
        x = F.relu(x)
        x = self.output_projection2_f(x)
        x_f = x.reshape(B, K, L)
        assert isinstance(x_t, torch.Tensor)
        assert isinstance(x_f, torch.Tensor)
        return x_t, x_f

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
