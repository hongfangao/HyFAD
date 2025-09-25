import math
import torch
from einops import rearrange
from torch.fft import irfft, rfft
import torch.nn.functional as F


# def dft(x: torch.Tensor) -> torch.Tensor:
#     """Compute the DFT of the input time series by keeping only the non-redundant components.

#     Args:
#         x (torch.Tensor): Time series of shape (batch_size, n_channels, max_len).

#     Returns:
#         torch.Tensor: DFT of x with the same size (batch_size, n_channels, max_len).
#     """

#     max_len = x.size(2)

#     # Compute the FFT until the Nyquist frequency
#     dft_full = rfft(x, dim=2, norm="ortho")
#     dft_re = torch.real(dft_full)
#     dft_im = torch.imag(dft_full)

#     # The first harmonic corresponds to the mean, which is always real
#     zero_padding = torch.zeros_like(dft_im[:, :, 0], device=x.device)
#     assert torch.allclose(
#         dft_im[:, :, 0], zero_padding
#     ), f"The first harmonic of a real time series should be real, yet got imaginary part {dft_im[:, 0, :]}."
#     dft_im = dft_im[:, :, 1:]

#     # If max_len is even, the last component is always zero
#     if max_len % 2 == 0:
#         assert torch.allclose(
#             dft_im[:, :, -1], zero_padding
#         ), f"Got an even {max_len=}, which should be real at the Nyquist frequency, yet got imaginary part {dft_im[:, -1, :]}."
#         dft_im = dft_im[:, :, :-1]

#     # energy balance
#     dft_im = dft_im / math.sqrt(2)
#     # Concatenate real and imaginary parts
#     x_tilde = torch.cat((dft_re, dft_im), dim=2)
#     assert (
#         x_tilde.size() == x.size()
#     ), f"The DFT and the input should have the same size. Got {x_tilde.size()} and {x.size()} instead."

#     return x_tilde


# def idft(x: torch.Tensor) -> torch.Tensor:
#     """Compute the inverse DFT of the input DFT that only contains non-redundant components.

#     Args:
#         x (torch.Tensor): DFT of shape (batch_size, n_channels, max_len).

#     Returns:
#         torch.Tensor: Inverse DFT of x with the same size (batch_size, n_channels, max_len).
#     """

#     max_len = x.size(2)
#     n_real = math.ceil((max_len + 1) / 2)

#     # Extract real and imaginary parts
#     x_re = x[:, :, :n_real]
#     x_im = x[:, :, n_real:]

#     # Create imaginary tensor
#     zero_padding = torch.zeros(size=(x.size(0), x.size(1), 1)).to(x_im)
#     x_im = torch.cat((zero_padding, x_im), dim=2)

#     # If number of time steps is even, put the null imaginary part
#     if max_len % 2 == 0:
#         x_im = torch.cat((x_im, zero_padding), dim=2)

#     assert (
#         x_im.size() == x_re.size()
#     ), f"The real and imaginary parts should have the same shape, got {x_re.size()} and {x_im.size()} instead."

#     # energy balance
#     x_im = x_im * math.sqrt(2)
#     x_freq = torch.complex(x_re, x_im)

#     # Apply IFFT
#     x_time = irfft(x_freq, n=max_len, dim=2, norm="ortho")

#     assert isinstance(x_time, torch.Tensor)
#     assert (
#         x_time.size() == x.size()
#     ), f"The inverse DFT and the input should have the same size. Got {x_time.size()} and {x.size()} instead."

#     return x_time


def _dct_ortho_matrix(L: int, device=None, dtype=None):
    '''
    construct DCT matrix
    '''
    n = torch.arange(L, device=device, dtype=dtype).reshape(L, 1)
    k = torch.arange(L, device=device, dtype=dtype).reshape(1, L)
    M = torch.cos(math.pi*(n+0.5)*k/L)
    M *= math.sqrt(2.0/L)
    M[:, 0] /= math.sqrt(2.0)
    return M


def dct(x: torch.Tensor) -> torch.Tensor:
    '''
    apply Discrete cosine transform to the length(last) dimension
    '''
    assert x.ndim >= 1
    L = x.size(-1)
    M = _dct_ortho_matrix(L, device=x.device, dtype=x.dtype)
    return torch.matmul(x, M)


def idct(x: torch.Tensor) -> torch.Tensor:
    '''
    apply inverse Discrete cosine transform to the length(last) dimension
    '''
    assert x.ndim >= 1
    L = x.size(-1)
    M = _dct_ortho_matrix(L, device=x.device)
    return torch.matmul(x, M.transpose(0, 1))


def dwt_haar(x: torch.Tensor) -> torch.Tensor:
    '''
    Haar DWT
    B, K, L -> B, K, L
    '''
    assert x.ndim == 3
    B, K, L = x.shape
    if L % 2 == 1:
        raise ValueError(f"Haar DWT requires even length, got L={L}")

    x1 = x.reshape(B*K, 1, L)
    h = torch.tensor(
        [1.0/math.sqrt(2.0), 1.0/math.sqrt(2.0)],
        device=x.device,
        dtype=x.dtype
    ).view(1, 1, 2)

    g = torch.tensor(
        [1.0/math.sqrt(2.0), -1.0/math.sqrt(2.0)],
        device=x.device,
        dtype=x.dtype
    ).view(1, 1, 2)

    cA = F.conv1d(x1, h, stride=2)
    cD = F.conv1d(x1, g, stride=2)

    coeff = torch.cat([cA, cD], dim=2).reshape(B, K, L)
    return coeff


def idwt_haar(coeff: torch.Tensor) -> torch.Tensor:
    assert coeff.ndim == 3
    B, K, L = coeff.shape
    if L % 2 == 1:
        raise ValueError(
            f"inverse haar dwt requires even sequence length, got L:{L}")

    Lh = L // 2

    cA = coeff[:, :, :Lh].reshape(B*K, 1, Lh)
    cD = coeff[:, :, Lh:].reshape(B*K, 1, Lh)

    h = torch.tensor(
        [1.0/math.sqrt(2.0), 1.0/math.sqrt(2.0)],
        device=coeff.device,
        dtype=coeff.dtype
    ).view(1, 1, 2)

    g = torch.tensor(
        [1.0/math.sqrt(2.0), -1.0/math.sqrt(2.0)],
        device=coeff.device,
        dtype=coeff.dtype
    ).view(1, 1, 2)

    xA = F.conv_transpose1d(cA, h, stride=2)
    xD = F.conv_transpose1d(cD, g, stride=2)
    x = (xA + xD).reshape(B, K, L)
    return x


def dft_unitary(x: torch.Tensor) -> torch.Tensor:
    """
    正交打包的 rFFT：把非冗余谱打平成长度 L 的实向量，并保证 ||x||_2 = ||y||_2
    x: (B, C, L)  -> y: (B, C, L)
    """
    assert x.dtype in (torch.float32, torch.float64)
    B, C, L = x.shape
    X = rfft(x, n=L, dim=2, norm="ortho")      # (B,C,Nr), Nr = floor(L/2)+1
    re = X.real                                # (B,C,Nr)
    im = X.imag                                # (B,C,Nr)

    Nr = re.size(2)
    # 去掉 DC 的虚部；若 L 为偶数，再去掉 Nyquist 的虚部
    if L % 2 == 0:
        im_trim = im[:, :, 1:Nr-1]             # (B,C,Nr-2)
    else:
        im_trim = im[:, :, 1:Nr]               # (B,C,Nr-1)

    sqrt2 = math.sqrt(2.0)
    # 对非 DC/非 Nyquist 的实部乘 sqrt(2)
    scale_re = torch.ones_like(re)
    if L % 2 == 0:
        if Nr - 2 > 0:
            scale_re[:, :, 1:Nr-1] = sqrt2
    else:
        scale_re[:, :, 1:Nr] = sqrt2

    re_scaled = re * scale_re
    im_scaled = im_trim * sqrt2

    y = torch.cat([re_scaled, im_scaled], dim=2)  # (B,C,L)
    return y


def idft_unitary(y: torch.Tensor) -> torch.Tensor:

    assert y.dtype in (torch.float32, torch.float64)
    B, C, L = y.shape
    Nr = L // 2 + 1                             # rFFT 非冗余长度

    re_scaled = y[:, :, :Nr]                    # (B,C,Nr)
    im_scaled = y[:, :, Nr:]                    # (B,C,L-Nr) == (Nr-1 或 Nr-2)

    sqrt2 = math.sqrt(2.0)
    # 还原实部缩放
    scale_re = torch.ones_like(re_scaled)
    if L % 2 == 0:
        if Nr - 2 > 0:
            scale_re[:, :, 1:Nr-1] = sqrt2
    else:
        scale_re[:, :, 1:Nr] = sqrt2
    re = re_scaled / scale_re

    # 还原虚部并补上 DC / Nyquist 的 0
    zeros = torch.zeros(B, C, 1, device=y.device, dtype=y.dtype)
    if L % 2 == 0:
        im = torch.cat([zeros, im_scaled / sqrt2, zeros], dim=2)  # (B,C,Nr)
    else:
        im = torch.cat([zeros, im_scaled / sqrt2], dim=2)         # (B,C,Nr)

    X = torch.complex(re, im)                 # (B,C,Nr)
    x = irfft(X, n=L, dim=2, norm="ortho")    # (B,C,L)
    return x


def dft(x: torch.Tensor) -> torch.Tensor:
    """Compute the DFT of the input time series by keeping only the non-redundant components.

    Args:
        x (torch.Tensor): Time series of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: DFT of x with the same size (batch_size, max_len, n_channels).
    """

    max_len = x.size(1)

    # Compute the FFT until the Nyquist frequency
    dft_full = rfft(x, dim=1, norm="ortho")
    dft_re = torch.real(dft_full)
    dft_im = torch.imag(dft_full)

    # The first harmonic corresponds to the mean, which is always real
    zero_padding = torch.zeros_like(dft_im[:, 0, :], device=x.device)
    assert torch.allclose(
        dft_im[:, 0, :], zero_padding
    ), f"The first harmonic of a real time series should be real, yet got imaginary part {dft_im[:, 0, :]}."
    dft_im = dft_im[:, 1:]

    # If max_len is even, the last component is always zero
    if max_len % 2 == 0:
        assert torch.allclose(
            dft_im[:, -1, :], zero_padding
        ), f"Got an even {max_len=}, which should be real at the Nyquist frequency, yet got imaginary part {dft_im[:, -1, :]}."
        dft_im = dft_im[:, :-1]

    # Concatenate real and imaginary parts
    x_tilde = torch.cat((dft_re, dft_im), dim=1)
    assert (
        x_tilde.size() == x.size()
    ), f"The DFT and the input should have the same size. Got {x_tilde.size()} and {x.size()} instead."

    return x_tilde


def idft(x: torch.Tensor) -> torch.Tensor:
    """Compute the inverse DFT of the input DFT that only contains non-redundant components.

    Args:
        x (torch.Tensor): DFT of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: Inverse DFT of x with the same size (batch_size, max_len, n_channels).
    """

    max_len = x.size(1)
    n_real = math.ceil((max_len + 1) / 2)

    # Extract real and imaginary parts
    x_re = x[:, :n_real, :]
    x_im = x[:, n_real:, :]

    # Create imaginary tensor
    zero_padding = torch.zeros(size=(x.size(0), 1, x.size(2)), device=x.device)
    x_im = torch.cat((zero_padding, x_im), dim=1)
    # If number of time steps is even, put the null imaginary part
    if max_len % 2 == 0:
        x_im = torch.cat((x_im, zero_padding), dim=1)

    assert (
        x_im.size() == x_re.size()
    ), f"The real and imaginary parts should have the same shape, got {x_re.size()} and {x_im.size()} instead."

    x_freq = torch.complex(x_re, x_im)

    # Apply IFFT
    x_time = irfft(x_freq, n=max_len, dim=1, norm="ortho")

    assert isinstance(x_time, torch.Tensor)
    assert (
        x_time.size() == x.size()
    ), f"The inverse DFT and the input should have the same size. Got {x_time.size()} and {x.size()} instead."

    return x_time
