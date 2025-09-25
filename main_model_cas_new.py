import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CD2
from fourier import dft_unitary, idft_unitary
from fourier import dct, idct
from fourier import dwt_haar, idwt_haar
from utils import cosine_beta_scheduler
from utils import set_noise_scaling, set_noise_scaling_identity
from utils import compute_alphas


class CD2_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.missing_ratio = config["model"]["missing_k"]
        trans = config["diffusion"]["trans"]
        assert trans in ["dft", "dct", "dwt"]
        if trans == "dft":
            self.f2t = idft_unitary
            self.noise_scaling = set_noise_scaling_identity
        elif trans == "dct":
            self.f2t = idct
            self.noise_scaling = set_noise_scaling_identity
        elif trans == "dwt":
            self.f2t = idwt_haar
            self.noise_scaling = set_noise_scaling_identity

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CD2(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"]

        # time schedulers
        if config_diff["schedule"] == "quad":
            beta = np.linspace(
                config_diff["beta_start"] ** 0.5,
                config_diff["beta_end"] ** 0.5,
                self.num_steps,
            ) ** 2
            beta_f = np.linspace(
                config_diff["beta_start_f"] ** 0.5,
                config_diff["beta_end_f"] ** 0.5,
                self.num_steps,
            ) ** 2
        elif config_diff["schedule"] == "linear":
            beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
            beta_f = np.linspace(
                config_diff["beta_start_f"], config_diff["beta_end_f"], self.num_steps
            )
        elif config_diff["schedule"] == "cosine":
            beta = cosine_beta_scheduler(self.num_steps)
            beta_f = cosine_beta_scheduler(self.num_steps)
        else:
            raise ValueError("Unknown time schedule")

        self.beta_torch = torch.tensor(beta, dtype=torch.float32, device=self.device)
        self.beta_f_torch = torch.tensor(beta_f, dtype=torch.float32, device=self.device)
        (
            self.alpha_hat_torch,
            self.alpha_bar_torch,
            self.sqrt_alpha_bar_torch,
            self.sqrt_one_minus_alpha_bar_torch,
        ) = compute_alphas(self.beta_torch)
        (
            self.alpha_hat_torch_f,
            self.alpha_bar_torch_f,
            self.sqrt_alpha_bar_torch_f,
            self.sqrt_one_minus_alpha_bar_torch_f,
        ) = compute_alphas(self.beta_f_torch)

        # mixing and auxiliary weights
        self.lambda_mix = float(config_diff.get("lambda_mix", 0.5))
        self.aux_branch_weight = float(config_diff.get("aux_branch_weight", 0.0))

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2, device=self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)

        feature_embed = self.embed_layer(torch.arange(self.target_dim, device=self.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        lam = self.lambda_mix
        sqrt_lam = lam ** 0.5
        sqrt_1m_lam = (1.0 - lam) ** 0.5

        if is_train != 1:
            t = (torch.ones(B, device=self.device) * set_t).long()
        else:
            t = torch.randint(0, self.num_steps, [B], device=self.device)

        G = self.noise_scaling(L, device=self.device)

        alpha_bar_t = self.alpha_bar_torch
        alpha_hat_t = self.alpha_hat_torch
        alpha_bar_f = self.alpha_bar_torch_f
        alpha_hat_f = self.alpha_hat_torch_f
        beta_t = self.beta_torch
        beta_f = 1.0 - alpha_hat_f

        noisy_x = torch.zeros_like(observed_data)
        true_t = torch.zeros_like(observed_data)
        true_f = torch.zeros_like(observed_data)

        # forward noise synthesis (splitting into time/frequency branches)
        for i in range(B):
            k = int(t[i].item())
            x0 = observed_data[i]
            sqrt_abk_t = torch.sqrt(alpha_bar_t[k])
            sqrt_abk_f = torch.sqrt(alpha_bar_f[k])

            noise_t = torch.zeros_like(x0)
            noise_f = torch.zeros_like(x0)

            for j in range(0, k + 1):
                if j < k:
                    A_t = torch.sqrt(alpha_bar_t[k] / alpha_bar_t[j])
                else:
                    A_t = torch.tensor(1.0, device=self.device)

                coeff_t = torch.sqrt(1.0 - alpha_hat_t[j]) * A_t
                eps_t = torch.randn(K, L, device=self.device)
                noise_t = noise_t + (sqrt_lam * coeff_t) * eps_t

                if j > 0:
                    A_t_extra = torch.sqrt(alpha_bar_t[k] / alpha_bar_t[j - 1])
                else:
                    A_t_extra = torch.tensor(1.0, device=self.device, dtype=alpha_bar_t.dtype)

                if j < k:
                    A_f = torch.sqrt(alpha_bar_f[k] / alpha_bar_f[j])
                else:
                    A_f = torch.tensor(1.0, device=self.device)

                coeff_f = torch.sqrt(1.0 - alpha_hat_f[j]) * A_t_extra * A_f
                eps_f = torch.randn(K, L, device=self.device)
                noise_f = noise_f + (sqrt_1m_lam * coeff_f) * (self.f2t(G * eps_f.unsqueeze(0)).squeeze(0))

            xk = (sqrt_abk_t * sqrt_abk_f) * x0 + noise_t + noise_f
            noisy_x[i] = xk
            true_t[i] = noise_t
            true_f[i] = noise_f

        target_mask = (observed_mask - cond_mask).float()
        num_eval = target_mask.sum()
        denom = (num_eval if num_eval > 0 else 1.0)

        # predict time noise
        inp_t = self.set_input_to_diffmodel(noisy_x, observed_data, cond_mask)
        pred_t, _ = self.diffmodel(inp_t, side_info, t)
        loss_time = (((true_t - pred_t) * target_mask) ** 2).sum() / denom

        # compute effective frequency variance coefficient sigma_f for current step
        sigma2 = torch.zeros(B, 1, 1, device=self.device, dtype=observed_data.dtype)
        for i in range(B):
            k = int(t[i].item())
            if k >= 0:
                s_idx = torch.arange(0, k + 1, device=self.device, dtype=torch.long)

                alpha_bar_t_sminus1 = torch.ones_like(s_idx, dtype=alpha_bar_t.dtype, device=self.device)
                if k >= 1:
                    pos = s_idx > 0
                    alpha_bar_t_sminus1[pos] = alpha_bar_t[s_idx[pos] - 1]
                chain_t = alpha_bar_t[k] / alpha_bar_t_sminus1

                alpha_bar_f_s = torch.ones_like(s_idx, dtype=alpha_bar_f.dtype, device=self.device)
                if k >= 1:
                    pos2 = s_idx < k
                    alpha_bar_f_s[pos2] = alpha_bar_f[s_idx[pos2]]
                chain_f = alpha_bar_f[k] / alpha_bar_f_s

                sb_f = 1.0 - alpha_hat_f[s_idx]
                w2 = ((sqrt_1m_lam ** 2) * sb_f * chain_t * chain_f)
                sigma2[i, 0, 0] = w2.sum()

        sigma_f = torch.sqrt(sigma2)

        # time-step update used to produce x_time
        c_t = (self.beta_torch[t] / torch.sqrt((1.0 - self.alpha_bar_torch[t]))).view(B, 1, 1)
        inv_sqrt_alpha_t = (1.0 / torch.sqrt(self.alpha_hat_torch[t])).view(B, 1, 1)
        x_t_time = (noisy_x - c_t * pred_t.detach()) * inv_sqrt_alpha_t

        # predict frequency noise (standard normal in frequency branch mapped to time domain)
        inp_f = self.set_input_to_diffmodel(x_t_time, observed_data, cond_mask)
        _, pred_f = self.diffmodel(inp_f, side_info, t)

        pred_f_time = sigma_f * self.f2t(G * pred_f)
        loss_freq = (((true_f - pred_f_time) * target_mask) ** 2).sum() / denom

        # optional consistency loss across branches
        recon_tot = pred_t + pred_f_time
        true_tot = true_t + true_f
        loss_consistency = (((true_tot - recon_tot) * target_mask) ** 2).sum() / denom

        loss = loss_time + loss_freq + self.aux_branch_weight * loss_consistency
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)
        return total_input

    @torch.no_grad()
    def impute(self, observed_data, cond_mask, side_info, n_samples: int):
        B, K, L = observed_data.shape
        device = self.device
        dtype = observed_data.dtype

        beta_t = self.beta_torch.to(device=device, dtype=dtype)
        alpha_hat_t = self.alpha_hat_torch.to(device=device, dtype=dtype)
        alpha_bar_t = self.alpha_bar_torch.to(device=device, dtype=dtype)

        beta_f = (1.0 - self.alpha_hat_torch_f).to(device=device, dtype=dtype)
        alpha_hat_f = self.alpha_hat_torch_f.to(device=device, dtype=dtype)

        sqrt_alpha_t = torch.sqrt(alpha_hat_t)
        inv_sqrt_alpha_t = 1.0 / sqrt_alpha_t
        sqrt_alpha_f = torch.sqrt(alpha_hat_f)
        inv_sqrt_alpha_f = 1.0 / sqrt_alpha_f

        sqrt_one_minus_abart = torch.sqrt(1.0 - alpha_bar_t)
        c_t = beta_t / sqrt_one_minus_abart

        sqrt_beta_f = torch.sqrt(beta_f)

        lam = self.lambda_mix
        sqrt_1m_lam = (1.0 - lam) ** 0.5

        G = self.noise_scaling(L, device=device)

        def init_prior(B, K, L):
            z_t = torch.randn(B, K, L, device=device, dtype=dtype)
            z_f = torch.randn(B, K, L, device=device, dtype=dtype)
            return (lam ** 0.5) * z_t + (sqrt_1m_lam) * self.f2t(G * z_f)

        imputed = torch.zeros(B, n_samples, K, L, device=device, dtype=dtype)

        T = self.num_steps
        for i in range(n_samples):
            x_t_cur = init_prior(B, K, L)

            for tt in reversed(range(T)):
                t_batch = torch.full((B,), tt, device=device, dtype=torch.long)

                # (1) time denoise
                inp_t = self.set_input_to_diffmodel(x_t_cur, observed_data, cond_mask)
                pred_t, _ = self.diffmodel(inp_t, side_info, t_batch)

                ct = c_t[tt].view(1, 1, 1)
                inv_sqrt_at = inv_sqrt_alpha_t[tt].view(1, 1, 1)
                x_time = (x_t_cur - ct * pred_t) * inv_sqrt_at

                # (2) frequency denoise
                inp_f = self.set_input_to_diffmodel(x_time, observed_data, cond_mask)
                _, pred_f = self.diffmodel(inp_f, side_info, t_batch)

                step_freq = sqrt_1m_lam * sqrt_beta_f[tt].view(1, 1, 1) * self.f2t(G * pred_f)
                inv_sqrt_af = inv_sqrt_alpha_f[tt].view(1, 1, 1)
                x_prev = (x_time - step_freq) * inv_sqrt_af

                # (3) optional stochasticity can be added here if needed

                x_t_cur = x_prev

            imputed[:, i] = x_t_cur

        return imputed

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)

        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
            for i in range(len(cut_length)):
                target_mask[i, ..., 0:cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


# ======================= Dataset Wrappers =======================

class CD2_PM25(CD2_base):
    def __init__(self, config, device, target_dim=36):
        super(CD2_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CD2_Physio(CD2_base):
    def __init__(self, config, device, target_dim=35):
        super(CD2_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data), device=self.device).long()
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CD2_Forecasting(CD2_base):
    def __init__(self, config, device, target_dim):
        super(CD2_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data), device=self.device).long()
        for_pattern_mask = observed_mask

        feature_id = torch.arange(self.target_dim_base, device=self.device).unsqueeze(0).expand(
            observed_data.shape[0], -1
        )

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id,
        )

    def sample_features(self, observed_data, observed_mask, feature_id, gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data, extracted_mask, extracted_feature_id, extracted_gt_mask = [], [], [], []
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k, ind[:size]])
            extracted_mask.append(observed_mask[k, ind[:size]])
            extracted_feature_id.append(feature_id[k, ind[:size]])
            extracted_gt_mask.append(gt_mask[k, ind[:size]])
        extracted_data = torch.stack(extracted_data, 0)
        extracted_mask = torch.stack(extracted_mask, 0)
        extracted_feature_id = torch.stack(extracted_feature_id, 0)
        extracted_gt_mask = torch.stack(extracted_gt_mask, 0)
        return extracted_data, extracted_mask, extracted_feature_id, extracted_gt_mask

    def get_side_info(self, observed_tp, cond_mask, feature_id=None):
        """
        Side information for forecasting:
        - If channel sub-sampling is used, pick embeddings by sampled feature_id; otherwise same as base class.
        """
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(torch.arange(self.target_dim, device=self.device))
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id,
        ) = self.process_data(batch)

        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask, feature_id, gt_mask = self.sample_features(
                observed_data, observed_mask, feature_id, gt_mask
            )
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else:
            cond_mask = self.get_test_pattern_mask(observed_mask, gt_mask)

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id,
        ) = self.process_data(batch)
        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1 - gt_mask)
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
        return samples, observed_data, target_mask, observed_mask, observed_tp
