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
        '''
        time schedulers
        '''
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
        self.alpha_hat_torch, self.alpha_bar_torch, self.sqrt_alpha_bar_torch, self.sqrt_one_minus_alpha_bar_torch = compute_alphas(
            self.beta_torch
        )
        self.alpha_hat_torch_f, self.alpha_bar_torch_f, self.sqrt_alpha_bar_torch_f, self.sqrt_one_minus_alpha_bar_torch_f = compute_alphas(
            self.beta_f_torch
        )
        # alpha_hat = 1.0 - beta                  
        # alpha_bar = np.cumprod(alpha_hat)       

        # self.beta_torch = torch.tensor(beta, dtype=torch.float32, device=self.device)           
        # self.alpha_hat_torch = torch.tensor(alpha_hat, dtype=torch.float32, device=self.device) 
        # self.alpha_bar_torch = torch.tensor(alpha_bar, dtype=torch.float32, device=self.device) 
        # self.sqrt_alpha_hat_torch = torch.sqrt(self.alpha_hat_torch)                            
        # self.sqrt_one_minus_alpha_bar_torch = torch.sqrt(1.0 - self.alpha_bar_torch)            
        '''
        freq schedulers
        '''
        
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
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)       # (B,L,Et)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)             # (B,L,K,Et)

        feature_embed = self.embed_layer(
            torch.arange(self.target_dim, device=self.device)
        )                                                                       # (K,Ef)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)  # (B,L,K,Ef)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)              # (B,L,K,Et+Ef)
        side_info = side_info.permute(0, 3, 2, 1)                               # (B,Et+Ef,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)                                   # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)                # (B,Et+Ef+1,K,L)
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

        alpha_bar_t = self.alpha_bar_torch     # (T,)
        alpha_hat_t = self.alpha_hat_torch 
        alpha_bar_f = self.alpha_bar_torch_f    # (T,)
        alpha_hat_f = self.alpha_hat_torch_f
        beta_t = self.beta_torch
        beta_f = 1.0 - alpha_hat_f
        
        noisy_x   = torch.zeros_like(observed_data)
        true_t    = torch.zeros_like(observed_data)
        true_f    = torch.zeros_like(observed_data)

        # ===== 前向合成：将噪声拆为时/频两路 =====
        for i in range(B):
            k = int(t[i].item())
            x0 = observed_data[i]                       # (K,L)
            sqrt_abk_t = torch.sqrt(alpha_bar_t[k])
            sqrt_abk_f = torch.sqrt(alpha_bar_f[k])

            noise_t = torch.zeros_like(x0)
            noise_f = torch.zeros_like(x0)

            # j = 0..k
            for j in range(0, k + 1):
                if j < k:
                    A_t = torch.sqrt(alpha_bar_t[k] / alpha_bar_t[j])
                else:
                    A_t = torch.tensor(1.0, device=self.device)

                coeff_t = torch.sqrt(1.0 - alpha_hat_t[j]) * A_t  # 可证离散系数
                eps_t = torch.randn(K, L, device = self.device)
                noise_t = noise_t + (sqrt_lam * coeff_t) * eps_t
                if j > 0:
                    A_t_extra = torch.sqrt(alpha_bar_t[k]/alpha_bar_t[j-1])
                else:
                    A_t_extra = torch.tensor(1.0, device=self.device, dtype=alpha_bar_t.dtype)
                
                if j < k:
                    A_f = torch.sqrt(alpha_bar_f[k] / alpha_bar_f[j])
                else:
                    A_f = torch.tensor(1.0, device = self.device)
                
                coeff_f = torch.sqrt(1.0 - alpha_hat_f[j]) * A_t_extra * A_f
                eps_f = torch.randn(K, L, device=self.device)

                noise_f = noise_f + (sqrt_1m_lam * coeff_f) * (self.f2t(G*eps_f.unsqueeze(0)).squeeze(0))

            xk = (sqrt_abk_t * sqrt_abk_f) * x0 + noise_t + noise_f
            noisy_x[i] = xk
            true_t[i] = noise_t
            true_f[i] = noise_f

        target_mask = (observed_mask - cond_mask).float()
        num_eval = target_mask.sum()
        denom = (num_eval if num_eval > 0 else 1.0)

        inp_t = self.set_input_to_diffmodel(noisy_x, observed_data, cond_mask)
        pred_t, _ = self.diffmodel(inp_t, side_info, t)     # (B,K,L)
        loss_time = (((true_t - pred_t) * target_mask) ** 2).sum()/denom

        # c_t = (self.beta_torch[t]/self.sqrt_one_minus_alpha_bar_torch[t]).view(B,1,1)
        sigma2 = torch.zeros(B, 1, 1, device=self.device, dtype=observed_data.dtype)
        for i in range(B):
            k = int(t[i].item())
            if k >= 0:
                s_idx = torch.arange(0, k + 1, device=self.device, dtype=torch.long)  # 0..k

                # 时域链：ᾱ_k^t / ᾱ_{s-1}^t，s=0 时分母取 1（即 ᾱ_0^t=1）
                # 构造 ᾱ_{s-1}^t
                alpha_bar_t_sminus1 = torch.ones_like(s_idx, dtype=alpha_bar_t.dtype, device=self.device)
                if k >= 1:
                    # 对 s>0 的位置填入 ᾱ_{s-1}^t
                    pos = s_idx > 0
                    alpha_bar_t_sminus1[pos] = alpha_bar_t[s_idx[pos] - 1]
                chain_t = alpha_bar_t[k] / alpha_bar_t_sminus1  # (k+1,)

                # 频域链：ᾱ_k^f / ᾱ_s^f，s=k 时取 1
                alpha_bar_f_s = torch.ones_like(s_idx, dtype=alpha_bar_f.dtype, device=self.device)
                if k >= 1:
                    pos2 = s_idx < k
                    alpha_bar_f_s[pos2] = alpha_bar_f[s_idx[pos2]]  # s<k 用 ᾱ_s^f，s=k 保持 1
                chain_f = alpha_bar_f[k] / alpha_bar_f_s          # (k+1,)

                sb_f = 1.0 - alpha_hat_f[s_idx]                   # β_s^f
                w2 = ( (sqrt_1m_lam ** 2)
                    * sb_f
                    * chain_t
                    * chain_f )                                # 注意：这里已经是 w_{k,s}^2（因为 sqrt项全平方）
                sigma2[i, 0, 0] = w2.sum()

        sigma_f = torch.sqrt(sigma2)      # (B,1,1)

        c_t = (self.beta_torch[t] / torch.sqrt((1.0 - self.alpha_bar_torch[t]))).view(B, 1, 1)
        inv_sqrt_alpha_t = (1.0 / torch.sqrt(self.alpha_hat_torch[t])).view(B, 1, 1)
        x_t_time = (noisy_x - c_t * pred_t.detach()) * inv_sqrt_alpha_t

        inp_f = self.set_input_to_diffmodel(x_t_time, observed_data, cond_mask)
        _, pred_f = self.diffmodel(inp_f, side_info, t)     # (B,K,L) —— 预测标准噪声 \tilde{ε}^f

            # 用 σ_f * F^{-1}(G * pred_f) 对齐 true_f
        pred_f_time = sigma_f * self.f2t(G * pred_f)        # (B,K,L)
        loss_freq = (((true_f - pred_f_time) * target_mask) ** 2).sum() / denom

            # ===== 一致性损失（总噪声） =====
        recon_tot = pred_t + pred_f_time
        true_tot  = true_t + true_f
        loss_consistency = (((true_tot - recon_tot) * target_mask) ** 2).sum() / denom

            # ===== 总损失 =====
        loss = loss_time + loss_freq + self.aux_branch_weight * loss_consistency
        return loss
    
    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        return total_input
    
    @torch.no_grad()
    def impute(self, observed_data, cond_mask, side_info, n_samples: int):
        """
        采样/插补（方案B对齐版）：
        反向每步：先“时域”去噪，再“频域”去噪。
        - 时域：x_t_time = (x_t - c_t * pred_t) / sqrt(alpha_t)                 # 经典 DDPM
        - 频域：x_{t-1}  = (x_t_time - sqrt(1-λ)*sqrt(β_f) * F^{-1}(G*pred_f)) / sqrt(alpha_f)
        其中 pred_t 预测时域噪声（你当前训练目标），pred_f 预测标准噪声（N(0,I)）。
        """
        B, K, L = observed_data.shape
        device = self.device
        dtype = observed_data.dtype

        # ===== 预计算/调度量 =====
        beta_t      = self.beta_torch.to(device=device, dtype=dtype)           # (T,)
        alpha_hat_t = self.alpha_hat_torch.to(device=device, dtype=dtype)      # (T,)  单步 α_t
        alpha_bar_t = self.alpha_bar_torch.to(device=device, dtype=dtype)      # (T,)  累计 ᾱ_t

        beta_f      = (1.0 - self.alpha_hat_torch_f).to(device=device, dtype=dtype)  # (T,)  单步 β_f
        alpha_hat_f = self.alpha_hat_torch_f.to(device=device, dtype=dtype)           # (T,)  单步 α_f
        # alpha_bar_f 仅用于可选的随机项（后验方差），这里不需要也可以不取

        sqrt_alpha_t    = torch.sqrt(alpha_hat_t)        # (T,)
        inv_sqrt_alpha_t= 1.0 / sqrt_alpha_t
        sqrt_alpha_f    = torch.sqrt(alpha_hat_f)        # (T,)
        inv_sqrt_alpha_f= 1.0 / sqrt_alpha_f

        sqrt_one_minus_abart = torch.sqrt(1.0 - alpha_bar_t)  # (T,)
        c_t = beta_t / sqrt_one_minus_abart  # (T,)

        sqrt_beta_f = torch.sqrt(beta_f)  # (T,)

        lam = self.lambda_mix
        sqrt_1m_lam = (1.0 - lam) ** 0.5

        # 频域缩放（与 f2t 配套；需保证能量配平的形状与 L 对齐）
        G = self.noise_scaling(L, device=device)

        # ===== 初始先验：与训练前向保持一致（频+时的混合高斯）=====
        def init_prior(B, K, L):
            z_t = torch.randn(B, K, L, device=device, dtype=dtype)
            z_f = torch.randn(B, K, L, device=device, dtype=dtype)
            return (lam ** 0.5) * z_t + (sqrt_1m_lam) * self.f2t(G * z_f)

        # ===== 输出容器 =====
        imputed = torch.zeros(B, n_samples, K, L, device=device, dtype=dtype)

        # ===== 采样循环 =====
        T = self.num_steps
        for i in range(n_samples):
            x_t_cur = init_prior(B, K, L)  # x_T

            for tt in reversed(range(T)):  # T-1, ..., 0
                t_batch = torch.full((B,), tt, device=device, dtype=torch.long)

                # ---- (1) 时域去噪（与 calc_loss 中 c_t / inv_sqrt_alpha_t 一致）----
                inp_t = self.set_input_to_diffmodel(x_t_cur, observed_data, cond_mask)
                pred_t, _ = self.diffmodel(inp_t, side_info, t_batch)  # (B,K,L)，预测时域噪声

                ct = c_t[tt].view(1, 1, 1)                   # 标量 → (1,1,1)
                inv_sqrt_at = inv_sqrt_alpha_t[tt].view(1,1,1)
                x_time = (x_t_cur - ct * pred_t) * inv_sqrt_at   # 约等于 𝓕^{-1}(x_t^f)

                # ---- (2) 频域去噪（本步频域项；pred_f 作为 N(0,I)）----
                inp_f = self.set_input_to_diffmodel(x_time, observed_data, cond_mask)
                _, pred_f = self.diffmodel(inp_f, side_info, t_batch)  # (B,K,L)，预测标准噪声

                step_freq = sqrt_1m_lam * sqrt_beta_f[tt].view(1,1,1) * self.f2t(G * pred_f)  # 本步频域噪声（时域贡献）
                inv_sqrt_af = inv_sqrt_alpha_f[tt].view(1,1,1)
                x_prev = (x_time - step_freq) * inv_sqrt_af

                # ---- (3) 可选：加入后验方差的随机项（若需要随机采样）----
                # 经典 DDPM 的随机项是 N(0, β̃_t I)。在本模型中，你可以分别在两步加入：
                #   - 时域随机项：eta_t * z_t，系数可取 sqrt(β̃_t^time)
                #   - 频域随机项：eta_f * z_f，系数可取 sqrt(β̃_t^freq) 的“时域贡献”
                # 为了简洁，默认 determinisitc predictor；若要加噪，请在这里加：
                # if tt > 0:
                #     z = torch.randn_like(x_prev)
                #     x_prev = x_prev + sigma_eff[tt].view(1,1,1) * z

                x_t_cur = x_prev

            imputed[:, i] = x_t_cur  # x_0

        return imputed

    # @torch.no_grad()
    # def impute(self, observed_data, cond_mask, side_info, n_samples):
        
    #     B, K, L = observed_data.shape
    #     G = self.noise_scaling(L, device=self.device)

    #     lam = self.lambda_mix
    #     sqrt_lam = lam ** 0.5
    #     sqrt_1m_lam = (1.0 - lam) ** 0.5

    #     def init_prior(B, K, L):
    #         return (
    #             sqrt_lam    * torch.randn(B, K, L, device=self.device) +
    #             sqrt_1m_lam * self.f2t(G * torch.randn(B, K, L, device=self.device))
    #         )

    #     imputed_samples = torch.zeros(B, n_samples, K, L, device=self.device)

    #     for i in range(n_samples):
    #         x_t = init_prior(B, K, L)
    #         #x_t = torch.randn(B, K, L, device=self.device)

    #         for t in range(self.num_steps - 1, -1, -1):
    #             t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)

    #             c_t = self.beta_torch[t] / (self.sqrt_one_minus_alpha_bar_torch[t])
    #             c_t_f = self.beta_f_torch[t] / (self.sqrt_one_minus_alpha_bar_torch_f[t])
    #             inv_sqrt_alpha_hat = 1.0 / torch.sqrt(self.alpha_hat_torch[t])

    #             # (1) 先“时域”去噪
    #             model_input_t = self.set_input_to_diffmodel(x_t, observed_data, cond_mask)
    #             pred_t, _ = self.diffmodel(model_input_t, side_info, t_batch)      # (B,K,L)
    #             x_t_time = x_t - c_t * pred_t

    #             # (2) 再“频域”去噪
    #             c_t_f = self.beta_f_torch[t] / (self.sqrt_one_minus_alpha_bar_torch_f[t])
    #             model_input_f = self.set_input_to_diffmodel(x_t_time, observed_data, cond_mask)
    #             _, pred_f = self.diffmodel(model_input_f, side_info, t_batch)       # (B,K,L)
    #             mean = inv_sqrt_alpha_hat * (x_t_time - c_t_f * self.f2t(G * pred_f))

    #             if t > 0:
    #                 # 方差
    #                 sigma_t_t = torch.sqrt(
    #                     self.beta_torch[t] * (1.0 - self.alpha_bar_torch[t - 1]) / (1.0 - self.alpha_bar_torch[t] )
    #                 )
    #                 sigma_t_f = torch.sqrt(
    #                     self.beta_f_torch[t] * (1.0 - self.alpha_bar_torch_f[t - 1]) / (1.0 - self.alpha_bar_torch_f[t])
    #                 )
    #                 # (3) 加噪顺序：先频 -> 后时
    #                 sigma_target = sigma_t_t
    #                 var_mix = (1-lam) * sigma_t_f ** 2 + lam * sigma_t_t ** 2
    #                 rho_t = sigma_target/(torch.sqrt(var_mix))
    #                 z_f = torch.randn_like(x_t)
    #                 z_tn = torch.randn_like(x_t)
    #                 x_tmp = mean + rho_t * sqrt_1m_lam * (sigma_t_f *  self.f2t(G * z_f))  # 先“频”
    #                 x_t = x_tmp +  rho_t * (sqrt_lam * (sigma_t_t * z_tn))               # 后“时”
    #             else:
    #                 x_t = mean

    #         imputed_samples[:, i] = x_t.detach()

    #     return imputed_samples
    # # ====================== 训练/评估入口 ======================
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
            for i in range(len(cut_length)):  # 避免重复评价
                target_mask[i, ..., 0:cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


# ======================= 数据集包装类 =======================

class CD2_PM25(CD2_base):
    def __init__(self, config, device, target_dim=36):
        super(CD2_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp   = batch["timepoints"].to(self.device).float()
        gt_mask       = batch["gt_mask"].to(self.device).float()
        cut_length    = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask       = gt_mask.permute(0, 2, 1)
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
        observed_tp   = batch["timepoints"].to(self.device).float()
        gt_mask       = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask       = gt_mask.permute(0, 2, 1)

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
        observed_tp   = batch["timepoints"].to(self.device).float()
        gt_mask       = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask       = gt_mask.permute(0, 2, 1)

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
        Forecasting 场景下的 side_info：
        - 若按通道子采样，则用被采样的 feature_id 去取 embedding；否则与基类一致。
        """
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,Et)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim, device=self.device)
            )  # (K,Ef)
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
        else:  # test pattern
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
