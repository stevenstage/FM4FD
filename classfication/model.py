from torch import nn
import torch
import torch.nn.functional as F
from loss import AutomaticWeightedLoss
from utils.tools import FFT_sim, generate_CLLabels
from utils.augmentations import augment_positive_test

class ComplexLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.Linear_real = torch.nn.Linear(in_features, out_features, bias=bias)
        self.Linear_img = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        real_real = self.Linear_real(input.real)
        img_real = self.Linear_img(input.real)
        real_img = self.Linear_real(input.imag)
        img_img = self.Linear_img(input.imag)
        return real_real - img_img + 1j * (real_img + img_real)


class Complex_Dropout(nn.Module):
    def __init__(self, p, inplace=False, size=None, device='cuda'):
        super().__init__()
        self.size = size
        self.device = device
        if self.size is not None:
            self.ones = torch.ones(size)
            if self.device is not None:
                self.ones = self.ones.to(self.device)
        self.real_dropout = nn.Dropout(p=p, inplace=inplace)

    def forward(self, input):
        if self.size is not None:
            return input * self.real_dropout(self.ones)
        else:
            if self.device is not None:
                return input * self.real_dropout(torch.ones(input.size()).to(self.device))
            return input * self.real_dropout(torch.ones(input.size()))


class FreNormLaryer_KB(nn.Module):
    def __init__(self, n_knlg, input_len, bias=True):
        super(FreNormLaryer_KB, self).__init__()
        self.embed_dim = input_len // 2 + 1
        self.n_knlg = n_knlg
        self.kb = nn.Parameter(torch.randn(n_knlg, self.embed_dim, dtype=torch.cfloat))
        self.scaling = self.embed_dim ** -0.5
        self.in_proj_q = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
        self.in_proj_k = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
        self.in_proj_v = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_dim = input_len

    def retrive_w(self, x):
        q = self.in_proj_q(x)
        k = self.in_proj_k(self.kb)
        v = self.in_proj_v(self.kb)

        attn_weights = torch.matmul(q, torch.conj_physical(k).T) * self.scaling
        real = torch.real(attn_weights)
        attn_weights = F.softmax(real, dim=-1).type(torch.complex64)
        w = torch.matmul(attn_weights, v)
        return w

    def forward(self, x):
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        w = self.retrive_w(x)
        y = x * w
        out = torch.fft.irfft(y, n=self.out_dim, dim=-1, norm="ortho")
        return out, w


class TFC_Encoder(nn.Module):
    def __init__(self, configs, args):
        super(TFC_Encoder, self).__init__()
        self.training_mode = args.training_mode

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.onelayer_out_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.onelayer_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.onelayer_out_channels, configs.twolayer_out_channels, kernel_size=configs.conv2_kernel_size, stride=1, bias=False, padding=configs.conv2_kernel_size // 2),
            nn.BatchNorm1d(configs.twolayer_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.twolayer_out_channels, configs.final_out_channels, kernel_size=configs.conv3_kernel_size, stride=1, bias=False, padding=configs.conv3_kernel_size // 2),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.dense = nn.Sequential(
            nn.Linear(configs.hidden_dimension, configs.d_ff),
            nn.BatchNorm1d(configs.d_ff),
            nn.ReLU(),
            nn.Linear(configs.d_ff, 128)
        )

    def forward(self, x_in_t):
        x = self.conv_block1(x_in_t)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        h = x.reshape(x.shape[0], -1)
        z = self.dense(h)

        return h, z


class target_classifier(nn.Module):  # Classification head
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(configs.hidden_dimension, 64)
        self.dropout = nn.Dropout(configs.ft_dropout)  # Dropout with a probability of 0.5
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        """2-layer MLP"""
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        emb = self.dropout(emb)
        pred = self.logits_simple(emb)
        return pred


class ChannelAttention(nn.Module):
    def __init__(self, n_channels=None, reduction=4):
        super(ChannelAttention, self).__init__()
        self.reduction = reduction
        self.n_channels = n_channels

    def forward(self, x):
        bs, feature_dim, n_channels = x.shape
        y = torch.mean(x, dim=1)  # [bs, n_channels]
        
        if not hasattr(self, 'fc') or self.fc[0].in_features != n_channels:
            hidden_dim = max(n_channels // self.reduction, 1)
            self.fc = nn.Sequential(
                nn.Linear(n_channels, hidden_dim, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, n_channels, bias=False),
                nn.Sigmoid()
            ).to(x.device)
            print(f"🔧 Dynamically created ChannelAttention.fc for n_channels={n_channels}")
        
        y = self.fc(y)  # [bs, n_channels]
        y = y.view(bs, 1, n_channels)  # [bs, 1, n_channels]
        return x * y  # [bs, feature_dim, n_channels]

    def get_channel_weights(self, device='cpu'):
        if not hasattr(self, 'fc') or len(self.fc) < 3:
            return None
            
        try:
            last_linear_weight = self.fc[2].weight  # [n_channels, hidden_dim]
            channel_scores = torch.mean(last_linear_weight, dim=1)  # [n_channels]
            channel_weights = torch.sigmoid(channel_scores)  # [n_channels]
            return channel_weights.detach().cpu().numpy()
        except:
            try:
                n_channels = self.fc[0].in_features
                dummy_input = torch.zeros(1, n_channels).to(device)
                y = self.fc(dummy_input)
                return y.squeeze(0).detach().cpu().numpy()
            except:
                return None


class Model(nn.Module):
    def __init__(self, configs, args):
        super(Model, self).__init__()
        self.args = args
        self.configs = configs
        self.fre_norm_encoder = FreNormLaryer_KB(self.configs.n_knlg, self.configs.TSlength_aligned)
        self.encoder = TFC_Encoder(configs, args)
        self.head = nn.Linear(configs.hidden_dimension, 512)
        self.labels_cl = None
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.awl = AutomaticWeightedLoss(2)
        self.mse = torch.nn.MSELoss()

        # ✅ 新增：通道注意力模块（分别用于 h 和 z）
        self.channel_attn_h = ChannelAttention(configs.input_channels, reduction=4)
        self.channel_attn_z = ChannelAttention(configs.input_channels, reduction=4)

    def pretrainWithContrast(self, batch_x):
        bs, seq_len, n_vars = batch_x.shape
        x_raw = batch_x.permute(0, 2, 1)
        sim_matrix = FFT_sim(x_raw)
        x_raw = x_raw.reshape(-1, seq_len)
        negative_index = torch.topk(sim_matrix, k=self.configs.negative_nums, dim=1).indices
        x_reformed, _ = self.fre_norm_encoder(x_raw)
        x_positives = augment_positive_test(x_reformed, self.configs.masking_ratio, self.configs.lm,
                                            k=self.configs.positive_nums)
        x_positives = x_positives.reshape(-1, seq_len)
        x_all = torch.cat([x_raw, x_positives], dim=0)
        h, z = self.encoder(x_all.unsqueeze(1))

        s_enc_out_norm = F.normalize(z, dim=1)
        s_q = s_enc_out_norm[: bs * n_vars]
        s_k = s_enc_out_norm[bs * n_vars:].reshape(bs * n_vars, self.configs.positive_nums, -1)
        if self.labels_cl is None:
            self.labels_cl = generate_CLLabels(x_raw, self.configs.positive_nums, self.configs.negative_nums)
        if self.configs.positive_nums == 1:
            positive_similarity_matrix = torch.matmul(s_q.unsqueeze(1), s_k.permute(0, 2, 1)).squeeze(-1)
        else:
            positive_similarity_matrix = torch.matmul(s_q.unsqueeze(1), s_k.permute(0, 2, 1)).squeeze()
        if self.configs.negative_nums == 1:
            negative_similarity_matrix = torch.matmul(s_q.unsqueeze(1),
                                                      s_k[:, 0, :][negative_index].permute(0, 2, 1)).squeeze(-1)
        else:
            negative_similarity_matrix = torch.matmul(s_q.unsqueeze(1),
                                                      s_k[:, 0, :][negative_index].permute(0, 2, 1)).squeeze()
        similarity_matrix = torch.cat([positive_similarity_matrix, negative_similarity_matrix], dim=-1)
        similarity_normed = self.log_softmax(similarity_matrix / self.args.temperature)

        loss_cl = self.kl(similarity_normed, self.labels_cl)

        positive_enc_out = h[bs * n_vars:].reshape(bs * n_vars, self.configs.positive_nums, -1)
        hardNegative_enc_out = positive_enc_out[:, 0, :][negative_index]
        similarity_matrix /= self.args.temperature
        rebuild_weight_matrix = self.softmax(similarity_matrix)
        rebuild_embed = torch.matmul(rebuild_weight_matrix[:, :self.configs.positive_nums].unsqueeze(1),
                                     positive_enc_out) + torch.matmul(
            rebuild_weight_matrix[:, self.configs.positive_nums:].unsqueeze(1), hardNegative_enc_out)
        pred_x = self.head(rebuild_embed)
        pred_x = pred_x.reshape(bs, n_vars, -1).permute(0, 2, 1)
        loss_rb = self.mse(batch_x, pred_x)
        loss = self.awl(loss_cl, loss_rb)
        return loss, loss_cl, loss_rb

    def clf(self, batch_x):
        bs, seq_len, n_vars = batch_x.shape
        x_raw = batch_x.permute(0, 2, 1)  # [bs, n_vars, seq_len]
        x_raw = x_raw.reshape(-1, seq_len)  # [bs * n_vars, seq_len]
        x_raw = x_raw.unsqueeze(1)  # [bs * n_vars, 1, seq_len]
        
        enc_out, z = self.encoder(x_raw)  # [bs * n_vars, feature_dim]
        
        # Reshape to [bs, n_vars, feature_dim]
        feature_dim_h = enc_out.shape[1]
        feature_dim_z = z.shape[1]
        enc_out = enc_out.reshape(bs, n_vars, feature_dim_h)  # [bs, n_vars, feature_dim_h]
        z = z.reshape(bs, n_vars, feature_dim_z)              # [bs, n_vars, feature_dim_z]
        
        enc_out = enc_out.permute(0, 2, 1)  # [bs, feature_dim_h, n_vars]
        enc_out = self.channel_attn_h(enc_out)  # [bs, feature_dim_h, n_vars]
        enc_out = torch.mean(enc_out, dim=2)    # [bs, feature_dim_h] ← 保留所有通道信息
        
        z = z.permute(0, 2, 1)  # [bs, feature_dim_z, n_vars]
        z = self.channel_attn_z(z)  # [bs, feature_dim_z, n_vars]
        z = torch.mean(z, dim=2)    # [bs, feature_dim_z]
        
        return enc_out, z

    def forward(self, batch_x, pretrain=False):
        batch_x = batch_x.transpose(-1, -2)
        if pretrain:
            return self.pretrainWithContrast(batch_x)
        else:
            return self.clf(batch_x)
