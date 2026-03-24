import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding
from layers.GNN_Branch import PSR_GNN_Branch

class Model(nn.Module):
    """
    [Tri-Fi 生产版] 
    集成时域 (Intra)、频域 (Inter) 与拓扑域 (Chaos) 的大一统时序预测架构。
    优化点：移除 I/O 阻塞打印，修复张量属性访问，固化频域投影层，引入可学习融合权重。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model
        
        # 1. 基础嵌入层
        self.intra_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.inter_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # 2. 频域分支组件
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.d_model, self.seq_len, kernel_size=(2, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(self.seq_len, self.d_model, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()
        )
        # 固化投影层，确保参数可学习且不在 forward 中重复初始化
        T_fft = self.seq_len // 2 + 1
        self.inter_projector = nn.Linear(T_fft, self.pred_len)

        # 3. 时域分支组件 (Inverted Transformer Encoder)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.intra_projector = nn.Linear(configs.d_model, configs.pred_len)

        # 4. 拓扑动力学分支 (GNN)
        self.m_dim = getattr(configs, 'm_dim', 3) 
        self.psr_gnn_branch = PSR_GNN_Branch(configs, self.m_dim)
        
        # 5. 🚀 核心新增：可学习的三维融合权重 (Learnable Fusion Weights)
        # 初始化为 1.0，保证训练初期的梯度等效于原来的直接相加
        self.W_intra = nn.Parameter(torch.tensor(1.0))
        self.W_inter = nn.Parameter(torch.tensor(1.0))
        self.W_chaos = nn.Parameter(torch.tensor(1.0))
        
        self.dropout = nn.Dropout(configs.dropout)

    def inter_frequency(self, x):
        """
        频域特征提取：FFT -> Conv2D -> IFFT -> Projection
        """
        B, M, D = x.shape 
        x_fft = torch.fft.rfft(x.permute(0, 2, 1), dim=-1) 
        x_fft_stacked = torch.stack([x_fft.real, x_fft.imag], dim=2) 
        y = x_fft_stacked.reshape(B, self.d_model, 2, -1)
        y = self.conv_layers(y) 
        output = self.inter_projector(y.squeeze(-2)) 
        return output.permute(0, 2, 1) 

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_chaos):
        # 归一化处理
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # --- 分支 1：Intra (时域) ---
        intra_in = self.intra_embedding(x_enc, x_mark_enc)
        intra_out, _ = self.encoder(intra_in, attn_mask=None)
        intra_out = self.intra_projector(intra_out).permute(0, 2, 1)[:, :, :N]

        # --- 分支 2：Inter (频域) ---
        inter_in = self.inter_embedding(x_enc, x_mark_enc)
        inter_out = self.inter_frequency(inter_in)[:, :, :N]

        # --- 分支 3：Chaos (拓扑域) ---
        chaos_out = self.psr_gnn_branch(x_chaos)[:, :, :N]

        # --- 🚀 最终自适应融合 (Adaptive Fusion) ---
        # 利用 nn.Parameter 动态调节各分支的强度
        co_out = self.dropout(
            self.W_intra * intra_out + 
            self.W_inter * inter_out + 
            self.W_chaos * chaos_out
        )

        # 反归一化
        co_out = co_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        co_out = co_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return co_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_chaos_graph=None, mask=None):
        if self.task_name == 'long_term_forecast':
            if x_chaos_graph is None:
                raise ValueError("TriFi 架构在预测任务中必须传入 x_chaos_graph 张量")
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_chaos_graph)
        return None