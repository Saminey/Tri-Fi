import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding
from layers.GNN_Branch import PSR_GNN_Branch
from torch.utils.checkpoint import checkpoint
from layers.ScaleAdaptiveGraph import PriorResidualGraph

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model
        self.c_out = configs.c_out
        
        self.num_nodes = configs.enc_in
        
        # 1. 基础嵌入层
        self.intra_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.inter_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # ==========================================================
        # 🚀 降维打击核心：尺度自适应图生成器 (动态获取路径)
        # ==========================================================
        if not hasattr(configs, 'prior_path'):
            raise ValueError("configs 中找不到 prior_path！请确认 run.py 中已成功执行了自动建图。")

        self.graph_generator = PriorResidualGraph(
            num_nodes=self.num_nodes, 
            d_e=10,  
            prior_file_path=configs.prior_path  # 🚀 拔掉硬编码，动态挂载
        )

        # 2. 频域分支组件 (FreTS 风格的非线性映射)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.d_model, self.seq_len, kernel_size=(2, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(self.seq_len, self.d_model, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()
        )
        T_fft = self.seq_len // 2 + 1
        self.inter_projector = nn.Linear(T_fft, self.pred_len)
        self.inter_channel_projector = nn.Linear(self.d_model, self.c_out)
        
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
        
        # 5. 采用统一的门控权重矩阵
        self.fusion_gate = nn.Parameter(torch.zeros(3, self.c_out))
        self.dropout = nn.Dropout(configs.dropout)

    def inter_frequency(self, x):
        """
        频域特征提取：基于 Conv2D 的非线性跨域映射
        """
        B, M, D = x.shape 
        x_fft = torch.fft.rfft(x.permute(0, 2, 1), dim=-1) 
        x_fft_stacked = torch.stack([x_fft.real, x_fft.imag], dim=2) 
        y = x_fft_stacked.reshape(B, self.d_model, 2, -1)
        y = self.conv_layers(y) 
        output = self.inter_projector(y.squeeze(-2)) 
        return output.permute(0, 2, 1) 

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_chaos):
        # --- 数据科学修复：RevIN 归一化 (无损计算) ---
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        B, L, N = x_enc.shape
        # --- 分支 1：Intra (时域) ---
        intra_in = self.intra_embedding(x_enc, x_mark_enc)
        intra_out, _ = self.encoder(intra_in, attn_mask=None)
        # 💻 CS修复：删除危险的切片操作，利用张量流转原生的完美对齐 [Batch, pred_len, num_nodes]
        intra_out = self.intra_projector(intra_out).permute(0, 2, 1)[:, :, :N]

        # --- 分支 2：Inter (频域) ---
        inter_in = self.inter_embedding(x_enc, x_mark_enc)
        inter_out_dmodel = self.inter_frequency(inter_in)
        inter_out = self.inter_channel_projector(inter_out_dmodel)
        inter_out = inter_out[:, :, :N]
        # --- 分支 3：Chaos (拓扑域) ---
        A_final = self.graph_generator()
        chaos_out = self.psr_gnn_branch(x_chaos, A_final) # 同样删去切片，保证张量完备性
        chaos_out = chaos_out[:, :, :N]

        # --- 🚀 核心修复 2：自适应凸组合融合 (Adaptive Convex Fusion) ---
        # 使得 W_intra + W_inter + W_chaos = 1，绝对防止系统发散！
        fusion_weights = torch.softmax(self.fusion_gate, dim=0) # 形状: [3, c_out]
        
        co_out = self.dropout(
            fusion_weights[0] * intra_out + 
            fusion_weights[1] * inter_out + 
            fusion_weights[2] * chaos_out
        )

        # --- 💻 核心修复 3：零拷贝底层广播的反归一化 ---
        # 利用 PyTorch 底层 C++ 广播机制，直接用 [B, pred_len, N] 乘以 [B, 1, N]
        co_out = co_out * stdev + means
        
        return co_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_chaos_graph=None, mask=None):
        if self.task_name == 'long_term_forecast':
            if x_chaos_graph is None:
                raise ValueError("TriFi 架构在预测任务中必须传入 x_chaos_graph 张量")
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_chaos_graph)
        return None