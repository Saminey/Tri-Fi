import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class Dense_ResGATGIN_Encoder(nn.Module):
    """
    基于稠密张量的 ResGATGIN 单元
    数学逻辑：
    1. GAT: $\alpha_{ij} = \text{Softmax}(\text{LeakyReLU}(\vec{a}^T [h_i || h_j])) \cdot A_{prior}$
    2. GIN: $h_i^{(l)} = \text{MLP}((1+\epsilon)h_i^{(l-1)} + \sum A_{ij} h_j^{(l-1)})$
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super(Dense_ResGATGIN_Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.feature_proj = nn.Linear(in_channels, hidden_channels)

        # GAT 参数
        self.a_src = nn.Linear(hidden_channels, 1, bias=False)
        self.a_dst = nn.Linear(hidden_channels, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gat_ln = nn.LayerNorm(hidden_channels)

        # GIN 参数
        self.eps = nn.Parameter(torch.zeros(1))
        self.gin_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels), 
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

        self.res_projector = nn.Linear(hidden_channels, out_channels) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x, A_final):
        B, M, _ = x.shape
        h = self.feature_proj(x) # [B, M, H]

        # --- 1. 稠密 GAT 注意力机制 ---
        src_scores = self.a_src(h) # [B, M, 1]
        dst_scores = self.a_dst(h).transpose(1, 2) # [B, 1, M]
        attn_scores = self.leaky_relu(src_scores + dst_scores) # [B, M, M]

        # 🚀 核心修复：解决 AMP/FP16 溢出问题
        # 自动探测当前 Tensor 精度（FP16 最小值为 -65504）
        fill_value = torch.finfo(attn_scores.dtype).min
        
        # 拓扑遮蔽：只允许在 A_final > 0 的地方进行消息传递
        zero_mask = (A_final <= 0).unsqueeze(0).expand(B, M, M)
        attn_scores = attn_scores.masked_fill(zero_mask, fill_value) 
        
        attention_weights = F.softmax(attn_scores, dim=-1)

        # 融合物理强度 $A_{final}$ 到注意力权重中，打通梯度流
        weighted_adj = attention_weights * A_final.unsqueeze(0) 

        # 聚合：$H_{gat} = \text{Attn} \cdot H$
        h_gat = torch.bmm(weighted_adj, h) 
        h_gat = F.relu(self.gat_ln(h_gat))
        h_gat = self.dropout(h_gat)

        # --- 2. 稠密 GIN 同构聚合 ---
        # 优化显存：直接利用 batch 广播机制
        # $H_{agg} = A_{final} \cdot H_{gat}$
        h_agg = torch.matmul(A_final, h_gat) 
        h_gin_in = (1.0 + self.eps) * h_gat + h_agg
        
        x_gin = self.gin_mlp(h_gin_in)
        x_res = self.res_projector(h_gat)
        
        return F.relu(x_gin + x_res)


class PSR_GNN_Branch(nn.Module):
    """
    PSR-GNN 分支：处理相空间重构后的节点特征与动力学拓扑
    """
    def __init__(self, configs, m_dim):
        super(PSR_GNN_Branch, self).__init__()
        self.configs = configs 
        
        gnn_hidden_dim = configs.d_model // 2 
        gnn_out_dim = configs.d_model         
        
        # 动态计算经过 PSR 重构后的实际特征长度
        tau = getattr(configs, 'tau', 1)
        effective_len = configs.seq_len - (m_dim - 1) * tau
        in_feats = effective_len * m_dim
        
        self.gnn_encoder = Dense_ResGATGIN_Encoder(
            in_channels=in_feats,          
            hidden_channels=gnn_hidden_dim,
            out_channels=gnn_out_dim,
            dropout=configs.dropout
        )
        
        self.final_projector = nn.Linear(gnn_out_dim, configs.pred_len)

    def forward(self, x_chaos, A_final):
        """
        x_chaos: [Batch, L_eff, M, m_dim]
        A_final: [M, M]
        """
        B, L, M, D = x_chaos.shape
        
        # 重构展平：将动力学分量 D 与有效时间步 L 融合为节点特征 [B, M, L*D]
        x_dense = x_chaos.transpose(1, 2).contiguous().view(B, M, L * D)

        # 梯度检查点，节省显存（RTX 6000 虽大，但 862 节点仍需节约）
        if getattr(self.configs, 'is_training', True) and self.training:
            if not x_dense.requires_grad:
                x_dense.requires_grad_(True)
            node_feats = checkpoint(self.gnn_encoder, x_dense, A_final, use_reentrant=False)
        else:
            node_feats = self.gnn_encoder(x_dense, A_final)

        # 映射至预测长度
        projected = self.final_projector(node_feats) # [B, M, pred_len]
        
        # 变换为标准预测输出形状 [B, pred_len, M]
        return projected.transpose(1, 2)