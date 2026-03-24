import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, global_mean_pool, knn_graph

class ResGATGIN_Encoder(nn.Module):
    """
    [生产版] ResGATGIN: 移除调试输出，专注于高性能特征聚合
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.1):
        super(ResGATGIN_Encoder, self).__init__()
        self.dropout_ratio = dropout

        # GAT 层：多头注意力特征过滤
        self.gat_conv = GATConv(in_channels, hidden_channels, heads=heads, 
                                dropout=dropout, concat=False, add_self_loops=True)
        self.gat_bn = nn.BatchNorm1d(hidden_channels)

        # GIN 层：拓扑结构同构映射
        gin_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels), 
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.gin_conv = GINConv(gin_mlp, train_eps=True)

        # 残差投影对齐
        self.res_projector = nn.Linear(hidden_channels, out_channels) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index):
        # 第一阶段：注意力计算与标准化
        x_gat = F.relu(self.gat_bn(self.gat_conv(x, edge_index)))
        x_gat = F.dropout(x_gat, p=self.dropout_ratio, training=self.training)

        # 第二阶段：GIN 聚合与残差相加
        x_res = self.res_projector(x_gat)
        x_gin = self.gin_conv(x_gat, edge_index)
        
        return F.relu(x_gin + x_res)

class PSR_GNN_Branch(nn.Module):
    """
    [生产版] 相空间重构 GNN 分支：全流程静默，支持 GPU 动态构图
    """
    def __init__(self, configs, m_dim, k=3):
        super(PSR_GNN_Branch, self).__init__()
        
        gnn_hidden_dim = configs.d_model // 2 
        gnn_out_dim = configs.d_model         
        self.k = k
        
        self.gnn_encoder = ResGATGIN_Encoder(
            in_channels=m_dim,          
            hidden_channels=gnn_hidden_dim,
            out_channels=gnn_out_dim,
            dropout=configs.dropout
        )
        
        # 映射至预测维度: [Batch, Pred_Len * M]
        self.final_projector = nn.Linear(gnn_out_dim, configs.pred_len * configs.enc_in)
        self.configs = configs 

    def forward(self, x_chaos):
        """
        输入: x_chaos [Batch, valid_len, M, m_dim]
        """
        B, L, M, D = x_chaos.shape
        
        # 1. 节点展平与 Batch 索引生成 (跨变量拓扑模式)
        x_flat = x_chaos.reshape(B * L * M, D).float()
        batch_idx = torch.arange(B, device=x_chaos.device).repeat_interleave(L * M)

        # 2. GPU 动态构图 (计算 $k\text{-NN}$ 边)
        edge_index = knn_graph(x_flat, k=self.k, batch=batch_idx, loop=False)

        # 3. 特征提取与全局池化
        node_feats = self.gnn_encoder(x_flat, edge_index)
        graph_feats = global_mean_pool(node_feats, batch_idx)

        # 4. 最终投影与对齐
        projected = self.final_projector(graph_feats)
        output = projected.view(B, self.configs.pred_len, self.configs.enc_in)
        
        return output