import torch
import torch.nn as nn
import numpy as np

class PriorResidualGraph(nn.Module):
    def __init__(self, num_nodes, d_e=10, prior_file_path=None, threshold_val=0.02):
        """
        尺度自适应拓扑生成器 (Scale-Adaptive Topology Generator)
        """
        super(PriorResidualGraph, self).__init__()
        self.num_nodes = num_nodes
        
        # 1. 物理先验图 A_prior (宏观稳态)
        if prior_file_path is not None:
            print(f"✅ [Graph] 成功加载静态物理先验图: {prior_file_path}")
            A_prior_np = np.load(prior_file_path)
            self.register_buffer('A_prior', torch.tensor(A_prior_np, dtype=torch.float32))
        else:
            print("⚠️ [Graph] 警告: 未提供先验图，退化为全数据驱动残差图！")
            self.register_buffer('A_prior', torch.zeros((num_nodes, num_nodes), dtype=torch.float32))

        # 强制添加自环 (Self-loop)
        self.register_buffer('I', torch.eye(num_nodes, dtype=torch.float32))

        # 2. 低秩可学习残差字典 E1, E2 (微观扰动)
        self.E1 = nn.Parameter(torch.empty(num_nodes, d_e))
        self.E2 = nn.Parameter(torch.empty(num_nodes, d_e))
        
        # 【核心修复 1】：摒弃 Xavier，采用具备更大初始方差的 Normal 分布
        # 保证 E1*E2^T 初始时有足够的激活值能越过阈值，避免“梯度死亡”
        nn.init.normal_(self.E1, mean=0.0, std=0.2)
        nn.init.normal_(self.E2, mean=0.0, std=0.2)
        
        # 3. 自适应控制因子 alpha (未激活的原始 logits)
        # 初始化为 -2.0左右，经过 sigmoid 后约为 0.1
        self.alpha_logits = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        
        # 4. 稀疏化阈值
        self.register_buffer('threshold', torch.tensor(threshold_val, dtype=torch.float32))

    def forward(self):
        """
        前向传播生成最终的时空拓扑结构 A_final
        返回: A_final (形状: [N, N])
        """
        # 计算基础低秩残差矩阵 A_res = E1 * E2^T
        A_res = torch.matmul(self.E1, self.E2.transpose(0, 1))
        raw_residual = torch.tanh(A_res)
        
        # 严格连续软阈值操作 (Soft Thresholding)
        sparse_residual = torch.sign(raw_residual) * torch.relu(torch.abs(raw_residual) - self.threshold)
        
        # 【核心修复 2】：通过 Sigmoid 施加物理约束
        # 限制 alpha 在 [0, 1] 之间，绝对禁止微观残差吞噬宏观物理基态
        alpha_bounded = torch.sigmoid(self.alpha_logits)
        
        # 融合与截断
        scaled_residual = alpha_bounded * sparse_residual
        A_combined = torch.relu(self.A_prior + scaled_residual)
        
        # 注入单位阵
        A_final = A_combined + self.I
        
        return A_final