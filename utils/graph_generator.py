import os
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import warnings

warnings.filterwarnings('ignore')

def generate_dynamic_prior_graph(args):
    """
    根据运行参数自动进行相空间重构 (PSR) 并生成物理先验拓扑图
    （严格遵循无未来数据泄露原则）
    """
    # 1. 定义保存路径 (直接放入 logs_benchmark)
    save_dir = './logs_benchmark'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 获取数据集名称前缀 (如 traffic, exchange_rate)
    data_name = os.path.basename(args.data_path).split('.')[0]
    
    # 🚀 缓存命名法：将核心拓扑参数写入文件名
    save_name = f"A_prior_{data_name}_m{args.m_dim}_tau{args.tau}_k{args.k}_enc{args.enc_in}.npy"
    save_path = os.path.join(save_dir, save_name)

    # 2. 缓存拦截机制
    if os.path.exists(save_path):
        print(f"✅ [Auto-Graph] 检测到缓存的先验图，直接加载: {save_path}")
        return save_path

    print(f"🚀 [Auto-Graph] 未检测到适用图，正在基于【训练集】重构相空间并建图...")
    
    # 3. 读取原始数据
    df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
    cols = list(df_raw.columns)
    if 'date' in cols:
        cols.remove('date')
    if args.features == 'S':
        cols = [args.target]
        
    data = df_raw[cols].values
    
    # ==========================================
    # 🚨 核心修复：防止未来数据泄露
    # ==========================================
    # 获取通用的训练集比例 (通常长序列基准是 0.7 或 0.6)
    # 汇率和交通通常用 0.7，也可以根据你的 data_loader 保持一致
    train_ratio = 0.7 
    if 'exchange' in data_name.lower():
        train_ratio = 0.6 # 之前建议过汇率数据短，可以只取 60% 做训练
        
    train_size = int(len(data) * train_ratio)
    data = data[:train_size]  # 🚀 切割数据！只用历史数据建图
    # ==========================================

    T, N = data.shape
    
    # 安全校验：确保 args.enc_in 与数据实际维度一致
    if N != args.enc_in:
        print(f"⚠️ [警告] 数据实际维度 ({N}) 与 args.enc_in ({args.enc_in}) 不符！将以实际维度建图。")

    # 4. 零拷贝相空间重构 (PSR)
    m, tau, k = args.m_dim, args.tau, args.k
    valid_len = T - (m - 1) * tau
    
    data_cont = np.ascontiguousarray(data)
    s0, s1 = data_cont.strides
    # shape: (valid_len, nodes, m_dim)
    psr_data = np.lib.stride_tricks.as_strided(
        data_cont,
        shape=(valid_len, N, m),
        strides=(s0, s1, tau * s0)
    )
    
    # 5. 计算 KNN 拓扑图
    node_features = psr_data.transpose(1, 0, 2).reshape(N, -1)
    
    actual_k = min(k, N - 1) 
    if actual_k < 1: actual_k = 1
    
    # 构建有向/无向邻接矩阵
    A = kneighbors_graph(node_features, n_neighbors=actual_k, mode='connectivity', include_self=True).toarray()
    
    # 🚀 强制转为 float32，防止模型在 PyTorch 里报 Double Tensor 类型错误
    A = A.astype(np.float32) 
    
    # 保存并返回路径
    np.save(save_path, A)
    print(f"✅ [Auto-Graph] 建图完成！只使用了前 {train_size} 行训练数据以防止泄露。矩阵维度: {A.shape}，已保存至: {save_path}")
    
    return save_path