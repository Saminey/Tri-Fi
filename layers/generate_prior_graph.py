import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import RobustScaler # 替换为 RobustScaler

def reconstruct_phase_space_true_zerocopy(series, tau, m):
    """
    【CS终极优化】: 基于 as_strided 的纯 C 语言底层指针映射，绝对零拷贝。
    """
    L = len(series)
    N_t = L - (m - 1) * tau
    
    if N_t <= 0:
        raise ValueError("时间序列太短或 tau/m 设置过大，无法进行相空间重构！")
        
    # 获取原始一维数组的内存步长 (通常 float64 是 8 字节)
    itemsize = series.strides[0]
    
    # 构造新的形状 (m, N_t)
    # 行步长为 tau * itemsize (实现延迟 tau)
    # 列步长为 itemsize (实现连续的时间推移)
    reconstructed_view = np.lib.stride_tricks.as_strided(
        series, 
        shape=(m, N_t), 
        strides=(tau * itemsize, itemsize),
        writeable=False # 保护原始数据防篡改
    )
    
    # .flatten() 会发生一次不可避免的连续化拷贝，但这已经是理论极限上的最优解了
    return reconstructed_view.flatten()

def generate_static_prior_graph(data_path, save_path, tau=3, m_dim=5, k=4, target='OT', features='M'):
    print(f"🚀 开始生成物理先验图...\n数据路径: {data_path}")
    
    # 1. 对齐切分逻辑
    df_raw = pd.read_csv(data_path)
    cols = list(df_raw.columns)
    if target in cols: cols.remove(target)
    if 'date' in cols: cols.remove('date')
    df_raw = df_raw[['date'] + cols + [target]]
    
    if features in ['M', 'MS']:
        df_data = df_raw[df_raw.columns[1:]]
    else:
        df_data = df_raw[[target]]
        
    num_train = int(len(df_raw) * 0.7)
    train_data_raw = df_data.values[:num_train, :]  
    num_nodes = train_data_raw.shape[1]
    
    # 2. 【数据科学修复】：使用 RobustScaler 抵抗传感器异常极值毛刺
    scaler = RobustScaler()
    train_data_scaled = scaler.fit_transform(train_data_raw)

    # 3. 提取完整的相空间轨迹
    print(f"🌌 正在重构全局相空间轨迹 (tau={tau}, m_dim={m_dim})...")
    sample_traj = reconstruct_phase_space_true_zerocopy(train_data_scaled[:, 0], tau, m_dim)
    traj_length = len(sample_traj)
    
    # 初始化预分配内存，避免碎片化
    phase_space_trajectories = np.empty((num_nodes, traj_length), dtype=np.float32) 
    for i in range(num_nodes):
        phase_space_trajectories[i] = reconstruct_phase_space_true_zerocopy(train_data_scaled[:, i], tau, m_dim)
        
    # 4. 计算轨迹间的距离 【数学修复：消除维度灾难】
    print("📏 正在计算动力学流形轨迹距离矩阵...")
    # 计算平方欧氏距离，并立刻除以时间维度进行惩罚，防止高维距离爆炸
    dist_matrix_sq = euclidean_distances(phase_space_trajectories, squared=True) / traj_length
    dist_matrix = np.sqrt(dist_matrix_sq)
    
    # 强行将对角线（自身距离）设为无穷大
    np.fill_diagonal(dist_matrix, np.inf)
    
    # 取每个节点最近的 k 个邻居的距离 
    knn_distances = np.sort(dist_matrix, axis=1)[:, :k]
    adaptive_sigma = np.median(knn_distances)
    if adaptive_sigma < 1e-5: adaptive_sigma = 1e-5
    print(f"🎯 自动计算得出最优自适应 Sigma (降维修正后): {adaptive_sigma:.4f}")

    # 5. k-NN 建图
    print(f"🕸️ 正在生成无自环的 k-NN (k={k}) 高斯先验图...")
    A_prior = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        nearest_indices = np.argsort(dist_matrix[i])[:k]
        for j in nearest_indices:
            dist_sq = dist_matrix[i, j] ** 2
            weight = np.exp(-dist_sq / (adaptive_sigma ** 2))
            A_prior[i, j] = weight
            
    # 对称化保证无向图属性
    A_prior = np.maximum(A_prior, A_prior.T)
    
    # 强制二次确认对角线为 0
    np.fill_diagonal(A_prior, 0.0)
    
    actual_degrees = np.count_nonzero(A_prior, axis=1)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, A_prior)
    print(f"✅ 物理先验图已成功保存至: {save_path}")
    print(f"💡 图中最大度数: {np.max(actual_degrees)}, 平均度数: {np.mean(actual_degrees):.2f}")

if __name__ == "__main__":
    DATA_FILE = "/home/featurize/data/exchange_rate.csv"  
    SAVE_FILE = "./A_prior_exchange.npy"             
    
    generate_static_prior_graph(
        data_path=DATA_FILE,
        save_path=SAVE_FILE,
        tau=1,
        m_dim=3,
        k=2,
        target='OT',
        features='M'
    )