import torch
import argparse

# 核心修改点：根据你的文件树，从 model 文件夹下的 Bi_FI.py 导入 Model 类
from model.Bi_FI import Model 

def test_bifi_forecast():
    print("=== 开始测试 Bi-FI 模型的长序列预测 (Forecast) 任务 ===")

    # 1. 模拟 argparse 的 Configs 对象
    # 真实训练时，这些参数通常由命令行传入，我们这里手动模拟
    class MockConfig:
        def __init__(self):
            self.task_name = 'long_term_forecast' # 核心：指定预测任务
            self.seq_len = 96       # T: 回溯历史窗口长度 (看过去 96 个小时)
            self.pred_len = 24      # S: 预测未来长度 (预测未来 24 个小时)
            self.enc_in = 7         # M: 变量维度 (如 ETT 数据集的 7 个传感器)
            
            # 模型超参数 (根据论文常规设置模拟)
            self.d_model = 512      # 隐藏层维度 H
            self.embed = 'timeF'    # 时间特征嵌入方式
            self.freq = 'h'         # 数据频率 (h代表小时)
            self.dropout = 0.1
            self.factor = 3
            self.n_heads = 8
            self.d_ff = 2048
            self.activation = 'gelu'
            self.e_layers = 2       # Transformer Encoder 的层数
            self.output_attention = False

    configs = MockConfig()
    
    # 2. 实例化模型
    print("\n[1/3] 正在初始化模型...")
    model = Model(configs)
    # 如果有 GPU 就放到 GPU 上测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"模型初始化成功！当前设备: {device}")

    # 3. 伪造输入数据 (模拟 DataLoader 吐出的数据)
    batch_size = 32
    time_feature_dim = 4 # 时间协变量的维度 (比如分别代表: 月, 日, 星期, 小时)

    print(f"\n[2/3] 正在生成虚拟的 Batch 数据 (Batch Size = {batch_size})...")
    # x_enc: 历史时序数据 [Batch, T, M]
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in).to(device)
    # x_mark_enc: 历史时间戳特征 [Batch, T, Time_Features]
    x_mark_enc = torch.randn(batch_size, configs.seq_len, time_feature_dim).to(device)
    
    # 注: 在 Informer/Autoformer 的标准框架中，预测任务还需要传入 decoder 的输入
    # 尽管你在 Bi-FI 的 forecast 方法中实际上并没有用到它们（它是 Encoder-only 架构）
    # 但由于 forward 方法签名的要求，我们依然要传给它占位符。
    label_len = 48 # 解码器通常需要一段已知的历史作为引导
    x_dec = torch.randn(batch_size, label_len + configs.pred_len, configs.enc_in).to(device)
    x_mark_dec = torch.randn(batch_size, label_len + configs.pred_len, time_feature_dim).to(device)

    print(f" -> 历史数据 (x_enc) 形状: {x_enc.shape}")
    print(f" -> 历史时间特征 (x_mark_enc) 形状: {x_mark_enc.shape}")

    # 4. 执行前向传播
    print("\n[3/3] 执行前向传播 (Forward Pass)...")
    # 关闭梯度计算，节省内存，仅做测试
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    print("\n=== 测试完成 ===")
    print(f"🚀 模型最终输出 (Prediction) 形状: {output.shape}")
    
    # 验证输出形状是否符合预期: [Batch, pred_len, enc_in]
    assert output.shape == (batch_size, configs.pred_len, configs.enc_in), "输出形状不符合预期！"
    print(f"✅ 形状验证通过！成功预测未来 {configs.pred_len} 步的 {configs.enc_in} 个变量。")

if __name__ == "__main__":
    test_bifi_forecast()