import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding
import numpy as np
import time
from torch.utils.checkpoint import checkpoint

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model
        self.hidden_size = configs.d_model  # hidden_size


        # Embedding
        self.intra_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.inter_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.seq_len, kernel_size=(2, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(self.seq_len, self.d_model, kernel_size=(2, 1), stride=(1, 1)),
            nn.ReLU()
        )

        # Encoder-only architecture
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


        self.projector1 = nn.Linear(configs.pred_len * 2, configs.pred_len, bias=True)
        self.dropout = nn.Dropout(configs.dropout)


        # Decoder
        if self.task_name == 'long_term_forecast' :
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projector = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projector = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

        if self.task_name == 'long_term_forecast' :
            self.out_len = configs.pred_len
        if self.task_name == 'anomaly_detection':
            self.out_len = configs.seq_len
        if self.task_name == 'classification':
            self.out_len = configs.seq_len


    def fc_layers(self, x):
        B, C, T = x.shape
        x = nn.Linear(T, self.out_len, bias=True).to(x.device)(x)
        return x

    def inter_frequency(self, x):
        x_fft = torch.fft.rfft(x, dim=1)
        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag

        # Combining real and imaginary parts into a complex tensor
        x_fft = torch.stack([x_fft_real, x_fft_imag], dim=-1).permute(0, 2, 1, 3)

        y = self.conv_layers(x_fft)
        y = torch.view_as_complex(y)

        x_time = torch.fft.irfft(y, dim=2)

        B, C, T = x_time.shape
        output = nn.Linear(T, self.out_len, bias=True).to(x_time.device)(x_time)

        return output

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape
        
        # ==========================================================
        # 🚀 新增: 张量维度透视探针 (Tensor Dimensionality Probe)
        # ==========================================================
        print("\n" + "🔍" + "="*20 + " [前向传播维度透视: forecast] " + "="*20)
        print(f"-> [0. 原始输入] x_enc (Batch, Seq_Len, Vars): {x_enc.shape}")

        # intra (时域分支)
        intra_in = self.intra_embedding(x_enc, x_mark_enc)
        print(f"-> [1. Intra] Inverted Embedding 后 (Batch, Vars, d_model): {intra_in.shape}")
        
        intra_out, attns = self.encoder(intra_in, attn_mask=None)
        print(f"-> [2. Intra] Transformer Encoder 后: {intra_out.shape}")
        
        intra_out = self.projector(intra_out).permute(0, 2, 1)[:, :, :N]
        print(f"-> [3. Intra] Projector & Permute 对齐后 (Batch, Pred_Len, Vars): {intra_out.shape}")

        # inter (频域分支)
        inter_in = self.inter_embedding(x_enc, x_mark_enc)
        print(f"-> [4. Inter] Standard Embedding 后 (Batch, Seq_Len, d_model): {inter_in.shape}")
        
        inter_out = self.inter_frequency(inter_in)
        print(f"-> [5. Inter] FFT + Conv2D 后: {inter_out.shape}")
        
        inter_out = inter_out.permute(0, 2, 1)[:, :, :N]
        print(f"-> [6. Inter] Permute 对齐后 (Batch, Pred_Len, Vars): {inter_out.shape}")

        # Fusion (特征融合)
        co_out = self.dropout(inter_out + intra_out)
        print(f"-> [7. Fusion] 双分支残差相加融合后 (co_out): {co_out.shape}")
        print("="*70 + "\n")
        # ==========================================================

        # De-Normalization from Non-stationary Transformer
        co_out = co_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        co_out = co_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return co_out


    def anomaly_detection(self, x_enc, x_mark_enc):
        # 保持原样...
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape  # B L N

        intra_in = self.intra_embedding(x_enc, x_mark_enc)
        intra_out, attns = self.encoder(intra_in, attn_mask=None)
        intra_out = self.projector(intra_out).permute(0, 2, 1)[:, :, :N]

        inter_in = self.inter_embedding(x_enc, x_mark_enc)
        inter_out = self.inter_frequency(inter_in)
        inter_out = inter_out.permute(0, 2, 1)[:, :, :N]

        co_out = self.dropout(inter_out + intra_out)

        co_out = co_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        co_out = co_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))

        return co_out

    def classification(self, x_enc, x_mark_enc):
        # 保持原样...
        intra_in = self.intra_embedding(x_enc, None)
        intra_out, attns = self.encoder(intra_in, attn_mask=None)
        intra_out = self.act(intra_out)
        intra_out = self.dropout(intra_out)
        intra_out = intra_out.reshape(intra_out.shape[0], -1)
        intra_out = self.projector(intra_out)


        inter_in = self.inter_embedding(x_enc, None)
        inter_out = self.inter_frequency(inter_in)
        inter_out = self.act(inter_out)
        inter_out = inter_out.reshape(inter_out.shape[0], -1)
        _, L = inter_out.shape
        inter_out = nn.Linear(L, self.num_class).to(inter_out.device)(inter_out)

        co_out = self.dropout(inter_out + intra_out)

        return co_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' :
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None


# ==============================================================================
# 独立测试入口 (Standalone Execution Entry Point)
# ==============================================================================
if __name__ == '__main__':
    import sys
    import os
    # 将项目根目录加入环境变量，防止 'from layers...' 导入报错
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("🚀 正在启动 Bi-FI 原生网络独立测试 (Standalone Test)...")
    
    # 1. 伪造 configs 对象
    class MockConfig:
        def __init__(self):
            self.task_name = 'long_term_forecast'
            self.seq_len = 96        # 历史窗口长度
            self.pred_len = 24       # 预测未来长度
            self.output_attention = False
            self.d_model = 512       # 隐藏层维度
            self.embed = 'timeF'
            self.freq = 'h'
            self.dropout = 0.1
            self.enc_in = 7          # 变量数 M (如 ETTh1 的 7个特征)
            self.factor = 1
            self.n_heads = 8
            self.d_ff = 2048
            self.activation = 'gelu'
            self.e_layers = 2        # Transformer 编码器层数
            self.num_class = 3

    configs = MockConfig()
    
    # 2. 实例化主干网络
    print("\n[1/3] 正在实例化 Bi-FI 主干网络...")
    model = Model(configs)
    
    # 打印极其详细的网络结构
    print("="*60)
    print("🧠 模型完整拓扑结构 (Model Architecture):")
    print("="*60)
    print(model)
    print("="*60)
    
    # 3. 伪造时序 DataLoader 张量
    print("\n[2/3] 正在生成伪造时序张量 (Dummy Data)...")
    batch_size = 32
    # 形状: [Batch, Seq_Len, Features]
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    # 形状: [Batch, Seq_Len, TimeFeatures(如小时、星期等通常为4)]
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 4) 
    # 解码器输入 (Bi-FI 预测时其实不怎么用解码器输入，但签名要求保留)
    x_dec = torch.randn(batch_size, configs.seq_len//2 + configs.pred_len, configs.enc_in)
    x_mark_dec = torch.randn(batch_size, configs.seq_len//2 + configs.pred_len, 4)
    
    # 4. 执行前向传播并触发维度探针
    print("\n[3/3] 执行前向传播 (Forward Pass)...")
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"\n🎉 测试成功! 最终预测输出形状 (Output Shape): {output.shape}")
    assert output.shape == (batch_size, configs.pred_len, configs.enc_in), "❌ 输出维度错误！"
