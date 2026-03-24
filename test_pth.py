# 您可以在终端运行此探针，请确保路径正确
import torch
import sys
sys.path.append('/home/featurize/work/Tri-Fi/')
from model.Tri_FI import Model

# 加载刚训练好的最佳权重
ckpt = torch.load('/home/featurize/work/Tri-Fi/checkpoints/long_term_forecast_ETTm1_96_96_TriFi_V2_Tri_FI_ETTm1_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df1024_fc1_ebtimeF_dtTrue_tau4_mdim3_k5_Exp_TriFi_ETTm1_V2_0/checkpoint.pth')
print(f"时域权重 (W_intra): {ckpt['W_intra'].item()}")
print(f"频域权重 (W_inter): {ckpt['W_inter'].item()}")
print(f"拓扑权重 (W_chaos): {ckpt['W_chaos'].item()}")