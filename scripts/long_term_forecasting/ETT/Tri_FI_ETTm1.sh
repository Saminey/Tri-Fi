export CUDA_VISIBLE_DEVICES=0
model_name=Tri_FI

# ==========================================
# 🚀 TriFi 全景视野终极配置 (ETTm1_V3)
# 1. 视野：seq_len=336, label_len=168 (观测3.5天)
# 2. 内存对冲：batch_size=16
# ==========================================

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/featurize/data/ETDataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_336_96_TriFi_V3 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --label_len 168 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 1024 \
  --e_layers 2 \
  --tau 4 --m_dim 3 --k 5 \
  --dropout 0.2 \
  --batch_size 16 \
  --train_epochs 20 \
  --patience 5 \
  --lradj type3 \
  --learning_rate 0.0003 \
  --loss MAE \
  --inverse \
  --des 'Exp_TriFi_ETTm1_V3_Horizon' \
  --itr 1