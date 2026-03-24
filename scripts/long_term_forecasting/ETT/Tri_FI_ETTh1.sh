export CUDA_VISIBLE_DEVICES=0
model_name=Tri_FI

# 🚀 优化点：
# 1. 增加了 --inverse 修复报错
# 2. 调整 patience 为 5，给模型更多进化空间
# 3. 调整 train_epochs 为 20
# 4. 调整 lradj 为 type3 (更平滑的学习率衰减)

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/featurize/data/ETDataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_TriFi \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 512 \
  --e_layers 2 \
  --tau 1 --m_dim 3 --k 3 \
  --dropout 0.2 \
  --train_epochs 20 \
  --patience 5 \
  --lradj type3 \
  --learning_rate 0.0001 \
  --inverse \
  --des 'Exp_TriFi_V2' \
  --itr 1