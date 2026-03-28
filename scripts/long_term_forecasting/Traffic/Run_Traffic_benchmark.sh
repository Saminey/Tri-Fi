#!/bin/bash

# ==========================================
# 1. 物理环境解禁 (RTX 6000 96GB 专供)
# ==========================================
export CUDA_LAUNCH_BLOCKING=0
# 开启所有内存扩展补丁，确保大 Batch Size 不碎片化
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

# ==========================================
# 2. PushPlus 告警配置 (请填入你的 Token)
# ==========================================
PUSH_TOKEN="你的PushPlus_Token" # 👈 在这里填入你的 Token
send_alert() {
    local title=$1
    local content=$2
    curl -s -X POST "http://www.pushplus.plus/send" \
        -H "Content-Type: application/json" \
        -d "{\"token\":\"$e9213accdb0f4156a5716c8ff5a35ad6\",\"title\":\"$title\",\"content\":\"$content\",\"template\":\"html\"}" > /dev/null
}

# ==========================================
# 3. 实验矩阵配置
# ==========================================
DATA_FILE="traffic.csv"
DATA_NAME="traffic"
PRED_LENS=(96 192 336 720)
RESULT_CSV="./logs_benchmark/summary_traffic_aggressive.csv"

# ==========================================
# 4. 激进参数锁定 (最大限度榨干 96GB 显存)
# ==========================================
ENC_IN=862
# 🚀 激进优化：Batch Size 从 16 暴力提升至 128
# 如果显存依然空闲，可以尝试 256
BATCH=128 
EPOCHS=30
LR="0.0002"              # 配合大 Batch，适当调高学习率
LR_ADJ="slow_step"       # 保持稳健衰减

SEQ_LEN=336
LABEL_LEN=168

# 🚀 模型计算密度增强
D_MODEL=512
D_FF=2048                # 增加前馈层宽度，提升 GPU 计算核心占用
E_LAYERS=4               # 增加编码器层数，深度特征提取
DROPOUT=0.1
NUM_WORKERS=16           # 🚀 暴力提升 CPU 并行预处理，消除 GPU 饥饿

# PSR 拓扑参数保持稳健
M_DIM=5
TAU=1
K_NEIGHBORS=4

LOG_DIR="./logs_benchmark/${DATA_NAME}_aggressive"
mkdir -p "$LOG_DIR"

# ==========================================
# 5. 执行训练循环
# ==========================================
send_alert "🚀 训练启动" "数据集: $DATA_NAME <br>模型: Tri_FI <br>Batch Size: $BATCH"

for PLEN in "${PRED_LENS[@]}"
do
    MODEL_ID="${DATA_NAME}_sl${SEQ_LEN}_pl${PLEN}_AGGRESSIVE"
    LOG_FILE="$LOG_DIR/${MODEL_ID}.log"

    echo "🔥 [AGRESSIVE RUN] $MODEL_ID | Batch: $BATCH | Workers: $NUM_WORKERS"
    
    # 记录开始时间
    START_TIME=$(date +%s)

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id "$MODEL_ID" \
      --model Tri_FI \
      --data "custom" \
      --root_path /home/featurize/data/ \
      --data_path "$DATA_FILE" \
      --features M \
      --target "OT" \
      --seq_len "$SEQ_LEN" \
      --label_len "$LABEL_LEN" \
      --pred_len "$PLEN" \
      --enc_in "$ENC_IN" \
      --dec_in "$ENC_IN" \
      --c_out "$ENC_IN" \
      --d_model "$D_MODEL" \
      --d_ff "$D_FF" \
      --e_layers "$E_LAYERS" \
      --tau "$TAU" \
      --m_dim "$M_DIM" \
      --dropout "$DROPOUT" \
      --k "$K_NEIGHBORS" \
      --batch_size "$BATCH" \
      --train_epochs "$EPOCHS" \
      --learning_rate "$LR" \
      --lradj "$LR_ADJ" \
      --use_amp \
      --use_gpu True \
      --num_workers "$NUM_WORKERS" > "$LOG_FILE" 2>&1

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
        MSE=$(grep "mse:" "$LOG_FILE" | tail -n 1 | awk -F'mse:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
        echo "✅ [SUCCESS] $MODEL_ID | MSE: $MSE"
        send_alert "✅ 任务完成: $PLEN" "预测长度: $PLEN <br>MSE: $MSE <br>耗时: ${DURATION}s"
    else
        echo "❌ [CRASH] $MODEL_ID 运行失败！"
        send_alert "🚨 训练崩溃: $PLEN" "任务 ID: $MODEL_ID <br>原因: 请检查日志 $LOG_FILE <br>建议: 如果是 OOM，请下调 BATCH。"
        # 崩溃后建议停止后续循环，避免连续轰炸告警
        exit 1
    fi
done

send_alert "🎉 全案执行结束" "所有预测长度已跑完，请登录服务器检查 summary CSV。"