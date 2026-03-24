#!/bin/bash

# ==============================================================================
# 🏆 Tri-Fi Model: Weather Benchmark Full Automation Script (3090 满血修复版)
# ⚙️ 完美继承 ETTm1_V3 成功配置，解决路径找不到与调试信息不直观问题
# ==============================================================================

# 1. 强制系统环境对齐
export CUDA_VISIBLE_DEVICES=0
model_name=Tri_FI

# 🚨 极其关键：强制进入项目根目录，确保 run.py 绝对能找到
cd /home/featurize/work/Tri-Fi

# 2. 基础目录初始化
mkdir -p logs_benchmark/Weather
mkdir -p checkpoints
mkdir -p results

# 3. 全局基础参数 (对齐您的 ETTm1 成功配置 V3 版)
SEQ_LEN=336
LABEL_LEN=168    # 之前的 48 改为 168，对齐 SOTA 视野
TRAIN_EPOCHS=20
BASE_PATIENCE=5
BASE_LR=0.0003   # 对齐 ETTm1 的 0.0003，更稳健

# 4. Weather 数据集专属配置 (21维特征)
DATASET="weather"
DATA_PATH="weather.csv"
ENC_IN=21
K=5         # 高频稠密数据，近邻数拉满
TAU=4       # 扩大相空间时间延迟
D_MODEL=512 
D_FF=1024    # 对齐 ETTm1 的 1024
BATCH_SIZE=32

echo "🚀 [Start] Tri-Fi Weather SOTA Benchmark Initiated on GPU: 0..."
echo "--------------------------------------------------"
echo "🌪️ [Config] Weather Data -> k=$K, tau=$TAU, d_model=$D_MODEL, d_ff=$D_FF"
echo "--------------------------------------------------"

# 5. 核心循环：遍历预测视界
for PRED_LEN in 96 192 336 720; do
    
    LOG_FILE="logs_benchmark/Weather/TriFi_${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}.log"
    
    # 🛡️ 防断电断点续传探针
    if [ -f "$LOG_FILE" ]; then
        if grep -q "Epoch: $TRAIN_EPOCHS" "$LOG_FILE" || grep -q "EarlyStopping" "$LOG_FILE"; then
            echo "  ⏭️  [Skip] ${DATASET} (pred=$PRED_LEN) already completed. Skipping."
            continue
        fi
    fi

    echo "  🔥 [Running] Training ${DATASET} | Seq: $SEQ_LEN -> Pred: $PRED_LEN"
    
    # 🧠 针对 720 步长的自适应策略
    CURRENT_LR=$BASE_LR
    if [ "$PRED_LEN" -eq 720 ]; then
        CURRENT_LR=0.0001
        echo "      ⚠️ [Auto-Tune] Extreme horizon (720). LR set to $CURRENT_LR"
    fi
    
    # 记录开始时间戳
    START_TIME=$(date +%s)
    echo "==================================================" > $LOG_FILE
    echo "Task: $DATASET | Seq: $SEQ_LEN | Pred: $PRED_LEN" >> $LOG_FILE
    echo "Started at: $(date)" >> $LOG_FILE
    echo "==================================================" >> $LOG_FILE

    # 🚀 引擎点火 (参数 100% 对齐您的成功案例)
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path /home/featurize/data/ \
        --data_path $DATA_PATH \
        --model_id ${DATASET}_${SEQ_LEN}_${PRED_LEN}_TriFi_V3 \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $SEQ_LEN \
        --label_len $LABEL_LEN \
        --pred_len $PRED_LEN \
        --enc_in $ENC_IN \
        --dec_in $ENC_IN \
        --c_out $ENC_IN \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --e_layers 2 \
        --tau $TAU \
        --m_dim 3 \
        --k $K \
        --dropout 0.2 \
        --batch_size $BATCH_SIZE \
        --train_epochs $TRAIN_EPOCHS \
        --patience $BASE_PATIENCE \
        --lradj type3 \
        --learning_rate $CURRENT_LR \
        --loss MAE \
        --inverse false\
        --des 'Exp_TriFi_Weather_V3' \
        --itr 1 2>&1 | tee -a $LOG_FILE
        
    # 🛡️ 神级调试机制
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        END_TIME=$(date +%s)
        EXECUTION_TIME=$((END_TIME - START_TIME))
        echo "  ✅ [Success] ${DATASET} (pred=$PRED_LEN) finished in ${EXECUTION_TIME}s."
    else
        echo "  ❌ [FATAL ERROR] ${DATASET} (pred=$PRED_LEN) crashed! Check logs above."
        exit 1
    fi
done

echo "🎉 [Mission Complete] Weather Benchmark tasks fully executed!"