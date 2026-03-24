#!/bin/bash

# ==============================================================================
# 🏆 Tri-Fi Model: Blackwell (sm_120) 96G 满血打榜脚本
# 🛠️ Fix: 强制预加载 NCCL 2.28.9 解决 undefined symbol 冲突
# ==============================================================================
export TARGET_NCCL="/environment/miniconda3/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2"

# 强行置顶加载顺序
export LD_PRELOAD=$TARGET_NCCL

# 验证路径是否存在
if [ ! -f "$LD_PRELOAD" ]; then
    echo "❌ 错误：找不到 Blackwell 专属 NCCL 库，请检查安装路径。"
    exit 1
fi

echo "📦 Blackwell Patch: LD_PRELOAD is active."
# 1. 🚨 Blackwell 架构核心补丁 (极其关键)
echo "📦 [System] Blackwell Patch Applied: Preloading $LD_PRELOAD"

# 2. 强制系统环境对齐
export CUDA_VISIBLE_DEVICES=0
model_name=Tri_FI
cd /home/featurize/work/Tri-Fi

# 3. 基础目录初始化
mkdir -p logs_benchmark/Big2
mkdir -p checkpoints
mkdir -p results

# 4. 全局参数 (对齐 ETTm1_V3 成功配置)
SEQ_LEN=336
LABEL_LEN=168
TRAIN_EPOCHS=20
BASE_PATIENCE=5
BASE_LR=0.0003

echo "🚀 [Start] Tri-Fi Blackwell 96G Benchmark Initiated..."

# 5. 遍历数据集
for DATASET in electricity traffic; do

    if [ "$DATASET" == "electricity" ]; then
        DATA_FILE="ECL.csv"; DATA_TYPE="custom"; ENC_IN=321; BATCH_SIZE=32; TARGET="MT_320"
        echo "--------------------------------------------------"
        echo "⚡ [Config] Electricity (321 dim) -> Target: $TARGET"
        
    elif [ "$DATASET" == "traffic" ]; then
        DATA_FILE="traffic.csv"; DATA_TYPE="custom"; ENC_IN=862; BATCH_SIZE=16; TARGET="OT"
        echo "--------------------------------------------------"
        echo "🚗 [Config] Traffic (862 dim) -> Target: $TARGET"
    fi

    for PRED_LEN in 96 192 336 720; do
        LOG_FILE="logs_benchmark/Big2/TriFi_${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}.log"
        
        # 探针：跳过已完成任务
        if [ -f "$LOG_FILE" ]; then
            if grep -q "Epoch: $TRAIN_EPOCHS" "$LOG_FILE" || grep -q "EarlyStopping" "$LOG_FILE"; then
                echo "  ⏭️  [Skip] ${DATASET} (pred=$PRED_LEN) already completed."
                continue
            fi
        fi

        echo "  🔥 [Running] Training ${DATASET} | Seq: $SEQ_LEN -> Pred: $PRED_LEN"
        
        CURRENT_LR=$BASE_LR
        [ "$PRED_LEN" -eq 720 ] && CURRENT_LR=0.0001
        
        echo "==================================================" > $LOG_FILE
        echo "Task: $DATASET | Target: $TARGET | Device: Blackwell (12.0)" >> $LOG_FILE
        echo "Started at: $(date)" >> $LOG_FILE
        echo "==================================================" >> $LOG_FILE

        # 🚀 引擎点火 (环境变量已通过 export 全局生效)
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path /home/featurize/data/ \
            --data_path $DATA_FILE \
            --model_id ${DATASET}_${SEQ_LEN}_${PRED_LEN}_TriFi_V3 \
            --model $model_name \
            --data $DATA_TYPE \
            --features M \
            --target $TARGET \
            --seq_len $SEQ_LEN \
            --label_len $LABEL_LEN \
            --pred_len $PRED_LEN \
            --enc_in $ENC_IN \
            --dec_in $ENC_IN \
            --c_out $ENC_IN \
            --d_model 512 \
            --d_ff 1024 \
            --e_layers 2 \
            --tau 1 --m_dim 3 --k 3 \
            --dropout 0.2 \
            --batch_size $BATCH_SIZE \
            --train_epochs $TRAIN_EPOCHS \
            --patience $BASE_PATIENCE \
            --lradj type3 \
            --learning_rate $CURRENT_LR \
            --loss MAE \
            --des 'Exp_TriFi_Blackwell' \
            --itr 1 2>&1 | tee -a $LOG_FILE
            
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "  ✅ [Success] ${DATASET} (pred=$PRED_LEN) finished."
        else
            echo "  ❌ [FATAL ERROR] ${DATASET} crashed! Check $LOG_FILE."
            exit 1 
        fi
    done
done
