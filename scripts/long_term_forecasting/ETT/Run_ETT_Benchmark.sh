#!/bin/bash

# ==============================================================================
# 🏆 Tri-Fi Model: ETT Benchmark Full Automation Script (Industrial Grade)
# 📝 Description: Reproduce Main Table SOTA results for NeurIPS/ICLR submission.
# ⚙️ Features: Dynamic topology, Auto-resume, Crash detection, Horizon adaptation.
# ==============================================================================

# 1. 基础设置与目录初始化
GPU_ID=0 # 显式指定使用的显卡号 (对于 3090/4090 多卡服务器非常重要)
mkdir -p logs_benchmark/ETT
mkdir -p checkpoints
mkdir -p results

# 2. 全局基础参数 (Global Base Parameters)
SEQ_LEN=336
LABEL_LEN=48
TRAIN_EPOCHS=20
BASE_PATIENCE=5
BASE_LR=0.0005

echo "🚀 [Start] Tri-Fi ETT Series SOTA Benchmark Initiated on GPU: $GPU_ID..."

# 3. 第一层循环：遍历 ETT 四大数据集
for DATASET in ETTh1 ETTh2 ETTm1 ETTm2; do

    # 🌟 核心逻辑：基于频率的“动态参数路由”
    ENC_IN=7 
    
    if [[ "$DATASET" == *"ETTh"* ]]; then
        # 小时级数据 (Low-frequency)
        K=3
        TAU=1
        D_MODEL=256
        BATCH_SIZE=64
        echo "--------------------------------------------------"
        echo "⏳ [Config] Hourly Data ($DATASET) -> k=$K, tau=$TAU, d_model=$D_MODEL"
    elif [[ "$DATASET" == *"ETTm"* ]]; then
        # 分钟级数据 (High-frequency)
        K=5
        TAU=4
        D_MODEL=512
        BATCH_SIZE=64
        echo "--------------------------------------------------"
        echo "⏱️ [Config] Minute Data ($DATASET) -> k=$K, tau=$TAU, d_model=$D_MODEL"
    fi

    # 4. 第二层循环：遍历预测视界
    for PRED_LEN in 96 192 336 720; do
        
        LOG_FILE="logs_benchmark/ETT/TriFi_${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}.log"
        
        # 🛡️ 防断电断点续传探针
        if [ -f "$LOG_FILE" ]; then
            # 严格检查是否正常结束 (包含早停或跑满 Epoch)
            if grep -q "Epoch: $TRAIN_EPOCHS" "$LOG_FILE" || grep -q "EarlyStopping" "$LOG_FILE"; then
                echo "  ⏭️  [Skip] ${DATASET} (pred=$PRED_LEN) already completed. Skipping."
                continue
            fi
        fi

        echo "  🔥 [Running] Training ${DATASET} | Seq: $SEQ_LEN -> Pred: $PRED_LEN"
        
        # 🧠 终极绝招：针对 720 步长的极限外推自适应策略
        CURRENT_LR=$BASE_LR
        CURRENT_PATIENCE=$BASE_PATIENCE
        if [ "$PRED_LEN" -eq 720 ]; then
            CURRENT_LR=0.0001  # 降低步伐，防止掉入局部最优
            CURRENT_PATIENCE=10 # 给模型更多时间在崎岖地形中探索
            echo "      ⚠️ [Auto-Tune] Extreme horizon detected (720). LR set to $CURRENT_LR, Patience to $CURRENT_PATIENCE."
        fi
        
        # 记录开始时间戳
        START_TIME=$(date +%s)
        echo "==================================================" > $LOG_FILE
        echo "Task: $DATASET | Seq: $SEQ_LEN | Pred: $PRED_LEN" >> $LOG_FILE
        echo "Started at: $(date)" >> $LOG_FILE
        echo "==================================================" >> $LOG_FILE

        # 🚀 引擎点火 (加入 CUDA_VISIBLE_DEVICES 保护)
        CUDA_VISIBLE_DEVICES=$GPU_ID python -u run.py \
            --is_training 1 \
            --root_path ./data_provider/ \
            --data_path ${DATASET}.csv \
            --model_id ${DATASET}_${SEQ_LEN}_${PRED_LEN} \
            --model Tri_FI \
            --data custom \
            --features M \
            --seq_len $SEQ_LEN \
            --label_len $LABEL_LEN \
            --pred_len $PRED_LEN \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in $ENC_IN \
            --dec_in $ENC_IN \
            --c_out $ENC_IN \
            --des 'Main_Table_SOTA' \
            --d_model $D_MODEL \
            --d_ff 512 \
            --batch_size $BATCH_SIZE \
            --learning_rate $CURRENT_LR \
            --patience $CURRENT_PATIENCE \
            --k $K \
            --tau $TAU \
            --m 3 \
            --train_epochs $TRAIN_EPOCHS >> $LOG_FILE 2>&1
            
        # 🛡️ 崩溃侦测与耗时计算
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        EXECUTION_TIME=$((END_TIME - START_TIME))
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "  ✅ [Success] ${DATASET} (pred=$PRED_LEN) finished in ${EXECUTION_TIME}s."
        else
            echo "  ❌ [FAILED] ${DATASET} (pred=$PRED_LEN) crashed with exit code $EXIT_CODE! Check logs."
            # 可以选择在此处加 exit 1 直接终止整个脚本，或者 continue 继续跑下一个
        fi
    done
done

echo "🎉 [Mission Complete] All ETT Benchmark tasks executed!"