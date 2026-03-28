#!/bin/bash

# 1. 物理环境补丁
export TARGET_NCCL="/environment/miniconda3/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD=$TARGET_NCCL
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 2. 实验矩阵配置
DATASETS=("WTH.csv" "ECL.csv" "traffic.csv")
PRED_LENS=(96 192 336 720)
SEQ_LEN=336

# 结果汇总路径
RESULT_CSV="./logs_benchmark/summary_seq336.csv"
mkdir -p ./logs_benchmark
if [ ! -f "$RESULT_CSV" ]; then
    echo "Dataset,Seq_Len,Pred_Len,MSE,MAE,Status" > "$RESULT_CSV"
fi

# 动态配置函数：返回 ENC_IN, BATCH, DATA_TYPE, TARGET, EPOCHS, LR
get_config() {
    case $1 in
        # ⚠️ 已将 WTH 的维度从 21 修正为真实的 12，Target 修正为 WetBulbCelsius
        "WTH.csv")     echo "12 64 custom WetBulbCelsius 20 0.0001"      ;; 
        "ECL.csv")     echo "321 4 custom MT_320 30 0.00005"  ;; 
        "traffic.csv") echo "862 2 custom OT 40 0.000025"   ;; 
    esac
}

# 3. 自动化循环逻辑
for DATA_FILE in "${DATASETS[@]}"
do
    config=($(get_config $DATA_FILE))
    ENC_IN=${config[0]}
    BATCH=${config[1]}
    DATA_TYPE=${config[2]}
    TARGET_COL=${config[3]}
    EPOCHS=${config[4]}
    LR=${config[5]}
    
    DATA_NAME=$(echo $DATA_FILE | cut -d'.' -f1)
    LOG_DIR="./logs_benchmark/$DATA_NAME"
    mkdir -p "$LOG_DIR"

    for PLEN in "${PRED_LENS[@]}"
    do
        MODEL_ID="${DATA_NAME}_sl${SEQ_LEN}_pl${PLEN}"
        LOG_FILE="$LOG_DIR/${MODEL_ID}.log"

        # 自动跳过逻辑：检查是否有完整的测试结果
        if [ -f "$LOG_FILE" ] && grep -q "mse:" "$LOG_FILE"; then
            echo "✅ [SKIP] $MODEL_ID 已完成。"
            continue
        fi

        echo "🚀 [RUNNING] $MODEL_ID | Target:$TARGET_COL | Vars:$ENC_IN | Batch:$BATCH"
        
        # 执行训练
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --model_id "$MODEL_ID" \
          --model Tri_FI \
          --data "$DATA_TYPE" \
          --root_path /home/featurize/data/ \
          --data_path "$DATA_FILE" \
          --features M \
          --target "$TARGET_COL" \
          --seq_len $SEQ_LEN \
          --label_len 168 \
          --pred_len "$PLEN" \
          --enc_in "$ENC_IN" \
          --dec_in "$ENC_IN" \
          --c_out "$ENC_IN" \
          --d_model 512 \
          --d_ff 512 \
          --batch_size "$BATCH" \
          --train_epochs "$EPOCHS" \
          --learning_rate "$LR" \
          --use_amp \
          --use_gpu True \
          --gpu 0 \
          --num_workers 4 > "$LOG_FILE" 2>&1

        # 检查是否成功并清理
        if [ $? -eq 0 ]; then
            MSE=$(grep "mse:" "$LOG_FILE" | tail -n 1 | awk -F'mse:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
            MAE=$(grep "mae:" "$LOG_FILE" | tail -n 1 | awk -F'mae:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
            echo "$DATA_NAME,$SEQ_LEN,$PLEN,$MSE,$MAE,SUCCESS" >> "$RESULT_CSV"
            rm -rf "./checkpoints/${MODEL_ID}"*
            echo "📈 [DONE] MSE: $MSE"
        else
            echo "$DATA_NAME,$SEQ_LEN,$PLEN,N/A,N/A,FAIL" >> "$RESULT_CSV"
            echo "❌ [ERROR] $MODEL_ID 崩溃，查阅 $LOG_FILE"
            # 微信推送替代方案 (只需一行代码)
            curl -s "http://www.pushplus.plus/send?token=e9213accdb0f4156a5716c8ff5a35ad6&title=🚨模型崩溃告警&content=任务 $MODEL_ID 失败，快去看看！"
        fi
    done
done