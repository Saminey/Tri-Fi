#!/bin/bash

# ==========================================
# 1. 物理环境与显存补丁 (Environment & Memory Patch)
# ==========================================
export TARGET_NCCL="/environment/miniconda3/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD=$TARGET_NCCL
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ==========================================
# 2. 全局实验矩阵配置 (Global Experiment Matrix)
# ==========================================
DATASETS=("WTH.csv" "traffic.csv")
PRED_LENS=(96 192 336 720)

RESULT_CSV="./logs_benchmark/summary_final.csv"
mkdir -p ./logs_benchmark
# 每次重新运行脚本时，先清空或初始化 CSV 表头，保证数据不重复追加
echo "Dataset,Seq_Len,Label_Len,Pred_Len,MSE,MAE,Status" > "$RESULT_CSV"

# ==========================================
# 3. 自动化循环与动态路由逻辑 (Automation & Dynamic Routing)
# ==========================================
for DATA_FILE in "${DATASETS[@]}"
do
    # --- 双轨制参数路由中心 ---
    case $DATA_FILE in
        "WTH.csv")
            ENC_IN=12
            BATCH=64
            TARGET_COL="WetBulbCelsius"
            EPOCHS=20
            LR="0.0001"
            LR_ADJ="cosine"
            SEQ_LEN=96
            D_MODEL=256
            D_FF=512
            DROPOUT=0.3
            K_NEIGHBORS=1
            ;;
            
        "traffic.csv")
            ENC_IN=862
            BATCH=2
            TARGET_COL="OT"
            EPOCHS=40
            LR="0.000025"
            LR_ADJ="slow_step"
            SEQ_LEN=336
            D_MODEL=512
            D_FF=512            
            DROPOUT=0.1
            K_NEIGHBORS=3
            ;;
            
        *)
            echo "⚠️ [WARN] 未知的配置文件 $DATA_FILE，跳过..."
            continue
            ;;
    esac

    # 🚨 核心修正：动态计算 Label_Len (保证永远等于 Seq_Len 的一半)
    LABEL_LEN=$((SEQ_LEN / 2))

    DATA_TYPE="custom"
    DATA_NAME=$(echo $DATA_FILE | cut -d'.' -f1)
    LOG_DIR="./logs_benchmark/$DATA_NAME"
    mkdir -p "$LOG_DIR"

    # 进入预测长度循环
    for PLEN in "${PRED_LENS[@]}"
    do
        MODEL_ID="${DATA_NAME}_sl${SEQ_LEN}_pl${PLEN}"
        LOG_FILE="$LOG_DIR/${MODEL_ID}.log"

        # 🛡️ 完善跳过逻辑：如果已存在，不仅跳过，还要把历史成绩写进本次的 CSV 中
        if [ -f "$LOG_FILE" ] && grep -q "mse:" "$LOG_FILE"; then
            MSE=$(grep "mse:" "$LOG_FILE" | tail -n 1 | awk -F'mse:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
            MAE=$(grep "mae:" "$LOG_FILE" | tail -n 1 | awk -F'mae:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
            echo "$DATA_NAME,$SEQ_LEN,$LABEL_LEN,$PLEN,$MSE,$MAE,SKIPPED_BUT_SUCCESS" >> "$RESULT_CSV"
            echo "✅ [SKIP] $MODEL_ID 已完成，已将历史成绩重新载入汇总表。"
            continue
        fi

        echo "🚀 [RUNNING] $MODEL_ID | Vars:$ENC_IN | Seq:$SEQ_LEN | Label:$LABEL_LEN | Batch:$BATCH"
        
        # 执行训练 (注入动态 LABEL_LEN)
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
          --seq_len "$SEQ_LEN" \
          --label_len "$LABEL_LEN" \
          --pred_len "$PLEN" \
          --enc_in "$ENC_IN" \
          --dec_in "$ENC_IN" \
          --c_out "$ENC_IN" \
          --d_model "$D_MODEL" \
          --d_ff "$D_FF" \
          --dropout "$DROPOUT" \
          --k "$K_NEIGHBORS" \
          --batch_size "$BATCH" \
          --train_epochs "$EPOCHS" \
          --learning_rate "$LR" \
          --lradj "$LR_ADJ" \
          --use_amp \
          --use_gpu True \
          --gpu 0 \
          --num_workers 4 > "$LOG_FILE" 2>&1

        # 检查是否成功
        if [ $? -eq 0 ] && grep -q "mse:" "$LOG_FILE"; then
            MSE=$(grep "mse:" "$LOG_FILE" | tail -n 1 | awk -F'mse:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
            MAE=$(grep "mae:" "$LOG_FILE" | tail -n 1 | awk -F'mae:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
            
            # 双重防呆检查，确保提取到了数字
            if [ -n "$MSE" ]; then
                echo "$DATA_NAME,$SEQ_LEN,$LABEL_LEN,$PLEN,$MSE,$MAE,SUCCESS" >> "$RESULT_CSV"
                rm -rf "./checkpoints/${MODEL_ID}"*
                echo "📈 [DONE] MSE: $MSE | MAE: $MAE"
            else
                echo "❌ [ERROR] 脚本成功退出但未找到评估指标！"
                echo "$DATA_NAME,$SEQ_LEN,$LABEL_LEN,$PLEN,N/A,N/A,METRIC_ERROR" >> "$RESULT_CSV"
            fi
        else
            echo "$DATA_NAME,$SEQ_LEN,$LABEL_LEN,$PLEN,N/A,N/A,FAIL" >> "$RESULT_CSV"
            echo "❌ [ERROR] $MODEL_ID 运行崩溃！详细信息已写入 $LOG_FILE"
            
            # 🗑️ 核心修正：崩溃也要清理 Checkpoints，防止磁盘爆炸
            rm -rf "./checkpoints/${MODEL_ID}"*
            echo "🧹 [CLEANUP] 已清理崩溃任务残留的权重文件。"
            
            curl -G -s "http://www.pushplus.plus/send" \
                --data-urlencode "token=e9213accdb0f4156a5716c8ff5a35ad6" \
                --data-urlencode "title=🚨 模型训练崩溃告警" \
                --data-urlencode "content=任务 [$MODEL_ID] 失败！请及时登录服务器排查。"
            
            # 阻断当前数据集后续任务
            echo "⚠️ [BREAK] 跳过当前数据集的后续预测任务..."
            break 
        fi
    done
done

echo "🎉 所有自动化训练任务执行完毕！请查看 $RESULT_CSV"