import os
import re
import csv
from pathlib import Path

def parse_logs():
    # 定义输入和输出路径
    log_dir = '/home/featurize/work/Tri-Fi/logs_benchmark'
    out_csv = os.path.join(log_dir, 'user_summary.csv')

    # 我们需要从 Namespace 中提取的关键参数列表
    target_args = [
        'data_path', 'model_id', 'seq_len', 'pred_len',
        'd_model', 'd_ff', 'e_layers', 'batch_size', 'learning_rate',
        'm_dim', 'tau', 'k'
    ]

    # CSV 表头（包含文件名、参数、结果、轮数）
    fieldnames = ['log_file'] + target_args + ['stopped_epoch', 'mse', 'mae']
    
    extracted_data = []

    # 遍历该目录下所有的 .log 文件
    log_files = list(Path(log_dir).rglob('*.log'))
    print(f"🔍 寻找到 {len(log_files)} 个日志文件，开始解析...")

    for log_path in log_files:
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️ 无法读取 {log_path.name}: {e}")
            continue

        row_data = {'log_file': log_path.name}

        # 1. 提取 Namespace 中的超参数
        for key in target_args:
            # 匹配 key='value' 或者 key=123 的格式
            match = re.search(rf"{key}=['\"]?([^,'\"\)]+)['\"]?", content)
            row_data[key] = match.group(1) if match else "N/A"

        # 2. 提取最终的 MSE 和 MAE
        metric_match = re.search(r"mse:\s*([0-9.]+),\s*mae:\s*([0-9.]+)", content)
        if metric_match:
            row_data['mse'] = metric_match.group(1)
            row_data['mae'] = metric_match.group(2)
        else:
            row_data['mse'] = "Crash/Unfinished"
            row_data['mae'] = "Crash/Unfinished"

        # 3. 提取实际运行的 Epoch 轮数（看模型是在第几轮早停的）
        epoch_matches = re.findall(r"Epoch:\s+(\d+),\s+Steps:", content)
        row_data['stopped_epoch'] = epoch_matches[-1] if epoch_matches else "0"

        extracted_data.append(row_data)

    # 将提取的数据写入 CSV 文件
    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_data)

    print(f"✅ 提取完成！成功处理了 {len(extracted_data)} 个实验记录。")
    print(f"📁 汇总文件已保存至: {out_csv}")

if __name__ == '__main__':
    parse_logs()