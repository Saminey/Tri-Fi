import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np
import datetime  
import json      

# 🚀 1. 导入你刚刚写好的自动建图工具
from utils.graph_generator import generate_dynamic_prior_graph

if __name__ == '__main__':
    fix_seed = 3407
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TriFi & Baselines')

    # ... (中间你所有的 parser.add_argument 保持完全不变) ...
    # 为了精简，这里省略长串的 add_argument
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='Tri_FI')
    parser.add_argument('--data', type=str, required=True, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--seq_len', type=int)
    parser.add_argument('--label_len', type=int)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--tau', type=int, default=1)
    parser.add_argument('--m_dim', type=int, default=3)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MAE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)
    parser.add_argument('--anomaly_ratio', type=float, default=0.25)
    parser.add_argument('--inverse', action='store_true', default=False)

    args = parser.parse_args()
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # ==========================================================
    # 🚀 2. 核心对接：在打印 Args 和初始化 Exp 之前，自动生成或加载物理图
    # ==========================================================
    if args.model == 'Tri_FI':
        print(f"============== 开始自动建图流水线 ==============")
        try:
            # 这一步会拦截数据，生成图，并将返回的路径挂载到 args 上
            args.prior_path = generate_dynamic_prior_graph(args)
            print(f"⚙️ 成功注入先验图路径至 args.prior_path: {args.prior_path}")
        except Exception as e:
            print(f"❌ [Auto-Graph Error] 自动建图失败，请检查数据路径或参数: {e}")
            raise e
        print(f"==================================================")

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            setting = f"{args.task_name}_{args.data}_{current_time}"

            res_path = os.path.join('./test_results', setting)
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            
            config_file = os.path.join(res_path, 'args_config.json')
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(vars(args), f, indent=4, ensure_ascii=False)
            print(f"✅ 超参数已自动保存至: {config_file}")

            exp = Exp(args)  
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        setting = f"{args.task_name}_{args.data}_{current_time}_test_only"

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()