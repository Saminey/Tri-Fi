import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np
import datetime  # 🚀 新增：用于获取时间戳
import json      # 🚀 新增：用于保存超参数字典
if __name__ == '__main__':
    fix_seed = 3407
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TriFi & Baselines')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='ETTh1_mask_0.125', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Bi_FI',
                        help='model name, options: [Tri_FI, Bi_FI, Autoformer, Transformer, TimesNet, iTransformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int,  help='input sequence length')
    parser.add_argument('--label_len', type=int,  help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # ==========================================================
    # 🚀 核心新增: TriFi 架构特有的非线性动力学与图网络参数
    # ==========================================================
    parser.add_argument('--tau', type=int, default=1, help='延迟时间 (Time Delay for PSR)')
    parser.add_argument('--m_dim', type=int, default=3, help='嵌入维数 (Embedding Dimension for PSR)')
    parser.add_argument('--k', type=int, default=3, help='KNN动态构图的近邻数 (Number of nearest neighbors)')
    # ==========================================================

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder', default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    
    # 🚀 改造点: 默认 Loss 从 MSE 修改为 MAE，击碎均值回归陷阱
    parser.add_argument('--loss', type=str, default='MAE', help='loss function (MSE/MAE/Huber)')
    
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    args = parser.parse_args()
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

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
            # 🚀 改造 1：任务名称 + 数据集 + 启动时间戳 (精确到秒)
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            setting = f"{args.task_name}_{args.data}_{current_time}"

            # 🚀 改造 2：在正式训练前，预先创建测试结果文件夹，并把当前的所有 args 存入 json
            res_path = os.path.join('./test_results', setting)
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            
            # 将 args 转化为字典并保存为 args_config.json
            config_file = os.path.join(res_path, 'args_config.json')
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(vars(args), f, indent=4, ensure_ascii=False)
            print(f"✅ 超参数已自动保存至: {config_file}")

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            
            # 自动化探针逻辑保持不变 ...
            if args.model == 'Tri_FI':
                # ... (保留您原有的探针打印代码) ...
                pass
            torch.cuda.empty_cache()
    else:
        ii = 0
        # 🚀 同样替换测试模式下的命名逻辑 (测试模式仅生成时间戳，不保存新 config)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        setting = f"{args.task_name}_{args.data}_{current_time}_test_only"

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
    