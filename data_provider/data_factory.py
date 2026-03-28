from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, Dataset_Solar
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute, # 🚀 对应 data_loader 中的类
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'Solar': Dataset_Solar
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag, drop_last, b_size = False, True, 1
    else:
        shuffle_flag, drop_last, b_size = True, True, args.batch_size

    if args.task_name in ['anomaly_detection', 'classification']:
        # 异常检测保持旧逻辑
        data_set = Data(root_path=args.root_path, win_size=args.seq_len, flag=flag)
        data_loader = DataLoader(data_set, batch_size=b_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last)
    else:
        # 🚀 预测任务：统一注入 PSR 参数
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=args.freq,
            tau=getattr(args, 'tau', 1),
            m_dim=getattr(args, 'm_dim', 3)
        )
        data_loader = DataLoader(
            data_set, 
            batch_size=b_size, 
            shuffle=shuffle_flag, 
            num_workers=args.num_workers, 
            drop_last=drop_last,
            pin_memory=True,  # 🚀 RTX 6000 提效关键
            persistent_workers=True if args.num_workers > 0 else False
        )
    return data_set, data_loader