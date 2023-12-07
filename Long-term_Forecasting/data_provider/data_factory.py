from data_provider.data_loader import Dataset_Custom, Dataset_Pred, Dataset_TSF, Dataset_ETT_hour, Dataset_ETT_minute
from data_provider.data_loader import Dataset_Custom_Test, Dataset_ETT_hour_Test, Dataset_ETT_minute_Test
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'tsf_data': Dataset_TSF,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
}

data_dict_at_test_time = {
    'custom': Dataset_Custom_Test,
    # 'tsf_data': Dataset_TSF,
    'ett_h': Dataset_ETT_hour_Test,
    'ett_m': Dataset_ETT_minute_Test,
}


def data_provider(args, flag, drop_last_test=True, train_all=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    max_len = args.max_len

    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


def data_provider_at_test_time(args, flag, drop_last_test=True, train_all=False):
    # 要改成从data_dict_at_test_time中获取
    Data = data_dict_at_test_time[args.data]
    
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    max_len = args.max_len

    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        # batch_size = args.batch_size
        freq = args.freq
        
        # 注意：因为我们要做TTT/TTA，所以一定要把batch_size设置成1 ！！！
        # batch_size = 32
        batch_size = args.adapted_batch_size
        # batch_size = 1
        # batch_size = 4
        # batch_size = 2857
        # batch_size = 256
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    elif flag == 'val':
        # shuffle_flag = True
        shuffle_flag = False
        drop_last = drop_last_test
        # batch_size = args.batch_size
        freq = args.freq
        
        # 注意：因为我们要做TTT/TTA，所以一定要把batch_size设置成1 ！！！
        batch_size = args.adapted_batch_size
        # batch_size = 32
    else:
        # shuffle_flag = True
        shuffle_flag = False
        drop_last = True
        # batch_size = args.batch_size
        batch_size = args.adapted_batch_size
        freq = args.freq
    
        
    # 再多几个use_nearest_data等参数
    use_nearest_data = args.use_nearest_data
    use_further_data = args.use_further_data
    adapt_start_pos = args.adapt_start_pos

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all,
        
        # 别忘了下面这几个参数
        test_train_num = args.test_train_num,
        use_nearest_data=use_nearest_data,
        use_further_data=use_further_data,
        adapt_start_pos=adapt_start_pos
    )
    
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    return data_set, data_loader

