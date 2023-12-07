from data_provider.data_factory import data_provider
from data_provider.data_factory import data_provider_at_test_time
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from utils.tools import test_TTA
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear


import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)

parser.add_argument('--random_seed', type=int, default=2021)


# test_train_num
parser.add_argument('--test_train_num', type=int, default=10, help='how many samples to be trained during test')
parser.add_argument('--adapted_lr_times', type=float, default=1, help='the times of lr during adapted')  # adaptation时的lr是原来的lr的几倍？
parser.add_argument('--adapted_batch_size', type=int, default=1, help='the batch_size for adaptation use')  # adaptation时的数据集取的batch_size设置为多大
parser.add_argument('--test_train_epochs', type=int, default=1, help='the batch_size for adaptation use')  # adaptation时的数据集取的batch_size设置为多大
parser.add_argument('--run_train', action='store_true')
parser.add_argument('--run_test', action='store_true')
parser.add_argument('--run_select_with_distance', action='store_true')
# selected_data_num表示从过去test_train_num个样本中按照距离挑选出最小的多少个出来
# 因此这里要求必须有lookback_data_num <= test_train_num成立
parser.add_argument('--selected_data_num', type=int, default=10)

parser.add_argument('--get_grads_from', type=str, default="test", help="options:[test, val]")
parser.add_argument('--adapted_degree', type=str, default="small", help="options:[small, large]")


# 改用更近（填0）或更远（周期性）的数据做adaptation
parser.add_argument('--use_nearest_data', action='store_true')
parser.add_argument('--use_further_data', action='store_true')
# 理论上当adapt_start_pos == pred_len时，本方法与原来方法相同;
# 但是但是由于实现的原因，要求必须保证：
# 1.当use_nearest_data时，要求保证adapt_start_pos严格小于pred_len
# 2.当use_further_data时，要求保证adapt_start_pos严格大于等于pred_len
parser.add_argument('--adapt_start_pos', type=int, default=1)

parser.add_argument('--adapt_part_channels', action='store_true')
# 仅对周期性数据做fine-tuning
parser.add_argument('--adapt_cycle', action='store_true')


args = parser.parse_args()

# 季节性映射mapping
SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

mses = []
maes = []

for ii in range(args.itr):
    
    # 设置随机种子
    random_seed = args.random_seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
    #                                                                 args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
    #                                                                 args.d_ff, args.embed, ii)
    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_seed{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                    args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                    args.d_ff, args.embed, random_seed, ii)
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.freq == 0:
        args.freq = 'h'

    # 获取数据
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    # 修改1：改成TTA式的数据获取
    test_data_TTA, test_loader_TTA = data_provider_at_test_time(args, 'test')

    if args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))

    device = torch.device('cuda:0')

    time_now = time.time()
    train_steps = len(train_loader)

    # 构建模型
    if args.model == 'PatchTST':
        model = PatchTST(args, device)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(args, device)
        model.to(device)
    else:
        model = GPT4TS(args, device)
    # mse, mae = test(model, test_data, test_loader, args, device, ii)

    # Adam优化器
    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    if args.loss_func == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_func == 'smape':
        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()
            def forward(self, pred, true):
                return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
        criterion = SMAPE()
    
    # 学习率调整策略，这里使用余弦退火曲线方法
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

    # 开始训练！！！
    if args.run_train:
        for epoch in range(args.train_epochs):

            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)

                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                # 模型给出输出
                # print(batch_x.shape)
                outputs = model(batch_x, ii)
                # print(outputs.shape)

                # 去掉前面label_len的部分，只保留最后长为pred_len的一段
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    
                # 梯度反向传播，并更新参数
                loss.backward()
                model_optim.step()

            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # 每个epoch都计算下这几个loss
            train_loss = np.average(train_loss)
            vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
            # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            # 调整学习率
            if args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, args)
            # 早停策略
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        print("------------------------------------")
    
    if args.run_test:
        # 最后记录测试集上的mse和mae
        mse, mae = test(model, test_data, test_loader, args, device, ii, setting)
        print("test results - original:")
        print("mse:", mse)
        print("mae:", mae)
    
    if args.run_select_with_distance:
        # 修改2：改成做TTA的test过程
        mse, mae = test_TTA(model, test_data_TTA, test_loader_TTA, 
                            args, device, ii, setting,
                            test=1, is_training_part_params=True, use_adapted_model=True,
                            test_train_epochs=1)
        print("test results - after run_select_with_distance:")
        print("mse:", mse)
        print("mae:", mae)
        
    
#     mses.append(mse)
#     maes.append(mae)

# mses = np.array(mses)
# maes = np.array(maes)
# print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
# print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))