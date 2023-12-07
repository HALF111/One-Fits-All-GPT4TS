import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime
from distutils.util import strtobool
import pandas as pd

from utils.metrics import metric

import os
import time
import copy
import math
import random
import warnings

plt.switch_backend('agg')

# 调整学习率的策略
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    # if args.decay_fac is None:
    #     args.decay_fac = 0.5
    # if args.lradj == 'type1':
    #     lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    # elif args.lradj == 'type2':
    #     lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    if args.lradj =='type1':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj =='type2':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    elif args.lradj =='type4':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch) // 1))}
    else:
        args.learning_rate = 1e-4
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


# 验证过程
def vali(model, vali_data, vali_loader, criterion, args, device, itr):
    total_loss = []
    # 不同模型eval的层不同
    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
        model.eval()
    else:
        model.in_layer.eval()
        model.out_layer.eval()
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # print(batch_x.shape)
            outputs = model(batch_x, itr)
            # print(outputs.shape)
            
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
    
    total_loss = np.average(total_loss)
    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
        model.train()
    else:
        model.in_layer.train()
        model.out_layer.train()
    
    return total_loss

def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))

# 测试过程
def test(model, test_data, test_loader, args, device, itr, setting=None):
    preds = []
    trues = []
    # mases = []

    # 全部模型都调成eval
    model.eval()
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            
            # print(i, batch_x.shape, batch_y.shape)
            
            # outputs_np = batch_x.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_input_itr{}_{}.npy".format(itr, i), outputs_np)
            # outputs_np = batch_y.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_true_itr{}_{}.npy".format(itr, i), outputs_np)

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            
            # print(batch_x.shape)
            outputs = model(batch_x[:, -args.seq_len:, :], itr)
            # print(outputs.shape)
            
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)

    preds = np.array(preds)
    trues = np.array(trues)
    # mases = np.mean(np.array(mases))
    print('test shape:', preds.shape, trues.shape)
    
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    # 打印在测试集上的结果
    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    # print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}, mases:{:.4f}'.format(mae, mse, rmse, smape, mases))
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    # 把结果保存一下
    # result save
    import os
    if setting is not None:
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

    return mse, mae


# 做TTA的test过程
def test_TTA(model, test_data, test_loader, args, device, itr, setting=None, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1, weights_given=None, adapted_degree="small", weights_from="test"):
    preds = []
    trues = []
    # mases = []
    
    data_len = len(test_data)
    
    # 再从checkpoints中把模型读出来，保证和train完的模型是一致的？
    if test:
        print('loading model from checkpoint !!!')
        model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

    # 全部模型都调成eval
    model.eval()
    
    # 记录中间一些重要的值
    a1, a2, a3, a4 = [], [], [], []
    all_angels = []
    all_distances = []
    error_per_pred_index = [[] for i in range(args.pred_len)]
    
    criterion = nn.MSELoss()  # 使用MSELoss
    test_time_start = time.time()
    
    
    # 判断哪些channels是有周期性的
    data_path = args.data_path
    if "ETTh1" in data_path: selected_channels = [1,3]  # [1,3, 2,4,5,6]
    # if "ETTh1" in data_path: selected_channels = [7]
    # elif "ETTh2" in data_path: selected_channels = [1,3,7]
    elif "ETTh2" in data_path: selected_channels = [7]
    elif "ETTm1" in data_path: selected_channels = [1,3, 2,4,5]
    elif "ETTm2" in data_path: selected_channels = [1,7, 3]
    elif "illness" in data_path: selected_channels = [1,2, 3,4,5]
    # elif "illness" in data_path: selected_channels = [6,7]
    # elif "weather" in data_path: selected_channels = [17,18,19, 5,8,6,13,20]  # [2,3,11]
    elif "weather" in data_path: selected_channels = [17,18,19]
    # elif "weather" in data_path: selected_channels = [5,8,6,13,20]
    # elif "weather" in data_path: selected_channels = [1,4,7,9,10]
    else: selected_channels = list(range(1, args.c_out))
    for channel in range(len(selected_channels)):
        selected_channels[channel] -= 1  # 注意这里要读每个item变成item-1，而非item

    # 判断各个数据集的周期是多久
    if "ETTh1" in data_path: period = 24
    elif "ETTh2" in data_path: period = 24
    elif "ETTm1" in data_path: period = 96
    elif "ETTm2" in data_path: period = 96
    elif "electricity" in data_path: period = 24
    elif "traffic" in data_path: period = 24
    elif "illness" in data_path: period = 52.142857
    elif "weather" in data_path: period = 144
    elif "Exchange" in data_path: period = 1
    else: period = 1
    
    
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        
        # 原来有的两个步骤
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float()
        
        # print(i, batch_x.shape, batch_y.shape)
        
        # outputs_np = batch_x.cpu().numpy()
        # np.save("emb_test/ETTh2_192_test_input_itr{}_{}.npy".format(itr, i), outputs_np)
        # outputs_np = batch_y.cpu().numpy()
        # np.save("emb_test/ETTh2_192_test_true_itr{}_{}.npy".format(itr, i), outputs_np)
        
        # 从model拷贝下来cur_model，并设置为train模式
        # model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
        cur_model = copy.deepcopy(model)
        # cur_model.train()
        cur_model.eval()
        
        
        if is_training_part_params:
            params = []
            names = []
            cur_model.requires_grad_(False)
            # print(list(cur_model.named_modules()))
            for n_m, m in cur_model.named_modules():
                # if "decoder.projection" == n_m:
                if "out_layer" in n_m:
                    m.requires_grad_(True)
                    for n_p, p in m.named_parameters():
                        if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{n_m}.{n_p}")

            # Adam优化器
            # model_optim = optim.Adam(params, lr=args.learning_rate*10 / (2**test_train_num))  # 使用Adam优化器
            lr = args.learning_rate * args.adapted_lr_times
            # model_optim = optim.Adam(params, lr=lr)  # 使用Adam优化器
            
            # 普通的SGD优化器？
            model_optim = optim.SGD(params, lr=lr)
        else:
            cur_model.requires_grad_(True)
            # model_optim = optim.Adam(cur_model.parameters(), lr=args.learning_rate*10 / (2**test_train_num))
            # model_optim = optim.Adam(cur_model.parameters(), lr=args.learning_rate)
            lr = args.learning_rate * args.adapted_lr_times
            model_optim = optim.SGD(cur_model.parameters(), lr=lr)
        
        
        # tmp loss
        # cur_model.eval()
        seq_len = args.seq_len
        pred_len = args.pred_len
        adapt_start_pos = args.adapt_start_pos
        if not args.use_nearest_data or args.use_further_data:
            pred, true = _process_one_batch_with_model(cur_model, test_data,
                batch_x[:, -seq_len:, :], batch_y, 
                batch_x_mark[:, -seq_len:, :], batch_y_mark,
                args, itr, device)
        else:
            pred, true = _process_one_batch_with_model(cur_model, test_data,
                batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark,
                args, itr, device)
        if args.adapt_part_channels:
            pred = pred[:, :, selected_channels]
            true = true[:, :, selected_channels]
        # 获取adaptation之前的loss
        # print(i, pred.shape, true.shape)
        loss_before_adapt = criterion(pred, true)
        a1.append(loss_before_adapt.item())
        # cur_model.train()
        
        
        # 先用原模型的预测值和标签值之间的error，做反向传播之后得到的梯度值gradient_0
        # 并将这个gradient_0作为标准答案
        # 然后，对测试样本做了adaptation之后，会得到一个gradient_1
        # 那么对gradient_1和gradient_0之间做对比，
        # 就可以得到二者之间的余弦值是多少（方向是否一致），以及长度上相差的距离有多少等等。
        # params_answer = get_answer_grad(is_training_part_params, use_adapted_model,
        #                                         lr, test_data, 
        #                                         batch_x, batch_y, batch_x_mark, batch_y_mark,
        #                                         setting)
        if use_adapted_model:
            seq_len = args.seq_len
            pred_len = args.pred_len
            adapt_start_pos = args.adapt_start_pos
            if not args.use_nearest_data or args.use_further_data:
                pred_answer, true_answer = _process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -seq_len:, :], batch_y, 
                    batch_x_mark[:, -seq_len:, :], batch_y_mark,
                    args, itr, device)
            else:
                pred_answer, true_answer = _process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                    batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark,
                    args, itr, device)
        
        if args.adapt_part_channels:
            pred_answer = pred_answer[:, :, selected_channels]
            true_answer = true_answer[:, :, selected_channels]
        # criterion = nn.MSELoss()  # 使用MSELoss
        # 计算MSE loss
        loss_ans_before = criterion(pred_answer, true_answer)
        loss_ans_before.backward()
        
        w_T = params[0].grad.T  # 先对weight参数做转置
        b = params[1].grad.unsqueeze(0)  # 将bias参数扩展一维
        params_answer = torch.cat((w_T, b), 0)  # 将w_T和b参数concat起来
        params_answer = params_answer.ravel()  # 最后再展开成一维的，就得到了标准答案对应的梯度方向

        model_optim.zero_grad()  # 清空梯度
        
        
        
        
        # 选择出合适的梯度
        # 注意：这里是减去梯度，而不是加上梯度！！！！！
        # selected_channels = selected_channels

        # 再获得未被选取的unselected_channels
        unselected_channels = list(range(args.c_out))
        for item in selected_channels:
            unselected_channels.remove(item)
        

        # 在这类我们需要先对adaptation样本的x和测试样本的x之间的距离做对比
        import torch.nn.functional as F
        
        if args.adapt_part_channels:  
            test_x = batch_x[:, -seq_len:, selected_channels].reshape(-1)
        else:
            test_x = batch_x[:, -seq_len:, :].reshape(-1)
        
        distance_pairs = []
        for ii in range(args.test_train_num):
            # 只对周期性样本计算x之间的距离
            if args.adapt_cycle:
                # 为了计算当前的样本和测试样本间时间差是否是周期的倍数
                # 我们先计算时间差与周期相除的余数
                if 'illness' in args.data_path:
                    import math
                    cycle_remainer = math.fmod(args.test_train_num-1 + args.pred_len - ii, period)
                cycle_remainer = (args.test_train_num-1 + args.pred_len - ii) % period
                # 定义判定的阈值
                threshold = period // 12
                # 如果余数在[-threshold, threshold]之间，那么考虑使用其做fine-tune
                # 否则的话不将其纳入计算距离的数据范围内
                if cycle_remainer > threshold or cycle_remainer < -threshold:
                    continue
                
            if args.adapt_part_channels:
                lookback_x = batch_x[:, ii : ii+seq_len, selected_channels].reshape(-1)
            else:
                lookback_x = batch_x[:, ii : ii+seq_len, :].reshape(-1)
            dist = F.pairwise_distance(test_x, lookback_x, p=2).item()
            distance_pairs.append([ii, dist])

        # 先按距离从小到大排序
        cmp = lambda item: item[1]
        distance_pairs.sort(key=cmp)

        # 筛选出其中最小的selected_data_num个样本出来
        selected_distance_pairs = distance_pairs[:args.selected_data_num]
        selected_indices = [item[0] for item in selected_distance_pairs]
        selected_distances = [item[1] for item in selected_distance_pairs]
        # print(f"selected_distance_pairs is: {selected_distance_pairs}")

        all_distances.append(selected_distances)

        # params_adapted = torch.zeros((1)).to(device)
        cur_grad_list = []

        # 开始训练
        for epoch in range(test_train_epochs):

            gradients = []
            accpted_samples_num = set()

            # num_of_loss_per_update = 1
            mean_loss = 0

            for ii in selected_indices:

                model_optim.zero_grad()

                seq_len = args.seq_len
                label_len = args.label_len
                pred_len = args.pred_len

                # batch_x.requires_grad = True
                # batch_x_mark.requires_grad = True

                pred, true = _process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                    batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :],
                    args, itr, device)

                # 这里当batch_size为1还是32时
                # pred和true的size可能为[1, 24, 7]或[32, 24, 7]
                # 但是结果的loss值均只包含1个值
                # 这是因为criterion为MSELoss，其默认使用mean模式，会对32个loss值取一个平均值

                if args.adapt_part_channels:
                    pred = pred[:, :, selected_channels]
                    true = true[:, :, selected_channels]
                
                # 判断是否使用最近的数据
                if not args.use_nearest_data or args.use_further_data:
                    loss = criterion(pred, true)
                else:
                    data_used_num = (args.test_train_num - (ii+1)) + args.adapt_start_pos
                    if data_used_num < pred_len:
                        loss = criterion(pred[:, :data_used_num, :], true[:, :data_used_num, :])
                    else:
                        loss = criterion(pred, true)
                    # loss = criterion(pred, true)

                # loss = criterion(pred, true)
                mean_loss += loss

                
                # # pass
                # loss.backward()
                # model_optim.step()

                loss.backward()
                w_T = params[0].grad.T
                b = params[1].grad.unsqueeze(0)
                params_tmp = torch.cat((w_T, b), 0)
                original_shape = params_tmp.shape
                # print("original_shape:", original_shape)
                params_tmp = params_tmp.ravel()

                # 将该梯度存入cur_grad_list中
                cur_grad_list.append(params_tmp.detach().cpu().numpy())

                model_optim.zero_grad()

                # 记录逐样本做了adaptation之后的loss
                # mean_loss += tmp_loss
                # mean_loss += loss
        
        
        # 定义一个权重和梯度相乘函数
        def calc_weighted_params(params, weights):
            results = 0
            for i in range(len(params)):
                results += params[i] * weights[i]
            return results
        
        # 权重分别乘到对应的梯度上
        if weights_given:
            weighted_params = calc_weighted_params(cur_grad_list, weights_given)
        else:
            weights_all_ones = [1 for i in range(args.test_train_num)]
            weighted_params = calc_weighted_params(cur_grad_list, weights_all_ones)
        
        # 将weighted_params从np.array转成tensor
        weighted_params = torch.tensor(weighted_params)
        weighted_params = weighted_params.to(device)


        # 计算标准答案的梯度params_answer和adaptation加权后的梯度weighted_params之间的角度
        import math
        # print(weighted_params.shape, params_answer.shape)
        product = torch.dot(weighted_params, params_answer)
        product = product / (torch.norm(weighted_params) * torch.norm(params_answer))
        angel = math.degrees(math.acos(product))
        all_angels.append(angel)
        

        # 还原回原来的梯度
        # 也即将weighted_params变回w_grad和b_grad
        weighted_params = weighted_params.reshape(original_shape)
        w_grad_T, b_grad = torch.split(weighted_params, [weighted_params.shape[0]-1, 1])
        w_grad = w_grad_T.T  # (7, 512)
        b_grad = b_grad.squeeze(0)  # (7)


        # 设置新参数为原来参数 + 梯度值
        from torch.nn.parameter import Parameter
        cur_lr = args.learning_rate * args.adapted_lr_times

        # 将未选择的channels上的梯度置为0
        if args.adapt_part_channels:
            w_grad[unselected_channels, :] = 0
            b_grad[unselected_channels] = 0

        # 注意：这里是减去梯度，而不是加上梯度！！！！！
        cur_model.out_layer.weight = Parameter(cur_model.out_layer.weight - w_grad * cur_lr)
        cur_model.out_layer.bias = Parameter(cur_model.out_layer.bias - b_grad * cur_lr)


        # mean_loss = mean_loss / test_train_num
        mean_loss = mean_loss / args.selected_data_num
        a2.append(mean_loss.item())
        
        # mean_loss.backward()
        # model_optim.step()


        seq_len = args.seq_len
        label_len = args.label_len
        pred_len = args.pred_len
        tmp_loss = 0
        for ii in selected_indices:
            pred, true = _process_one_batch_with_model(cur_model, test_data,
                batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :],
                args, itr, device)
            if args.adapt_part_channels:
                pred = pred[:, :, selected_channels]
                true = true[:, :, selected_channels]
            tmp_loss += criterion(pred, true)
        tmp_loss = tmp_loss / args.selected_data_num
        a3.append(tmp_loss.item())

        a3.append(0)



        cur_model.eval()

        seq_len = args.seq_len
        pred_len = args.pred_len
        adapt_start_pos = args.adapt_start_pos
        if use_adapted_model:
            if not args.use_nearest_data or args.use_further_data:
                pred, true = _process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -seq_len:, :], batch_y, 
                    batch_x_mark[:, -seq_len:, :], batch_y_mark,
                    args, itr, device)
            else:
                pred, true = _process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                    batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark,
                    args, itr, device)
        # else:
        #     # pred, true = _process_one_batch_with_model(model, test_data,
        #     #     batch_x[:, -args.seq_len:, :], batch_y, 
        #     #     batch_x_mark[:, -args.seq_len:, :], batch_y_mark)
        #     if not args.use_nearest_data or args.use_further_data:
        #         pred, true = _process_one_batch_with_model(cur_model, test_data,
        #             batch_x[:, -seq_len:, :], batch_y, 
        #             batch_x_mark[:, -seq_len:, :], batch_y_mark)
        #     else:
        #         pred, true = _process_one_batch_with_model(cur_model, test_data,
        #             batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
        #             batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)

        # 如果需要筛选部分维度，那么做一次筛选：
        if args.adapt_part_channels:
            pred = pred[:, :, selected_channels]
            true = true[:, :, selected_channels]

        # 获取adaptation之后的loss
        loss_after_adapt = criterion(pred, true)
        a4.append(loss_after_adapt.item())
        
        
        
        # # print(batch_x.shape)
        # outputs = model(batch_x[:, -args.seq_len:, :], itr)
        # # print(outputs.shape)
        
        # # encoder - decoder
        # outputs = outputs[:, -args.pred_len:, :]
        # batch_y = batch_y[:, -args.pred_len:, :].to(device)
        
        
        # 这里别忘记给batch_y做一下裁剪，保留最后的pred_len的部分
        pred = pred[:, -args.pred_len:, :]
        batch_y = batch_y[:, -args.pred_len:, :].to(device)

        # 将预测值转换为numpy并保存下来
        # pred = outputs.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        true = batch_y.detach().cpu().numpy()
        
        preds.append(pred)
        trues.append(true)
        
        
        # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计
        pred_len = args.pred_len
        for index in range(pred_len):
            # cur_pred = pred.detach().cpu().numpy()[0][index]
            # cur_true = true.detach().cpu().numpy()[0][index]
            cur_pred = pred[0][index]
            cur_true = true[0][index]
            cur_error = np.mean((cur_pred - cur_true) ** 2)
            error_per_pred_index[index].append(cur_error)


        if (i+1) % 100 == 0 or (data_len - i) < 100 and (i+1) % 10 == 0:
            print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
            print(gradients)
            tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
            tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
            tmp_mae, tmp_mse, *_ = metric(tmp_p, tmp_t)
            print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))
            
            avg_1, avg_2, avg_3, avg_4 = 0, 0, 0, 0
            avg_angel = 0
            num = len(a1)
            for iii in range(num):
                avg_1 += a1[iii]; avg_2 += a2[iii]; avg_3 += a3[iii]; avg_4 += a4[iii]; avg_angel += all_angels[iii]
            avg_1 /= num; avg_2 /= num; avg_3 /= num; avg_4 /= num; avg_angel /= num
            print("1.before_adapt, 2.adapt_sample, 3.adapt_sample_after, 4.after_adapt, 5.angel_between_answer")
            print("average:", avg_1, avg_2, avg_3, avg_4, avg_angel)
            print("last one:", a1[-1], a2[-1], a3[-1], a4[-1], all_angels[-1])

            printed_selected_channels = [item+1 for item in selected_channels]
            print(f"adapt_part_channels: {args.adapt_part_channels}, and adapt_cycle: {args.adapt_cycle}")
            print(f"first 25th selected_channels: {printed_selected_channels[:25]}")
            print(f"selected_distance_pairs are: {selected_distance_pairs}")


        # if i % 20 == 0:
        #     input = batch_x.detach().cpu().numpy()
        #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
        #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
        #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        cur_model.eval()
        # cur_model.cpu()
        del cur_model
        torch.cuda.empty_cache()
            

    preds = np.array(preds)
    trues = np.array(trues)
    # mases = np.mean(np.array(mases))
    print('test shape:', preds.shape, trues.shape)
    
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    # 打印在测试集上的结果
    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    # print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}, mases:{:.4f}'.format(mae, mse, rmse, smape, mases))
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    # # 把结果保存一下
    # # result save
    # import os
    # if setting is not None:
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     np.save(folder_path + 'pred.npy', preds)
    #     np.save(folder_path + 'true.npy', trues)
    #     # np.save(folder_path + 'x.npy', inputx)

    return mse, mae



def _process_one_batch_with_model(model, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, args, itr, device, return_mid_embedding=False):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    
    # print(batch_x.shape)
    outputs = model(batch_x[:, -args.seq_len:, :], itr)
    # print(outputs.shape)
    
    f_dim = -1 if args.features=='MS' else 0

    outputs = outputs[:, -args.pred_len:, f_dim:]
    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

    # outputs为我们预测出的值pred，而batch_y则是对应的真实值true
    # if return_mid_embedding:
    #     return outputs, batch_y, mid_embedding
    # else:
    #     return outputs, batch_y
    return outputs, batch_y

