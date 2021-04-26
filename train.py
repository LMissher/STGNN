from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import math

from utils import log_string, loadPEMSData
from model import STGNN

parser = argparse.ArgumentParser()
# parser.add_argument('--time_slot', type = int, default = 5,
#                     help = 'a time step is 5 mins')
parser.add_argument('--P', type = int, default = 12,
                    help = 'history steps')
parser.add_argument('--Q', type = int, default = 12,
                    help = 'prediction steps')
parser.add_argument('--L', type = int, default = 1,
                    help = 'number of STAtt Blocks')
parser.add_argument('--K', type = int, default = 4,
                    help = 'number of attention heads')
parser.add_argument('--d', type = int, default = 16,
                    help = 'dims of each head attention outputs')

parser.add_argument('--train_ratio', type = float, default = 0.6,
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.2,
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,
                    help = 'testing set [default : 0.2]')
parser.add_argument('--batch_size', type = int, default = 16,
                    help = 'batch size')
parser.add_argument('--max_epoch', type = int, default = 50,
                    help = 'epoch to run')
# parser.add_argument('--patience', type = int, default = 10,
#                     help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default = 0.001,
                    help = 'initial learning rate')
parser.add_argument('--traffic_file', default = '**.npz',
                    help = 'traffic file')
parser.add_argument('--SE_file', default = '**.npy',
                    help = 'spatial emebdding file')
parser.add_argument('--model_file', default = 'PEMS',
                    help = 'save the model to disk')
parser.add_argument('--log_file', default = 'log(PEMS)',
                    help = 'log file')

args = parser.parse_args()

log = open(args.log_file, 'w')

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

log_string(log, "loading data....")

trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std = loadPEMSData(args)
SE = torch.from_numpy(SE).to(device)


log_string(log, "loading end....")

def res(model, valX, valTE, valY, mean, std):
    model.eval() # 评估模式, 这会关闭dropout
    # it = test_iter.get_iterator()
    num_val = valX.shape[0]
    pred = []
    label = []
    num_batch = math.ceil(num_val / args.batch_size)
    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(valX[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                # te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)

                y_hat = model(X)

                pred.append(y_hat.cpu().numpy()*std+mean)
                label.append(y)
    
    pred = np.concatenate(pred, axis = 0)
    label = np.concatenate(label, axis = 0)

    # print(pred.shape, label.shape)
    maes = []
    rmses = []
    mapes = []
    wapes = []

    for i in range(12):
        mae, rmse , mape, wape = metric(pred[:,i,:], label[:,i,:])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        wapes.append(wape)
        # if i == 11:
        log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f, wape: %.4f' % (i+1, mae, rmse, mape, wape))
            # print('step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
    
    mae, rmse, mape, wape = metric(pred, label)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    wapes.append(wape)
    log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f, wape: %.4f' % (mae, rmse, mape, wape))
    # print('average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
    
    return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

def train(model, trainX, trainTE, trainY, valX, valTE, valY, mean, std):
    num_train = trainX.shape[0]
    min_loss = 10000000.0
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15],
    #                                                         gamma=0.2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,    
                                    verbose=False, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=2e-6, eps=1e-08)
    
    for epoch in tqdm(range(1,args.max_epoch+1)):
        model.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        # trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        num_batch = math.ceil(num_train / args.batch_size)
        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(trainX[start_idx : end_idx]).float().to(device)
                y = torch.from_numpy(trainY[start_idx : end_idx]).float().to(device)
                # te = torch.from_numpy(trainTE[start_idx : end_idx]).to(device)

                optimizer.zero_grad()

                y_hat = model(X)

                y_d = y
                y_hat_d = y_hat


                loss = _compute_loss(y, y_hat*std+mean)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                
                train_l_sum += loss.cpu().item()
                # print(f"\nbatch loss: {l.cpu().item()}")
                n += y.shape[0]
                batch_count += 1
                pbar.update(1)
        # lr = lr_scheduler.get_lr()
        log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
              % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        # print('epoch %d, lr %.6f, loss %.4f, time %.1f sec'
        #       % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        mae, rmse, mape = res(model, valX, valTE, valY, mean, std)
        # lr_scheduler.step()
        lr_scheduler.step(mae[-1])
        if mae[-1] < min_loss:
            min_loss = mae[-1]
            torch.save(model, args.model_file)

def test(model, valX, valTE, valY, mean, std):
    model = torch.load(args.model_file)
    mae, rmse, mape = res(model, valX, valTE, valY, mean, std)
    return mae, rmse, mape

def _compute_loss(y_true, y_predicted):
        return masked_mae(y_predicted, y_true, 0.0)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        wape = np.divide(np.sum(mae), np.sum(label))
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape, wape

if __name__ == '__main__':
    maes, rmses, mapes = [], [], []
    for i in range(5):
        log_string(log, "model constructed begin....")
        model = STGNN(1, args.K*args.d, args.L, args.d).to(device)
        log_string(log, "model constructed end....")
        log_string(log, "train begin....")
        train(model, trainX, trainTE, trainY, testX, testTE, testY, mean, std)
        log_string(log, "train end....")
        mae, rmse, mape = test(model, testX, testTE, testY, mean, std)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
    log_string(log, "\n\nresults:")
    maes = np.stack(maes, 1)
    rmses = np.stack(rmses, 1)
    mapes = np.stack(mapes, 1)
    for i in range(12):
        log_string(log, 'step %d, mae %.4f, rmse %.4f, mape %.4f' % (i+1, maes[i].mean(), rmses[i].mean(), mapes[i].mean()))
        log_string(log, 'step %d, mae %.4f, rmse %.4f, mape %.4f' % (i+1, maes[i].std(), rmses[i].std(), mapes[i].std()))
    log_string(log, 'average, mae %.4f, rmse %.4f, mape %.4f' % (maes[-1].mean(), rmses[-1].mean(), mapes[-1].mean()))
    log_string(log, 'average, mae %.4f, rmse %.4f, mape %.4f' % (maes[-1].std(), rmses[-1].std(), mapes[-1].std()))
