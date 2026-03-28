import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
    # lr_adjust = {epoch: 7e-5}
    if args.lradj == 'type1':
        # 激进指数衰减：每 1 轮减半
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        
    elif args.lradj == 'type2':
        # 硬编码阶梯衰减
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        
    elif args.lradj == 'type3':
        # 缓慢指数衰减：每 3 轮打 8 折
        lr_adjust = {epoch: args.learning_rate * (0.8 ** ((epoch - 1) // 3))}
        
    # ==========================================
    # 🌟 针对 WTH (低维混沌系统) 定制：余弦退火 (Cosine Annealing)
    # ==========================================
    elif args.lradj == 'cosine':
        # 逻辑：呈现平滑的半个钟罩形曲线。前期下降缓慢（多探索），中期加速下降，末期极缓（稳稳落入极小值）
        # 依赖 args.train_epochs 来计算总周期
        lr = args.learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch - 1) / args.train_epochs))
        lr_adjust = {epoch: lr}
        
    # ==========================================
    # 🧱 针对 Traffic (高维空间图网) 定制：延迟阶梯衰减 (Delayed Step)
    # ==========================================
    elif args.lradj == 'slow_step':
        # 逻辑：Traffic 有 862 个节点，GNN 寻址极难。前 10 轮绝不降速，让其充分建立路网拓扑边；
        # 之后每隔 10 轮才减半一次 (10, 20, 30...)
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 10))}

    # 🛡️ 安全兜底：如果类型不匹配，保持原学习率不崩溃
    else:
        lr_adjust = {epoch: args.learning_rate}

    # 执行更新
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


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
