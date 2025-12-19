import numpy as np
import torch

class EarlyStopping:
    def __init__(self, args, settings, train_iter, patience, delta=0):
        self.args = args
        self.settings = settings
        self.train_iter = train_iter
        self.patience = patience  # 验证损失不会再改善后，继续等待的轮次数
        self.counter = 0   # 记录自上次验证损失改善以来经过的轮次数
        self.best_score = None  # 目前最好的验证分数
        self.early_stop = False   # 是否停止训练的标志位嗯
        self.val_loss_min = np.Inf  # 记录迄今最小的验证损失
        self.delta = delta  # 允许验证损失再多少范围内波动而不视为改善
        self.best_loss = None

    def __call__(self, vali_loss, model, path):
        if self.best_loss is None:
            self.best_loss = vali_loss
            self.save_checkpoint(vali_loss, model, path)
        elif vali_loss < self.best_loss:
            self.best_loss = vali_loss
            self.save_checkpoint(vali_loss, model, path)
            self.counter = 0

        elif self.args.optimizer == 'Adam':
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered. No improvement  {self.counter} / {self.patience}.")


    def save_checkpoint(self, val_loss, model, path):  # 早停后保存模型
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).Saving model ...')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss
