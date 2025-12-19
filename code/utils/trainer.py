import torch.optim as optim
import os
import torch
import torch.nn as nn
from .metric import Metrics
from .early_stopping import EarlyStopping
from .save_result import save_results_to_csv
from .plot_results import PlotResults
from .Initial_tools import Initial_optimizer, Initial_scheduler
import pandas as pd
from .performance_per_class import save_per_class_metrics_to_csv, build_per_class_values, build_per_class_columns

# 模型训练和测试
class ModelTrainer:
    def __init__(self, model, args, results_path):
        self.args = args
        self.results_path = results_path
        self.device = args.device
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #模型
        self.model = model.to(self.device)

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            self.scheduler = None

        elif args.optimizer == 'SGD':
            # 优化器参数设置
            opti_settings = {
                'params': self.model.parameters(),
                'lr': args.optim_params['lr'],
                'momentum': args.optim_params['momentum'],
                'weight_decay': args.optim_params['weight_decay']
            }
            sche_settings = args.scheduler_params
            self.optimizer = Initial_optimizer(opti_settings, args)
            self.scheduler = Initial_scheduler(self.optimizer, sche_settings, args)

        #初始化损失函数和损失记录
        self.criterion = nn.CrossEntropyLoss()
        self.best_loss = float('inf')
        self.train_loss = []
        self.val_loss = []
        self.test_balanced_loss = []
        self.test_imbalanced_loss = []

    def train(self,train_loader,val_loader,test_balanced_loader, test_imbalanced_loader,settings,train_iter):
        early_stopping = EarlyStopping(args=self.args, settings=settings, train_iter=train_iter,patience=self.args.patience)

        for epoch in range(self.args.epochs):
            self.model.train()
            if self.args.optimizer=='SGD':
                self.scheduler.step()
            for batch_idx, (data, target) in enumerate(train_loader):

                # 梯度清零
                self.optimizer.zero_grad()

                output = self.model(data)

                loss = self.criterion(output, target)  # 计算损失   `
                self.train_loss.append(loss.item())
                loss.backward()  # 反向传播

                # 参数更新
                self.optimizer.step()
                #self.scheduler.step()

                if batch_idx % 2 == 0: #100
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,batch_idx * len(data),
                                        len(train_loader.dataset),100 * batch_idx / len(train_loader),loss.item()))
                                        # 轮数  训练样本数  样本总数  进度
                    # break

            #均衡测试集
            aver_test_balanced_loss = self.val_model(test_balanced_loader)
            self.test_balanced_loss.append(aver_test_balanced_loss)
            print('Train Epoch: {}, balanced test loss {:.6f}'.format(epoch, aver_test_balanced_loss))
            #非均衡测试集
            if test_imbalanced_loader is not None:
                aver_test_imbalanced_loss = self.val_model(test_imbalanced_loader)
                self.test_imbalanced_loss.append(aver_test_imbalanced_loss)
                print('Train Epoch: {}, imbalanced test loss {:.6f}'.format(epoch, aver_test_imbalanced_loss))

            best_model_pth_file = self.results_path +'checkpoints/{}_best_model_iter{}.pth'.format(settings,train_iter)
            if self.args.use_val:
                aver_val_loss = self.val_model(val_loader)
                print('Train Epoch: {}, validation loss:{:.6f}'.format(epoch, aver_val_loss))
                self.val_loss.append(aver_val_loss)

                if self.args.use_early_stop:
                    early_stopping(aver_val_loss, self.model, best_model_pth_file)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

        if not self.args.use_val or not self.args.use_early_stop:
            self._save_model(settings, train_iter)

        # 写入训练损失和测试损失
        self._save_loss_to_file('train', self.train_loss, settings, train_iter)
        self._save_loss_to_file('test_balanced', self.test_balanced_loss, settings, train_iter)
        if test_imbalanced_loader is not None:
            self._save_loss_to_file('test_imbalanced', self.test_imbalanced_loss, settings, train_iter)

        # 如果使用验证集，则写入验证损失
        if self.args.use_val:
            self._save_loss_to_file('val', self.val_loss, settings, train_iter)

        ### 加载最优模型 (确保test的时候self.model出于最优的状态)
        self.model.load_state_dict(torch.load(best_model_pth_file, weights_only=True))

    def _save_loss_to_file(self, flag, loss_value, settings, train_iter):
        file_path = self.results_path + f"loss/{settings}_{flag}_loss_iter{train_iter}.txt"
        with open(file_path, 'w') as f: #w：覆盖模式  a：追加模式
            f.write(str(loss_value))

    def val_model(self,val_loader):  # 定义针对验证集的函数
        self.model.eval()
        val_add_loss = 0
        with torch.no_grad():        # 禁用梯度计算
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                val_add_loss += self.criterion(output, target).item()
            aver_val_loss = val_add_loss / len(val_loader)
        self.model.train()
        return aver_val_loss

    def test(self, test_loader, settings, train_iter, flag):
        file_path = self.results_path + f"csv/{settings}_pred_and_tar_iter{train_iter}.csv"
        metrics = Metrics()
        self.model.eval()

        # 初始化三个列表，用于存储所有 batch 的结果
        preds, targets, outputs = [], [], []

        with torch.no_grad():  # 禁用梯度计算
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                pred = output.argmax(dim=1, keepdim=False)  # 找出每个样本预测概率最高的类别索引
                metrics.update(pred, target)

                # 保存每个 batch 的预测结果到内存
                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
                outputs.append(output.cpu().numpy())

        # 一次性将所有结果合并并保存到 CSV
        save_results_to_csv(flag, preds, targets, outputs, settings, train_iter, self.results_path,
                            self.args.num_classes)

        # ========= 新增：per-class 指标 & 保存 =========2025.12.9
        per_class_df, cm = metrics.compute_per_class(self.args.num_classes)
        save_per_class_metrics_to_csv(
            per_class_df, cm,
            settings=settings, train_iter=train_iter, result_path=self.results_path
        )

        # 计算并输出指标
        accuracy, precision, recall, f1 = metrics.compute()
        test_metric = [accuracy, precision, recall, f1]
        print(f'Test set: Accuracy: {accuracy:.6f}, Precision: {precision:.6f},'
              f'Recall: {recall:.6f}, F1 Score: {f1:.6f}')

        # ========= 新增：计算 per-class 指标并铺平成列 =========2025.12.09
        per_class_df, cm = metrics.compute_per_class(self.args.num_classes)

        per_class_columns = build_per_class_columns(self.args.num_classes)
        per_class_values = build_per_class_values(per_class_df, self.args.num_classes)
        # ===================================================

        results_summary_path = self.results_path + '/{}_{}_Summary_Results.csv'.format(self.args.cross_cond,self.args.model_name)
        optim_params_str = ", ".join(f"{k}:{v}" for k, v in self.args.optim_params.items())
        sche_params_str = ", ".join(f"{k}:{v}" for k, v in self.args.scheduler_params.items())
        # 总体指标
        test_metric = [accuracy, precision, recall, f1]
        # ====== data_info / columns_info（追加 per-class）======2025.12.09
        data_info = [(
            settings, self.args.use_val, self.args.add_noise, flag,
            self.args.optimizer, optim_params_str, sche_params_str,
            self.args.batch_size, self.args.epochs,
            *test_metric,
            *per_class_values
        )]
        columns_info = [
            'settings', 'use_val', 'add_noise', 'data_type',
            'optimizer', 'optim_settings', 'scheduler_settings',
            'bs', 'epochs',
            'acc', 'prec', 'rec', 'f1',
            *per_class_columns
        ]
        # =======================================================

        if os.path.exists(results_summary_path):
            results = pd.read_csv(results_summary_path, index_col=0, header=0)
            results = pd.concat([results, pd.DataFrame(data_info, columns=columns_info)], axis=0)
        else:
            results = pd.DataFrame(data_info, index=[train_iter], columns=columns_info)
        results.to_csv(results_summary_path, index=True, header=True)

        # 可选：保存每次 run 的 per-class 明细 (可删，因为前面总表里保存过一次了)
        per_class_detail_path = self.results_path + f"/csv/{settings}_per_class_metrics_iter{train_iter}.csv"
        os.makedirs(os.path.dirname(per_class_detail_path), exist_ok=True)
        per_class_df.to_csv(per_class_detail_path, index=False)

        PlotResults(flag, args=self.args, settings=settings, train_iter=train_iter, result_path=self.results_path, confusion_fig=True, loss_fig=True)

        self.model.train()
        torch.cuda.empty_cache()

    def _save_model(self,settings,train_iter):
        print(f"Saving new best model with loss: {self.best_loss:.6f}")
        torch.save(self.model.state_dict(), self.results_path +
                   'checkpoints/{}_best_model_iter{}.pth'.format(settings,train_iter))
