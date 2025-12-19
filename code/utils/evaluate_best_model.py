import torch
from .metric import Metrics
from .save_result import save_results_to_csv
from .plot_results import PlotResults
import os

class Evaluate_best_model:
    def __init__(self, model, args, test_loader, results_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定设备运行
        self.args = args
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.results_path = results_path

    def evaluate_best_model(self,settings,best_model_pth_file,train_iter, flag):
        print("Loading best model...")
        self.model.load_state_dict(torch.load(best_model_pth_file, weights_only=True))
        self.model.eval()
        metrics = Metrics()

        # 初始化三个列表，用于存储所有 batch 的结果
        preds, targets, outputs = [], [], []

        with torch.no_grad():  # 禁用梯度计算
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=False)  # 找出每个样本预测概率最高的类别索引
                metrics.update(pred, target)

                # 保存每个 batch 的预测结果到内存
                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
                outputs.append(output.cpu().numpy())

        # 一次性将所有结果合并并保存到 CSV
        save_results_to_csv(flag, preds, targets, outputs, settings, train_iter, self.results_path, self.args.num_classes)

        # 计算并输出指标
        accuracy, precision, recall, f1 = metrics.compute()
        print(f'Test set: Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, '
              f'Recall: {recall:.6f}, F1 Score: {f1:.6f}')

        # 绘制混淆矩阵
        PlotResults(args=self.args, settings=settings, train_iter=train_iter, result_path=self.results_path, confusion_fig=True, loss_fig=False)

