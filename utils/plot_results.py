import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

class PlotResults:
    def __init__(self, flag, args, settings, train_iter, result_path,confusion_fig, loss_fig):
        self.flag = flag
        self.args = args
        self.settings = settings
        self.train_iter = train_iter
        self.results_path = result_path
        self._plot_loss() if loss_fig else None
        self._plot_matrix() if confusion_fig else None

    # 读取存储为txt文件的数据
    def data_read(self, txt_path):
        with open(txt_path, "r") as f:
            raw_data = f.read()
            data = raw_data[1:-1].split(", ")  # [-1:1]是为了去除文件中的前后中括号"[]"
        return np.asfarray(data, float)

    def _plot_loss(self):
        plt.figure(figsize=(10, 6))
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('train_iter')
        plt.ylabel('loss')

        colors = ['RoyalBlue', 'MediumOrchid', 'DarkOrange', 'Orchid', 'Gray']

        # 定义一个通用的损失读取函数（内置函数）
        def read_loss(file_path,train_loss_num,repeat_epoch=False):
            y_loss = self.data_read(file_path)
            if repeat_epoch is not None:
                repeat_num = train_loss_num /len(y_loss)
                y_loss = [item for item in y_loss for _ in range(int(repeat_num))]
            return range(len(y_loss)), y_loss

        # 路径生成和读取损失
        train_loss_path = os.path.join(self.results_path, f"loss/{self.settings}_train_loss_iter{self.train_iter}.txt")
        test_loss_path = os.path.join(self.results_path, f"loss/{self.settings}_test_{self.flag}_loss_iter{self.train_iter}.txt")
        x_train_loss, y_train_loss = read_loss(train_loss_path,train_loss_num=None,repeat_epoch=None)

        # 绘制训练损失和测试损失的曲线
        x_test_loss, y_test_loss = read_loss(test_loss_path,train_loss_num=len(y_train_loss),repeat_epoch=True)
        plt.plot(x_train_loss, y_train_loss, linewidth=2, linestyle="solid", label="train loss", color=colors[0])
        plt.plot(x_test_loss, y_test_loss, linewidth=2, linestyle="solid", label=f"test_{self.flag} loss", color=colors[1])

        # 如果使用验证集，绘制验证集损失
        if self.args.use_val:
            val_loss_path = os.path.join(self.results_path, f"loss/{self.settings}_val_loss_iter{self.train_iter}.txt")
            x_val_loss, y_val_loss = read_loss(val_loss_path,train_loss_num=len(y_train_loss),repeat_epoch=True)
            plt.plot(x_val_loss, y_val_loss, linewidth=2, linestyle="solid", label="val loss", color=colors[2])

        plt.legend()
        plt.title(f'{self.settings}_{self.flag}_Loss curve_{self.train_iter}')

        plt.savefig(self.results_path + f"fig/{self.settings}_results_{self.flag}_loss_iter_{self.train_iter}.jpg", bbox_inches='tight', dpi=300)
        plt.close()
        # plt.show()  # 可选择展示图像

    def _plot_matrix(self):
        df = pd.read_csv(self.results_path + f"csv/{self.settings}_{self.flag}_pred_and_tar_iter{self.train_iter}.csv")
        preds = df['Predicted'].tolist()
        targs = df['True'].tolist()
        confusion_mat = confusion_matrix(targs,preds)

        plt.figure(figsize=(8, 6))  # 图像大小 宽8英寸 高6英寸
        plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix_{self.train_iter}\n{self.settings}")
        plt.colorbar()
        tick_marks = np.arange(self.args.num_classes)
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)

        thresh = confusion_mat.max() / 2.
        for i in range(confusion_mat.shape[0]):   # 行
            for j in range(confusion_mat.shape[1]):  # 列
                plt.text(j, i, format(confusion_mat[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if confusion_mat[i, j] > thresh else "black")

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        plt.savefig(self.results_path + f"fig/{self.settings}_{self.flag}_Confusion Matrix_iter_iter{self.train_iter}.png", format='png')
        plt.close()



