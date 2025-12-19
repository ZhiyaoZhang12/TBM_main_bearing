import pandas as pd
import numpy as np

def save_results_to_csv(flag, preds,targets,softmax_outputs,settings,train_iter,results_path,num_classes):
    # 将所有预测结果、目标值和输出合并成一个完整的数组
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    softmax_outputs = np.concatenate(softmax_outputs, axis=0)

    # 根据 num_classes 参数生成列名
    label_names = [f'Class_{k}' for k in range(num_classes)]

    # 如果 softmax_outputs 不是 DataFrame，转换为 DataFrame
    if not isinstance(softmax_outputs, pd.DataFrame):
        softmax_outputs = pd.DataFrame(softmax_outputs, columns=label_names)

    # 创建结果 DataFrame
    df_results = pd.DataFrame({'Predicted': preds, 'True': targets})
    df_results = pd.concat([df_results, softmax_outputs], axis=1)
    df_results.to_csv(results_path + f"csv/{settings}_{flag}_pred_and_tar_iter{train_iter}.csv", index=False, sep=',')


