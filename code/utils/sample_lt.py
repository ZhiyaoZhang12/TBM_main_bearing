from collections import defaultdict
from sklearn.utils import resample
import numpy as np
# 根据 samples_per_class 对数据进行上下采样
def resample_data_by_class(data, labels, samples_per_class):
    """根据每个类别的样本数量对数据进行上下采样"""
    class_data = defaultdict(list)

    # 按类别分组数据
    for sample, label in zip(data, labels):
        class_data[label].append(sample)

    resampled_data = []
    resampled_labels = []

    # 对每个类别进行采样
    for class_idx, target_samples in samples_per_class.items():
        if class_idx in class_data:
            current_data = class_data[class_idx]
            if len(current_data) >= target_samples:
                # 下采样
                sampled_data = resample(current_data, replace=False, n_samples=target_samples, random_state=42)
            else:
                # 上采样
                sampled_data = resample(current_data, replace=True, n_samples=target_samples, random_state=42)
            
            resampled_data.extend(sampled_data)
            resampled_labels.extend([class_idx] * target_samples)

    return np.array(resampled_data), np.array(resampled_labels)

