import numpy as np
def standard_normal_long_tail(cls_num, n_max, imb):
    '''
    cls_num: 类别数
    n_max: 头部样本量
    imb: 不平衡因子
    '''
    # 尾部样本量
    n_min = n_max * imb  

    # 计算尾部位置 x
    x = np.sqrt(-2 * np.log(imb))
    
    # 生成类别索引映射到 [0, x]
    i = np.arange(cls_num)
    x_i = x * i / (cls_num - 1)
    
    # 计算概率密度值
    f_i = np.exp(-x_i**2 / 2) / np.sqrt(2 * np.pi)
    
    # 定义密度值范围
    f_max = 1 / np.sqrt(2 * np.pi)  # 头部密度值
    f_min = imb / np.sqrt(2 * np.pi)  # 尾部密度值
    
    # 线性映射到样本数量
    n_samples = n_min + (f_i - f_min) / (f_max - f_min) * (n_max - n_min)
    n_samples = np.round(n_samples).astype(int)
    
    return np.clip(n_samples, n_min, n_max)

# 示例参数
if __name__ == "__main__":
    cls_num = 10    # 类别数
    n_max = 1000    # 头部样本量
    imb = 0.1       # 不平衡因子

    # 生成样本分布
    img_num_per_cls = standard_normal_long_tail(cls_num=cls_num, n_max=n_max, imb=imb)
    print("样本分布:", img_num_per_cls)