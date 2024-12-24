# 创建一个示例数据集（假设是一个 DataFrame）
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, test_size=0.2, valid_size=0.1, train_size=0.7):
        self.test_size = test_size
        self.valid_size = valid_size
        self.train_size = train_size

    def _split_data(self, x, y):
        # 获取原始数据的索引
        original_indices = x.index if isinstance(x, pd.DataFrame) else np.arange(len(x))

        # 第一次划分: 训练集和测试集
        x_train, x_test, y_train, y_test, train_indices, test_indices = (
            train_test_split(
                x, y, original_indices, test_size=self.test_size, random_state=42
            )
        )

        # 第二次划分: 训练集和验证集
        x_train, x_valid, y_train, y_valid, train_indices, valid_indices = (
            train_test_split(
                x_train,
                y_train,
                train_indices,
                test_size=self.valid_size / (self.train_size + self.valid_size),
                random_state=42,
            )
        )

        # 获取训练集、验证集和测试集在原始数据中的位置
        return (
            x_train,
            x_valid,
            x_test,
            y_train,
            y_valid,
            y_test,
            train_indices,
            valid_indices,
            test_indices,
        )


# 示例数据
data = pd.DataFrame(
    {
        "feature1": range(10),
        "feature2": range(10, 20),
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }
)

# 特征和标签
X = data[["feature1", "feature2"]]
y = data["label"]

# 初始化数据划分器
splitter = DataSplitter(test_size=0.2, valid_size=0.1)

# 划分数据
(
    x_train,
    x_valid,
    x_test,
    y_train,
    y_valid,
    y_test,
    train_indices,
    valid_indices,
    test_indices,
) = splitter._split_data(X, y)

# 打印结果
print("训练集索引:", train_indices)
print("验证集索引:", valid_indices)
print("测试集索引:", test_indices)
