import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

"""
# 数据及处理说明
删除存在“？”的行，即信息缺失的行，对于连续变量划分类别，10%作为测试集，90%作为训练集
# 表格中选取的列说明
## 因变量
income 收入
## 自变量
### 离散变量
workclass 工作类型
educational-num和education正相关，两者均表示教育程度，故只取前者
marital-status 婚姻状况
occupation 职业
relationship 家庭成员类型
race 人种
gender 性别
native-country 国籍
### 连续变量
age 年龄
capital-gain 投资收入
capital-loss 投资损失
hours-per-week 每周工作时长
"""
data = pd.read_csv("./adult.csv")
for i in data.columns:  # 遍历列，删去缺失的数据
    data = data[-data["{}".format(i)].isin(["?"])]
data = data.sample(frac=1, random_state=10).reset_index(drop=True)  # 随机打乱数据集

# 对于连续变量划分类别
# capital_gain_class = ["0", "!0"]  # capital-gain 分为0，和非0两个类别
# capital_loss_class = ["0", "!0"]  # capital-loss 分为0，和非0两个类别
age_interval = 5  # age 15-90 5为间隔 左闭右开
hours_per_week_interval = 5  # hours-per-week 0-99 5为间隔 左闭右开

# 根据划分的类别将连续变量替换为类别
data["capital-loss"] = data["capital-loss"].apply(lambda x: "0" if x < 0.01 else "!0")
data["capital-gain"] = data["capital-gain"].apply(lambda x: "0" if x < 0.01 else "!0")
data["age"] = ((data["age"] - 15) / age_interval).astype(np.int32)
data["hours-per-week"] = ((data["age"] - 15) / age_interval).astype(np.int32)

N = int(data.shape[0] * 0.9)
train_data = data.loc[0:N-1, :].reset_index(drop=True)
test_data = data.loc[N:, :].reset_index(drop=True)

"""
极大似然估计计算先验概率, 并在figure文件夹中保存，income两个类别下各个影响变量的分布柱状图
"""
probability = {}  # 储存先验概率
lamda = 1  # 使用拉普拉斯平滑
y_class = ['>50K', '<=50K']  # income的类别整合
color = ["red", "blue"]  # 画柱状图用的颜色
bar_width = 0.35  # 画柱状图，柱宽
bar_bias = [-0.55*bar_width, 0.55*bar_width]
x_total = ['workclass', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', "age", "capital-loss", "capital-gain", "hours-per-week"]  # 所有的影响变量

n_y = []  # 储存每个类别的数目
# 计算各类别的概率值
for k in range(len(y_class)):  # 对于income的类别循环
    n_y.append(train_data[train_data.income == y_class[k]].shape[0])  # 属于第k类别的个数
    probability["P(income:{})".format(y_class[k])] = (n_y[k] + lamda) / (N + len(y_class) * lamda)

# 计算变量的概率值
for x in x_total:
    x_class = list(set(train_data.loc[:, "{}".format(x)]))  # x的类别整合
    for k in range(len(y_class)):  # 对于income的类别循环
        for j in range(len(x_class)):
            n_x = train_data[(train_data["{}".format(x)] == x_class[j]) & (train_data.income == y_class[k])].shape[0]  # 第k类别中属于x变量第j类的个数
            probability["P({}:{}|income:{})".format(x, x_class[j], y_class[k])] = (n_x + lamda) / (n_y[k] + len(x_class) * lamda)
            plt.xlabel("class")
            plt.ylabel("probability")
            plt.title("{}".format(x))
            if j == 0:
                plt.bar(j+1 + bar_bias[k], probability["P({}:{}|income:{})".format(x, x_class[j], y_class[k])], width=bar_width, color=color[k], label="income{}".format(y_class[k]))
            else:
                plt.bar(j + 1 + bar_bias[k], probability["P({}:{}|income:{})".format(x, x_class[j], y_class[k])],
                        color=color[k], width=bar_width)
    plt.legend()
    plt.savefig("./figure/{}.png".format(x))
    plt.clf()
"""
后验概率最大化，进行预测
"""
test_data["prediction"] = 0
for i in range(test_data.shape[0]):
    p0 = probability["P(income:{})".format(y_class[0])]
    for x in x_total:
        p0 = p0 * probability["P({}:{}|income:{})".format(x, test_data.loc[i, x], y_class[0])]
    p1 = probability["P(income:{})".format(y_class[0])]
    for x in x_total:
        p1 = p1 * probability["P({}:{}|income:{})".format(x, test_data.loc[i, x], y_class[1])]
    if p0 > p1:
        test_data.loc[i, "prediction"] = y_class[0]
    else:
        test_data.loc[i, "prediction"] = y_class[1]
test_data["True_or_False"] = test_data[["income", "prediction"]].apply(lambda x: x["income"] == x["prediction"], axis=1)  # 得出预测是否正确
print("测试集的准确率为：{}".format(np.sum(test_data["True_or_False"]) / test_data.shape[0]))

