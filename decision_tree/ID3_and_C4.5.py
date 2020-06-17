import numpy as np
import pandas as pd
import argparse


# 书的节点
class TreeNode:
    def __init__(self, feature_index=None, feature_name=None, child_tree=None, node_value=None):
        """
        :param feature_index: 该节点划分依据的特征索引
        :param feature_name: 该节点划分依据的特征名
        :param child_tree: 子树
        :param node_value: 叶节点储存的类别值
        """
        self._feature_index = feature_index
        self._feature_name = feature_name
        self._child_tree = child_tree
        self._node_value = node_value


# 决策树模型
class DecisionTree:
    def __init__(self, feature_name_list, tree_type="ID3", epsilon=0.01):
        """
        :param feature_name_list: 所有变量的特征名
        :param tree_type: 决策树的类型：ID3 or C4.5
        :param epsilon: 当信息增益或信息增益比小于该阈值时直接把对应结点作为叶结点
        """
        self._feature_name_list = feature_name_list  # 所有的特征名列表
        self._tree_type = tree_type
        self._epsilon = epsilon
        self._root = None  # 初始化根节点为空
        self._feature_index_list = list(np.arange(0, len(feature_name_list), 1))  # # 剩下的所有特征索引的列表
        self.selected_fea_name_list = []  # 按划分时选取的顺序保存特征名

    # 模型的训练
    def fit(self, X, y):
        """
        :param X: 训练集变量数组，每行为一个样本，每一列为一个变量
        :param y: 训练集的标签
        """
        self._root = self._build_tree(X, y)

    # 预测
    def predict(self, x):
        """
        :param x: 单个待预测样本的特征数据
        :return: 预测的类别
        """
        return self._predict(x, self._root)

    # 树的生成
    def _build_tree(self, X, y):
        if np.unique(y).shape[0] == 1:  # 如果只有一个类别，将该子树设为叶节点
            return TreeNode(node_value=self._vote_label(y))

        if len(self._feature_index_list) == 0:  # 如果特征列表中只有一个特征，将该子树设为叶节点
            return TreeNode(node_value=self._vote_label(y))

        max_gain = -np.inf  # 初始化最大信息增益或信息增益比为负无穷
        max_fea_ind = 0  # main_gain对应的特征索引值

        # 遍历剩下的特征列表，找出信息增益或增益比最大的特征
        for i in range(len(self._feature_index_list)):
            if self._tree_type == "ID3":
                gain = self._calc_entropy(y)
            else:  # C4.5
                gain = self._calc_condition_entropy(X[:, i], y)
            if gain > max_gain:
                max_fea_ind = i
                max_gain = gain

        # 如果以该特征进行划分信息增益或增益比小于epsilon,则不进行划分
        if max_gain < self._epsilon:
            return TreeNode(node_value=self._vote_label(y))

        feature_name = self._feature_name_list[self._feature_index_list[max_fea_ind]]  # 选取的特征名
        feature_index = self._feature_index_list[max_fea_ind]  # 选取的特征在剩余特征中的索引
        self._feature_index_list.remove(feature_index)  # 删去选取的特征索引
        self.selected_fea_name_list.append(feature_name)

        child_tree = dict()
        # 遍历所选特征每一个可能的值，对每一个值构建子树
        feature_val = np.unique(X[:, max_fea_ind])
        for fea_val in feature_val:
            # 该子树对应的数据集和标签
            child_X = X[X[:, max_fea_ind] == fea_val]
            child_y = y[X[:, max_fea_ind] == fea_val]
            child_X = np.delete(child_X, max_fea_ind, 1)
            # 构建子树
            child_tree[fea_val] = self._build_tree(child_X, child_y)
        return TreeNode(feature_index=feature_index, feature_name=feature_name, child_tree=child_tree)

    # 计算信息熵
    def _calc_entropy(self, y):
        entropy = 0
        N = y.shape[0]
        _, count_num = np.unique(y, return_counts=True)
        for n in count_num:  # 遍历标签值
            p = n / N
            entropy -= p * np.log2(p)
        return entropy

    # 计算条件熵
    def _calc_condition_entropy(self, x, y):
        """
        :param x: 选取的特征变量x的数据值
        :param y: 标签
        :return: 条件熵
        """
        cond_entropy = 0
        N = y.shape[0]
        x_val, count_num = np.unique(x, return_counts=True)
        for v, n in zip(x_val, count_num):  # 遍历x的取值
            sub_y = y[x == v]
            p = n / N
            sub_entropy = self._calc_entropy(sub_y)
            cond_entropy += p * sub_entropy
        return cond_entropy

    # 将y中出现次数最多的类别作为输出
    def _vote_label(self, y):
        label, count_num = np.unique(y, return_counts=True)
        return label[np.argmax(count_num)]

    # 给定输入样本，将其划分到所属叶结点, 最终返回预测的类别
    def _predict(self, x, tree=None):
        if tree is None:
            tree = self._root

        if tree._node_value is not None:  # 如果是叶节点，则返回类别值
            return tree._node_value

        feature_index = tree._feature_index
        for fea_val, child_node in tree._child_tree.items():
            if x[feature_index] == fea_val:
                # 继续去子树中找
                return self._predict(x, child_node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="决策树算法代码命令行参数")
    parser.add_argument("--epsilon", type=float, default=0.01, help="当信息增益或信息增益比小于该阈值时直接把对应结点作为叶结点")
    parser.add_argument("--tree_type", type=str, default="C4.5", help="选用ID3或者C4.5")
    args = parser.parse_args()

    # 数据预处理，各变量解释见"../navie_bayes/navie_bayes.py"，数据处理方式也相同
    data = pd.read_csv("../navie_bayes/adult.csv")
    for i in data.columns:  # 遍历列，删去缺失的数据
        data = data[-data["{}".format(i)].isin(["?"])]
    data = data.sample(frac=1, random_state=10).reset_index(drop=True)  # 随机打乱数据集

    # 设置连续变量的间隔
    age_interval = 5  # age 15-90 5为间隔 左闭右开
    hours_per_week_interval = 5  # hours-per-week 0-99 5为间隔 左闭右开

    # 根据划分的类别将连续变量替换为类别
    data["capital-loss"] = data["capital-loss"].apply(lambda x: "0" if x < 0.01 else "!0")
    data["capital-gain"] = data["capital-gain"].apply(lambda x: "0" if x < 0.01 else "!0")
    data["age"] = ((data["age"] - 15) / age_interval).astype(np.int32)
    data["hours-per-week"] = ((data["age"] - 15) / age_interval).astype(np.int32)

    # 0.9训练集，0.1测试集
    N = int(data.shape[0] * 0.9)
    train_data = data.loc[0:N - 1, :].reset_index(drop=True)
    test_data = data.loc[N:, :].reset_index(drop=True)

    x_total = ['workclass', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender',
               'native-country', "age", "capital-loss", "capital-gain", "hours-per-week"]  # 所有的影响变量
    train_x, train_y = train_data[x_total].values, train_data["income"].values
    test_x, test_y = test_data[x_total].values, test_data["income"].values

    # 模型的训练
    model = DecisionTree(feature_name_list=x_total, tree_type=args.tree_type, epsilon=args.epsilon)
    model.fit(train_x, train_y)
    print("特征变量的选取顺序，即重要性排序：{}".format(model.selected_fea_name_list))

    # 模型的测试
    count = 0  # 正确判断的次数
    test_n = test_x.shape[0]  # 测试样本数
    for i in range(test_n):
        pre_label = model.predict(test_x[i])
        if pre_label == test_y[i]:
            count += 1
    print("{}模型的预测准确率为：{}".format(args.tree_type, count / test_n))