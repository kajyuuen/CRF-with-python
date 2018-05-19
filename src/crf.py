import numpy as np
from math import exp
from features import flatten

class CRF:
    def __init__(self, features, labels, learning_rate = 0.1, random_seed = None):
        self.features = features
        # 重みベクトル
        if random_seed is not None:
            np.random.seed(random_seed)
        self.w_lambda = np.random.rand(features.dimension) * 0.0001
        self.labels = labels
        self.learning_rate = learning_rate

    def predict(self, x_list):
        """
        ビタビアルゴリズムを利用したラベル列の推定
        """
        edge = []
        node = []
        # TODO: edge, nodeをlabelで表現するのではなく，idで表現して行列積を可能にする
        # TODO: m>0にまとめる
        # m=1のとき
        current_edge = {}
        current_node = {}
        max_score = -10e+9
        for predict_y in self.labels:
            predict_feature_vec = [func(x_list[0], predict_y) for func in self.features.functions] + [func("<BOS>", predict_y) for func in self.features.label_functions]
            score = np.dot(self.w_lambda, predict_feature_vec)
            current_edge[("<BOS>", predict_y)] = score
            current_node[predict_y] = score
        edge.append(current_edge)
        node.append(current_node)

        # m>1のとき
        for ind in range(1, len(x_list)):
            current_edge = {}
            current_node = {}
            for predict_y in self.labels:
                max_score = -10e+9
                for before_y in self.labels:
                    predict_feature_vec = [func(x_list[ind], predict_y) for func in self.features.functions] \
                                          + [func(before_y, predict_y) for func in self.features.label_functions]
                    score = np.dot(self.w_lambda, predict_feature_vec) + node[ind-1][before_y]
                    current_edge[(before_y, predict_y)] = score
                    if(score > max_score):
                        max_score = score
                current_node[predict_y] = max_score
            edge.append(current_edge)
            node.append(current_node)

        self.edge = edge
        self.node = node

        # ラベル列の推定
        # m=M
        current_score = sorted(node[-1].items(), key=lambda x: x[1])[0][1]
        predict_labels = [sorted(node[-1].items(), key=lambda x: x[1])[0][0]]

        # m<M
        for ind, labels in enumerate(node[-2::-1]):
            max_score = -10e+9
            for label, score in labels.items():
                score = edge[-1-ind][(label, predict_labels[ind])]
                if(score > max_score):
                    max_score = score
                    predict_label = label
            predict_labels.append(predict_label)
        return predict_labels

    def fit(self, x_lists, y_lists):
        for x_list, y_list in zip(x_lists, y_lists):
            self.predict(x_list)
            self._forward_algorithm(x_list)
            self._backward_algorithm(x_list)
            self.w_lambda += self.learning_rate * self._gradient(x_list, y_list)

    def _backward_algorithm(self, x_list):
        beta_node = [{label : 1 for label in self.labels}]

        for ind in range(len(x_list)):
            current_beta_node = {}
            if(ind < len(x_list)-1):
                labels = self.labels
            else:
                labels = ["<BOS>"]
            for label in labels:
                current_beta = 0
                for after_label in self.labels:
                    current_beta += exp(self.edge[-1-ind][(label, after_label)]) * beta_node[ind][after_label]
                current_beta_node[label] = current_beta
            beta_node.append(current_beta_node)

        beta_node.reverse()
        self.state_sum = beta_node[0]["<BOS>"]
        self.beta_node = beta_node
        return beta_node

    def _forward_algorithm(self, x_list):
        alpha_node = [{label: exp(self.node[0][label]) for label in self.labels}]

        for ind in range(1, len(x_list)):
            current_alpha_node = {}
            for label in self.labels:
                current_alpha = 0
                for before_label in self.labels:
                    current_alpha += exp(self.edge[ind][(before_label, label)]) * alpha_node[ind-1][before_label]
                current_alpha_node[label] = current_alpha
            alpha_node.append(current_alpha_node)

        # 分配関数
        alpha_node.append({"<EOS>": sum([v for v in alpha_node[-1].values()])})
        self.state_sum = alpha_node[-1]["<EOS>"]
        self.alpha_node = alpha_node
        return alpha_node

    def _gradient(self, x_list, y_list):
        gradient_value = 0
        y_list = ["<BOS>"] + y_list
        for m in range(1, len(x_list)):
            x, before_y, y = x_list[m], y_list[m-1], y_list[m]
            feature_vec = [func(x, y) for func in self.features.functions] + [func(before_y, y) for func in self.features.label_functions]
            term = 0
            for i_label in self.labels:
                for j_label in self.labels:
                    predict_feature_vec = np.array([func(x, i_label) for func in self.features.functions] + [func(i_label, j_label) for func in self.features.label_functions])
                    term += self._marginal_probability(x_list, i_label, j_label, m) * predict_feature_vec
            gradient_value += feature_vec - term
        return gradient_value


    def _marginal_probability(self, x_list, i_label, j_label, m):
        """
        時刻m-1のときのラベルが: i_label
        時刻mのときのラベルが: j_label
        周辺確率p(i_label, j_label|x_list, m)を求める
        """
        return (self.alpha_node[m-1][i_label] * exp(self.edge[m][(i_label, j_label)]) * self.beta_node[m+1][j_label]) / self.state_sum

    # TODO: ラベル列が予測される確率を計算するメソッド
