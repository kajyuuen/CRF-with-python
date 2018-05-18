import numpy as np

def flatten(x):
    z = []
    for i in x:
        z.extend(flatten(i) if isinstance(i, list) else [i])
    return z

class Features:
    def __init__(self, labels):
        self.functions = []
        self.label_functions = []
        self.labels = flatten(labels)

    def create_feature(self, word_set):
        # 素性関数ベクトルの作成
        for label in self.labels:
            for word in word_set:
                # x_m, y_mからなる素性関数ベクトル
                self.add_feature(lambda x, y, w=word, l=label: 1 if y==l and x==w else 0)
            # y_{m-1}, y_mからなる素性関数ベクトル
            for before_label in ["<BOS>"] + self.labels:
                self.add_label_feature(lambda y_b, y_c, bl=before_label, l=label: 1 if y_b==bl and y_c==l else 0)
            # 一次マルコフ性を完全にカバーする為に必要
            # TODO: x_m, y_{m-1}, y_mからなる素性関数ベクトル

            # 素性関数ベクトルの次元数
            self.dimension = len(self.functions) + len(self.label_functions)

    def add_feature(self, f):
        self.functions.append(f)

    def add_label_feature(self, f):
        self.label_functions.append(f)

class FeatureVector:
    def __init__(self, features, x_list, y_list):
        func_list = features.functions
        label_func_list = features.label_functions
        # ラベルの種類
        L = len(features.labels)

        # ある時点mの素性ベクトルはmat[m]
        self.mat = np.empty((0, features.dimension), int)
        for x, y_b, y in zip(x_list, ["<BOS>"]+y_list, y_list):
            # \sum_{m=1}^{M} \pi(y_{m-1}, y_m)
            pp_vec = [func(y_b, y) for func in label_func_list]
            # \sum_{m=1}^{M} \pi(x_m, y_m)
            pw_vec = [func(x, y) for func in func_list]
            # ある時刻における素性関数ベクトル
            current_vec = np.concatenate([pp_vec, pw_vec], axis=0)
            self.mat = np.append(self.mat, current_vec[np.newaxis, :], axis=0)
