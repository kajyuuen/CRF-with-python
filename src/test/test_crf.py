import sys
import unittest
import copy
import numpy as np
sys.path.append('..')
from features import Features
from crf import CRF

class CRFTest(unittest.TestCase):
    def setUp(self):
        # 訓練データ
        self.train_word_data = [["Peter", "Daniel", "Blackburn"], ["1966", "World", "Cup"]]
        self.train_pos_data = [["NNP", "NNP", "NNP"], ["CD", "NNP", "NNP"]]
        word_set = set([word for words in self.train_word_data for word in words])
        self.pos_set = set([pos for poses in self.train_pos_data for pos in poses])
        self.features = Features(list(self.pos_set))
        self.features.create_feature(word_set)
        # テストデータ
        self.test_word_data =  [["Peter", "Daniel", "Blackburn"], ["1980", "World"]]


    def test_predict(self):
        """
        系列x_listに対して，予測させた系列ラベルy_listの長さが一致している
        """
        predicts = []
        crf = CRF(self.features, self.pos_set)
        for x_list in self.test_word_data:
            predict = crf.predict(x_list)
            self.assertEqual(len(predict), len(x_list))

    def test_forward_backward_algorithm(self):
        """
        前向きアルゴリズムと後ろ向きアルゴリズムで求めた状態和が一致する
        """
        crf = CRF(self.features, self.pos_set)
        for x_list in self.test_word_data:
            crf.predict(x_list)
            alpha_z = round(crf._forward_algorithm(x_list)[-1]["<EOS>"], 3)
            beta_z = round(crf._backward_algorithm(x_list)[0]["<BOS>"], 3)
            self.assertEqual(alpha_z, beta_z)

    def test_marginal_probability(self):
        """
        各時点m-1からmへのラベル遷移の確率の合計が1になる
        """
        crf = CRF(self.features, self.pos_set)
        x_list = self.test_word_data[0]
        crf.predict(x_list)
        crf._forward_algorithm(x_list)
        crf._backward_algorithm(x_list)
        for m in range(1, len(x_list)):
            p_sum = 0
            for i_label in self.pos_set:
                for j_label in self.pos_set:
                    p_sum += crf._marginal_probability(x_list, i_label, j_label, m)
            self.assertEqual(1, round(p_sum))

    def test_fit(self):
        """
        学習前と学習後で重みが異なる
        """
        crf = CRF(self.features, self.pos_set)
        before_w = copy.deepcopy(crf.w_lambda)
        crf.fit(self.train_word_data, self.train_pos_data)
        current_w = copy.deepcopy(crf.w_lambda)
        self.assertNotEqual(np.sum(before_w), np.sum(current_w))


if __name__ == '__main__':
    unittest.main()
