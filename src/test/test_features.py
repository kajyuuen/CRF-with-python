import sys
import unittest
import numpy as np
sys.path.append('..')
from features import Features, FeatureVector

class FeaturesTest(unittest.TestCase):
    def setUp(self):
        self.word_data = [["Peter", "Daniel", "Blackburn"], ["1966", "World", "Cup"]]
        self.pos_data = [["NNP", "NNP", "NNP"], ["CD", "NNP", "NNP"]]
        self.word_set = set([word for words in self.word_data for word in words])
        pos_set = set([pos for poses in self.pos_data for pos in poses])
        self.features = Features(list(pos_set))

    def test_create_feature(self):
        features = self.features
        features.create_feature(self.word_set)
        # x_m, y_mからなる素性
        self.assertEqual(len(features.functions), 6*2)
        # y_{m-1}, y_mからなる素性(BOSがあることに注意)
        self.assertEqual(len(features.label_functions), (2+1)*2)
        # x, y_{m-1}, y_mからなる素性
        self.assertEqual(len(features.word_label_functions), 6*(2+1)*2)


    def test_create_pos_word_feature_vec(self):
        # HACK: test_create_featureで既にやっている処理．DRYにしたい
        features = self.features
        features.create_feature(self.word_set)

        feature_vector = [FeatureVector(features, x_list, y_list) for x_list, y_list in zip(self.word_data, self.pos_data)]

        # 各素性ベクトル
        self.assertEqual(np.sum(feature_vector[0].mat[0]), 3)
        self.assertEqual(np.sum(feature_vector[1].mat[1]), 3)

if __name__ == '__main__':
    unittest.main()
