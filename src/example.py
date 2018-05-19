from crf import CRF
from features import Features, FeatureVector
import sys
import numpy as np
from collections import defaultdict
from load import load_data


if __name__ == "__main__":
    f = open(sys.argv[1])
    word_data, pos_data, _, _ = load_data(f)
    f.close()

    word_set = set([word for words in word_data for word in words])
    pos_set = set([pos for poses in pos_data for pos in poses])
    features = Features(list(pos_set))

    # 素性の作成
    features.create_feature(word_set)
    feature_vector = [FeatureVector(features, x_list, features.labels) for x_list in zip(word_data)]

    # 学習
    crf = CRF(features, pos_set)
    crf.fit(word_data, pos_data)

    # 性能評価
    f = open(sys.argv[2])
    test_word_data, test_pos_data, _, _ = load_data(f)
    f.close()

    pos_count = 0
    pos_ans_count = 0
    label_ans_count = 0
    for x_list, y_list in zip(test_word_data, test_pos_data):
        predict_y_list = crf.predict(x_list)
        if predict_y_list == y_list:
            label_ans_count += 1
        for pred_y, ans_y in zip(predict_y_list, y_list):
            pos_count += 1
            if(pred_y == ans_y):
                pos_ans_count += 1

    print("Accuracy in sentence:{}".format(label_ans_count/len(test_pos_data)))
    print("Accuracy in label:{}".format(pos_ans_count/pos_count))
