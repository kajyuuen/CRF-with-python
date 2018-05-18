import crf
import sys
import numpy as np
from collections import defaultdict
from load import load_data

if __name__ == "__main__":
    f = open(sys.argv[1])
    word_data, pos_data, _, _ = load_data(f)
    f.close()

    #features = Features(chunk_tag_data)
    word_set = set([word for words in word_data for word in words])
    pos_set = set([pos for poses in pos_data for pos in poses])
    features = Features(list(pos_set))

    # 素性の作成
    features.create_feature(word_set)

    feature_vector = [FeatureVector(features, x_list, y_list) for x_list, y_list in zip(word_data, features.labels)]



