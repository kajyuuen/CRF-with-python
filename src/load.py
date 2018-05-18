# Word | POS | Syntactic chunk tag | The named entity tag
def load_data(raw_text):
    words, word_data = [], []
    poses, pos_data = [], []
    chunk_tags, chunk_tag_data = [], []
    entity_tags, entity_tag_data = [], []
    for i, line in enumerate(raw_text):
        data = line.rstrip().split()
        if len(data) > 0:
            word = data[0]
            if word == "-DOCSTART-": continue
            pos = data[1]
            chunk_tag = data[2]
            # entity_tag = data[3] // TODO: 必要かどうか確認する
            entity_tag = ""
            words.append(word)
            poses.append(pos)
            chunk_tags.append(chunk_tag)
            entity_tags.append(entity_tag)
        else:
            if len(words) == 0: continue
            word_data.append(words)
            pos_data.append(poses)
            chunk_tag_data.append(chunk_tags)
            entity_tag_data.append(entity_tags)
            words, poses, chunk_tags, entity_tags = [], [], [], []
    word_data.append(words)
    pos_data.append(poses)
    chunk_tag_data.append(chunk_tags)
    entity_tag_data.append(entity_tags)

    return word_data, pos_data, chunk_tag_data, entity_tag_data
