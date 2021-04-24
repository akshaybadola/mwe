import numpy as np
from collections import Counter


def merge_matrix(mat1, mat2):
    result = {}
    for rel_type, matrix in mat1.items():
        m1 = Counter(matrix)
        m2 = Counter(mat2[rel_type])
        result[rel_type] = dict(m1 + m2)
    return result


def process_pretrain(data, glove_path):
    with open(glove_path, "r") as f:
        glove_embeddings = f.readlines()
    # remove \n
    print("split")
    glove_embeddings = [x.strip().split() for x in glove_embeddings]
    width = len(glove_embeddings[-1])
    glove_embeddings.append([0]*width)
    glove_embeddings = [x for x in glove_embeddings if len(x) == 603]
    print(len(glove_embeddings))
    word2index = {value[0]: counter for counter, value in enumerate(glove_embeddings)}
    print("filter")
    target_embeddings = np.array([glove_embeddings[word2index.get(
        data.id2word[i], -1)][1:300] for i in range(len(data.id2word))]).astype(np.float)
    context_embeddings = np.array([glove_embeddings[word2index.get(
        data.id2word[i], -1)][301:600] for i in range(len(data.id2word))]).astype(np.float)
    print(target_embeddings, context_embeddings)
    np.savetxt("2018_glove_U", target_embeddings, fmt='%f')
    np.savetxt("2018_glove_V", context_embeddings, fmt='%f')
    return target_embeddings, context_embeddings
