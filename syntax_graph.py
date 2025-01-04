# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
import itertools
import hanlp
import pickle
import stanza
import torch
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors

window_size = 3

# 初始化并加载英文模型，包括分词、词性标注和依存句法分析器
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', model_dir='/home/pptan/anaconda3/envs/Sar2env/lib/python3.8/site-packages/stanza/stanza_resources')
# 加载GloVe词向量
glove_path = "/home/pptan/Codex/dual-channel-for-sarcasm-main/data/glove.42B.300d.txt"  # 修改为你的GloVe文件路径
word_vectors = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
EXPECTED_DIMENSION = 300

def build_dependency_graphs(text):
    doc = nlp(text)
    G = nx.DiGraph()
    offset = 0  # 初始化偏移量

    for sent in doc.sentences:
        for token in sent.words:
            node_id = int(token.id) - 1 + offset  # 考虑前一个句子的偏移
            G.add_node(node_id, word=token.text, pos=token.pos, lemma=token.lemma)
            if token.head > 0:
                # 增加偏移量以正确连接头部和当前节点
                G.add_edge(int(token.head) - 1 + offset, node_id, relation=token.deprel)
        offset += len(sent.words)  # 更新偏移量

    return G


def preprocess_text(text):
    doc = nlp(text.lower())  # 使用stanza处理并转换为小写
    words = [word.lemma for sent in doc.sentences for word in sent.words if word.pos not in {'PUNCT', 'SYM', 'X'}]
    return words

def build_cooccurrence_graph(sentence):
    words = preprocess_text(sentence)
    G = nx.Graph()

    # Use a sliding window to add edges between co-occurring words
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        # Consider all pairs of words in the window
        for w1, w2 in itertools.combinations(window, 2):
            if G.has_edge(w1, w2):
                G[w1][w2]['weight'] += 1
            else:
                G.add_edge(w1, w2, weight=1)
    return G

def merge_graphs(G_dep, G_cooc):
    G_combined = nx.DiGraph()

    # 添加G_dep中的节点和边
    for node, data in G_dep.nodes(data=True):
        G_combined.add_node(node, **data)
    for u, v, data in G_dep.edges(data=True):
        G_combined.add_edge(u, v, type='dependency', **data)

    # 检查G_cooc的节点是否在G_combined中，然后添加边
    for u, v, data in G_cooc.edges(data=True):
        if u not in G_combined or v not in G_combined:
            # 如果节点不存在，打印出哪些节点是新的
            if u not in G_combined:
                print(f"Missing node from cooccurrence graph: {u}")
            if v not in G_combined:
                print(f"Missing node from cooccurrence graph: {v}")
            continue  # 可以选择跳过这些边或添加缺失的节点

        if G_combined.has_edge(u, v):
            # 如果边已存在，累加共现权重
            if 'cooccurrence_weight' in G_combined[u][v]:
                G_combined[u][v]['cooccurrence_weight'] += data.get('weight', 1)
            else:
                G_combined[u][v]['cooccurrence_weight'] = data.get('weight', 1)
        else:
            # 添加新的共现边
            G_combined.add_edge(u, v, type='cooccurrence', weight=data.get('weight', 1))

    return G_combined


def create_feature_matrix(G):
    features = []
    for _, node_data in G.nodes(data=True):
        word = node_data.get('word', '')
        # 获取词向量
        if word in word_vectors:
            vector = word_vectors[word]
        else:
            vector = np.zeros(EXPECTED_DIMENSION)
        features.append(vector)

    feature_matrix = np.stack(features) if features else np.empty((0, EXPECTED_DIMENSION))
    return feature_matrix


def main():
    data_paths = [
        "/home/pptan/Codex/ADGCN-main_2/ADGCN-main/datasets/riloff/train.csv",
        "/home/pptan/Codex/ADGCN-main_2/ADGCN-main/datasets/riloff/test.csv"
    ]
    encodings = ['utf-8','utf-8']
    total_files_processed = 0  # 处理文件的计数器

    for path, encoding in zip(data_paths, encodings):
        df = pd.read_csv(path, encoding=encoding)
        texts = df['text'].tolist()

        all_features = []
        all_adj_matrices = []

        for text in tqdm(texts, desc=f"Processing {path}"):
            G_cooc = build_cooccurrence_graph(text)
            # print(f"共现图节点数: {G_cooc.number_of_nodes()}")
            G_dep = build_dependency_graphs(text)
            # print(f"依赖关系图节点数: {G_dep.number_of_nodes()}")
            G_combined = merge_graphs(G_dep, G_cooc)
            # print(f"合并后的图节点数: {G_combined.number_of_nodes()}")
            feature_matrix = create_feature_matrix(G_combined)
            adj_matrix = nx.to_numpy_array(G_combined)

            all_features.append(feature_matrix)
            all_adj_matrices.append(adj_matrix)

            # 准备保存数据的字典
            data_for_gat = {
                'features': all_features,
                'combine_adjacencies': all_adj_matrices,
            }

            # 定义保存文件的路径
            save_path = path.replace('.csv', '_combined.pkl')

            # 使用pickle保存处理结果
            with open(save_path, 'wb') as f:
                pickle.dump(data_for_gat, f)

        total_files_processed += 1  # 每处理完一个文件，计数器加一

    # 检查是否所有文件都已处理
    if total_files_processed == len(data_paths):
        print("全部成功！")
        print(f"处理并保存了{total_files_processed}个文件。")
    else:
        print(f"成功处理了{total_files_processed}个文件，但有{len(data_paths) - total_files_processed}个文件未能处理。")

if __name__ == "__main__":
    main()
