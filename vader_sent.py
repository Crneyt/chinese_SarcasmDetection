# -*- coding: utf-8 -*-
import pickle
import numpy as np
import networkx as nx
import hanlp
import torch
import pandas as pd
import stanza
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.models.keyedvectors import KeyedVectors

analyzer = SentimentIntensityAnalyzer()
# 加载分词模型
tokenizer = stanza.Pipeline(lang='en', processors='tokenize', use_gpu=False, model_dir='/home/pptan/anaconda3/envs/Sar2env/lib/python3.8/site-packages/stanza/stanza_resources')

# 设置批处理大小
BATCH_SIZE = 32
threshold=0.5
similarity_threshold=0.1

glove_path = '/home/pptan/Codex/dual-channel-for-sarcasm-main/data/glove.42B.300d.txt'  # 示例路径，根据实际情况修改
word_vectors = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)


def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似度"""
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # 确保两个向量的范数都不为零
    if norm_vec1 > 0 and norm_vec2 > 0:
        dot_product = np.dot(vec1, vec2)
        return dot_product / (norm_vec1 * norm_vec2)
    else:
        # 如果任一向量的范数为零，相似度返回0（或其他适当的默认值）
        return 0.0


def get_word_vector(word):
    """从GloVe中获取词向量，如果不存在则返回零向量"""
    try:
        return word_vectors[word]
    except KeyError:
        return np.zeros(300)  # 假设使用的是300维的GloVe向量

def build_sentiment_graph(sentence):
    doc = tokenizer(sentence)
    graph = nx.DiGraph()

    # 初始化全局单词计数器
    global_word_index = 0

    # 遍历每个句子和其中的单词
    for sent in doc.sentences:
        for word in sent.words:
            node_id = f"{word.text}{global_word_index}"
            sentiment_score = analyzer.polarity_scores(word.text)['compound']
            graph.add_node(node_id, sentiment_score=sentiment_score, original_token=word.text)
            global_word_index += 1  # 更新全局索引

    # 添加边
    nodes = list(graph.nodes())
    for i, node_a in enumerate(nodes):
        vec_a = get_word_vector(graph.nodes[node_a]['original_token'].lower())
        for j, node_b in enumerate(nodes):
            if i != j:
                vec_b = get_word_vector(graph.nodes[node_b]['original_token'].lower())
                semantic_similarity = cosine_similarity(vec_a, vec_b)
                score_a = graph.nodes[node_a]['sentiment_score']
                score_b = graph.nodes[node_b]['sentiment_score']
                score_diff = abs(score_a - score_b)
                polarity_weight = 1 if score_a * score_b > 0 else -1
                weight = (score_diff * polarity_weight) + semantic_similarity
                graph.add_edge(node_a, node_b, weight=weight)

    return graph


def convert_graphs_to_matrices(graphs):
    adjacency_matrices = []

    for G in graphs:
        adj_matrix = nx.to_numpy_array(G)
        adjacency_matrices.append(adj_matrix)

    return adjacency_matrices


def process_texts_in_batches(texts, batch_size):
    sent_adjacency_matrices = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts"):
        batch_texts = texts[i:i + batch_size]

        for text in batch_texts:
            G = build_sentiment_graph(text)  # 直接获取构建好的图
            # 确保使用图中存在的节点列表，以保持矩阵的顺序和完整性
            adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
            sent_adjacency_matrices.append(adj_matrix)

    return sent_adjacency_matrices

def process_and_save_each_dataset(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 尝试使用不同的编码读取
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='gb18030')

    texts = df['text'].tolist()

    # 使用批处理方式处理文本，并显示进度
    sent_adjacency_matrices = process_texts_in_batches(texts, BATCH_SIZE)

    # 准备保存数据的字典
    data_for_gat = {
        'adjacency_matrices': sent_adjacency_matrices
    }

    # 定义保存文件的路径
    save_path = file_path.replace('.csv', '_sent.pkl')

    # 使用pickle保存处理结果
    with open(save_path, 'wb') as f:
        pickle.dump(data_for_gat, f)

    print(f"Processed and saved {len(texts)} texts from {file_path}.")

data_paths = [
        "/home/pptan/Codex/ADGCN-main_2/ADGCN-main/datasets/IAC-V2/train.csv",
        "/home/pptan/Codex/ADGCN-main_2/ADGCN-main/datasets/IAC-V2/test.csv"
]

# 对每个数据集执行处理和保存
for path in data_paths:
    process_and_save_each_dataset(path)



