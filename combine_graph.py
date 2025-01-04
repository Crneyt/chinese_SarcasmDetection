# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
import hanlp
import pickle
from cooccurrence import build_cooccurrence_graph
import torch
from tqdm import tqdm

# 初始化HanLP的NLP工具
tokenizer = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_BASE)
dep_parser = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)
word_vectors = hanlp.load(hanlp.pretrained.word2vec.MERGE_SGNS_BIGRAM_CHAR_300_ZH)
EXPECTED_DIMENSION = 300
window_size = 5


def build_dependency_graph_single(sentence):
    tokenized_sentence = tokenizer(sentence)
    # print(f"语法分词结果：{tokenized_sentence}")
    parsed_result = dep_parser(tokenized_sentence)
    # print(f"解析结果：{parsed_result}")  # 检查解析结果的结构
    G = nx.DiGraph()

    for token in parsed_result:
        node_id = token['id'] - 1
        # print(f"Token: {token}, Node ID: {node_id}")  # 打印每个节点的信息
        G.add_node(node_id, word=token['form'])
        if 'head' in token and 'deprel' in token and token['head'] > 0:
            G.add_edge(node_id, token['head'] - 1, relation=token['deprel'])

    return G

def build_dependency_graphs(text):
    tokenized_text = tokenizer([text])
    parsed_result = dep_parser(tokenized_text[0])
    G = nx.DiGraph()

    for token in parsed_result:
        node_id = token['id'] - 1
        G.add_node(node_id, word=token['form'])
        # 修正：确保添加的边方向正确，即从依赖词指向头词。
        if 'head' in token and 'deprel' in token and token['head'] > 0:
            G.add_edge(node_id, token['head'] - 1, relation=token['deprel'])

    return G

def merge_graphs(G_dep, G_cooc):
    G_combined = nx.DiGraph()
    # 合并节点
    for node, data in G_dep.nodes(data=True):
        G_combined.add_node(node, **data)
    for node in G_cooc.nodes():
        if not G_combined.has_node(node):
            G_combined.add_node(node, word=node)
    # 合并边
    for u, v, data in G_dep.edges(data=True):
        G_combined.add_edge(u, v, weight=1, type='dependency', relation=data['relation'])
    for u, v, data in G_cooc.edges(data=True):
        if G_combined.has_edge(u, v):
            G_combined[u][v]['weight'] += data['weight']
        else:
            G_combined.add_edge(u, v, weight=data['weight'], type='cooccurrence')

    return G_combined


def create_feature_matrix(G, word_vectors):
    features = []
    missing_words = []  # 用于记录没有找到词向量的词
    for node, data in G.nodes(data=True):
        word = data.get('word', '')
        # 使用HanLP的word_vectors模型来获取词向量
        vector = word_vectors([word])[0].cpu().numpy()
        # 检查是否成功获取了词向量
        if vector is None or vector.shape[0] != EXPECTED_DIMENSION:
            vector = np.zeros(EXPECTED_DIMENSION)  # 对于未找到的词向量，创建一个零向量
            missing_words.append(word)  # 将未找到的词添加到列表中
        features.append(vector)

    # 打印所有未找到词向量的词
    if missing_words:
        print("Missing word vectors for the following words:", missing_words)

    # 将features列表转换为NumPy数组
    feature_matrix = np.vstack(features) if features else np.empty((0, EXPECTED_DIMENSION))
    return feature_matrix


