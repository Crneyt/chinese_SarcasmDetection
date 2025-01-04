# -*- coding: utf-8 -*-
import pickle
import numpy as np
import networkx as nx
import torch
import pandas as pd
import stanza
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel

analyzer = SentimentIntensityAnalyzer()

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('/home/pptan/Codex/bert-base-uncase')
bert_model = BertModel.from_pretrained('/home/pptan/Codex/bert-base-uncase')
print(f"BERT hidden size: {bert_model.config.hidden_size}")
bert_model.eval()  # 设置为评估模式

# 初始化Stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', use_gpu=False, model_dir='/home/pptan/anaconda3/envs/Sar2env/lib/python3.8/site-packages/stanza/stanza_resources')

BATCH_SIZE = 32
threshold = 0.5
similarity_threshold = 0.1

def cosine_similarity(vec1, vec2):
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 > 0 and norm_vec2 > 0:
        return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
    else:
        return 0.0

def get_bert_embeddings(words):
    inputs = tokenizer(words, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # 使用[CLS]向量作为句子表示，或取平均
    cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embeddings

def build_sentiment_graph(sentence):
    doc = tokenizer(sentence)
    graph = nx.DiGraph()
    global_word_index = 0
    words = sentence.split()

    # 获取BERT嵌入
    bert_embeddings = get_bert_embeddings(words)

    for i, word in enumerate(words):
        node_id = f"{word}{global_word_index}"
        sentiment_score = analyzer.polarity_scores(word)['compound']
        graph.add_node(node_id, sentiment_score=sentiment_score, original_token=word, embedding=bert_embeddings[i])
        global_word_index += 1

    nodes = list(graph.nodes())
    for i, node_a in enumerate(nodes):
        vec_a = graph.nodes[node_a]['embedding']
        for j, node_b in enumerate(nodes):
            if i != j:
                vec_b = graph.nodes[node_b]['embedding']
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
            G = build_sentiment_graph(text)
            adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
            sent_adjacency_matrices.append(adj_matrix)
    return sent_adjacency_matrices

def process_and_save_each_dataset(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='gb18030')

    texts = df['text'].tolist()
    sent_adjacency_matrices = process_texts_in_batches(texts, BATCH_SIZE)
    data_for_gat = {
        'adjacency_matrices': sent_adjacency_matrices
    }
    save_path = file_path.replace('.csv', '_sent_bert.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data_for_gat, f)

    print(f"Processed and saved {len(texts)} texts from {file_path}.")

data_paths = [
    "/home/pptan/Codex/ADGCN-main_2/ADGCN-main/datasets/IAC-V1/train.csv",
    "/home/pptan/Codex/ADGCN-main_2/ADGCN-main/datasets/IAC-V1/test.csv"
]

for path in data_paths:
    process_and_save_each_dataset(path)
