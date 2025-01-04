# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
import itertools
import pickle
import stanza
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

window_size = 3

# 初始化并加载英文模型，包括分词、词性标注和依存句法分析器
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', use_gpu=False,
                      model_dir='/home/pptan/anaconda3/envs/Sar2env/lib/python3.8/site-packages/stanza/stanza_resources')

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('/home/pptan/Codex/bert-base-uncase')
bert_model = BertModel.from_pretrained('/home/pptan/Codex/bert-base-uncase')
bert_model.eval()

EXPECTED_DIMENSION = 768  # BERT的嵌入维度


def build_dependency_graphs(text):
    doc = nlp(text)
    G = nx.DiGraph()
    offset = 0

    # 从 stanza 的 doc 中提取词列表
    words = [word.text for sent in doc.sentences for word in sent.words]

    # 获取 BERT 嵌入
    bert_embeddings = get_bert_embeddings(words)

    for sent in doc.sentences:
        for token in sent.words:
            node_id = int(token.id) - 1 + offset
            # 确保 token.id -1 不超出 bert_embeddings 的范围
            if (token.id - 1) < len(bert_embeddings):
                embedding = bert_embeddings[token.id - 1]
            else:
                embedding = np.zeros(EXPECTED_DIMENSION)
            G.add_node(node_id, word=token.text, pos=token.pos, lemma=token.lemma,
                       embedding=embedding)
            if token.head > 0:
                G.add_edge(int(token.head) - 1 + offset, node_id, relation=token.deprel)
        offset += len(sent.words)
    return G


def get_bert_embeddings(words):
    """
    获取每个词的 BERT 嵌入。由于 BERT 可能会将一个词拆分成多个子词，这里对每个词的子词嵌入取平均值作为该词的嵌入。
    """
    # 将词列表转换为 BERT 输入格式
    sentence = ' '.join(words)
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    last_hidden_state = outputs.last_hidden_state.squeeze(0)  # [seq_length, hidden_size]

    # 获取所有子词
    tokenized = tokenizer.tokenize(sentence)
    word_embeddings = []
    word_idx = 0
    current_word = words[word_idx] if words else ""
    word_tokens = tokenizer.tokenize(current_word)
    word_token_len = len(word_tokens)
    for i in range(0, len(tokenized), word_token_len):
        if word_idx >= len(words):
            break
        # 取当前词的子词嵌入并平均
        start = i
        end = i + word_token_len
        if end > len(last_hidden_state):
            end = len(last_hidden_state)
        if start >= end:
            embedding = np.zeros(EXPECTED_DIMENSION)
        else:
            embedding = last_hidden_state[start:end].mean(dim=0).numpy()
        word_embeddings.append(embedding)
        word_idx += 1
    # 如果有多余的嵌入，忽略它们
    embeddings_array = np.array(word_embeddings)
    # print(f"word_embeddings shape: {embeddings_array.shape}")  # 应该是 [num_words, 768]
    return embeddings_array


def preprocess_text(text):
    doc = nlp(text.lower())
    words = [word.lemma for sent in doc.sentences for word in sent.words if word.pos not in {'PUNCT', 'SYM', 'X'}]
    return words


def build_cooccurrence_graph(sentence):
    words = preprocess_text(sentence)
    G = nx.Graph()
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        for w1, w2 in itertools.combinations(window, 2):
            if G.has_edge(w1, w2):
                G[w1][w2]['weight'] += 1
            else:
                G.add_edge(w1, w2, weight=1)
    return G


def merge_graphs(G_dep, G_cooc):
    G_combined = nx.DiGraph()
    for node, data in G_dep.nodes(data=True):
        G_combined.add_node(node, **data)
    for u, v, data in G_dep.edges(data=True):
        G_combined.add_edge(u, v, type='dependency', **data)
    for u, v, data in G_cooc.edges(data=True):
        if u not in G_combined or v not in G_combined:
            continue
        if G_combined.has_edge(u, v):
            if 'cooccurrence_weight' in G_combined[u][v]:
                G_combined[u][v]['cooccurrence_weight'] += data.get('weight', 1)
            else:
                G_combined[u][v]['cooccurrence_weight'] = data.get('weight', 1)
        else:
            G_combined.add_edge(u, v, type='cooccurrence', weight=data.get('weight', 1))
    return G_combined


def create_feature_matrix(G):
    features = []
    for _, node_data in G.nodes(data=True):
        embedding = node_data.get('embedding', np.zeros(EXPECTED_DIMENSION))
        features.append(embedding)
    feature_matrix = np.stack(features) if features else np.empty((0, EXPECTED_DIMENSION))
    # print(f"Feature matrix shape: {feature_matrix.shape}")
    return feature_matrix


def main():
    data_paths = [
        "/home/pptan/Codex/ADGCN-main_2/ADGCN-main/datasets/IAC-V1/train.csv",
        "/home/pptan/Codex/ADGCN-main_2/ADGCN-main/datasets/IAC-V1/test.csv"
    ]
    encodings = ['utf-8', 'utf-8']
    total_files_processed = 0

    for path, encoding in zip(data_paths, encodings):
        df = pd.read_csv(path, encoding=encoding)
        texts = df['text'].tolist()
        all_features = []
        all_adj_matrices = []
        for text in tqdm(texts, desc=f"Processing {path}"):
            G_cooc = build_cooccurrence_graph(text)
            G_dep = build_dependency_graphs(text)
            G_combined = merge_graphs(G_dep, G_cooc)
            feature_matrix = create_feature_matrix(G_combined)
            adj_matrix = nx.to_numpy_array(G_combined)
            all_features.append(feature_matrix)
            all_adj_matrices.append(adj_matrix)

        data_for_gat = {
            'features': all_features,
            'combine_adjacencies': all_adj_matrices,
        }
        save_path = path.replace('.csv', '_combined_bert.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data_for_gat, f)
        total_files_processed += 1

    if total_files_processed == len(data_paths):
        print("全部成功！")
        print(f"处理并保存了{total_files_processed}个文件。")
    else:
        print(f"成功处理了{total_files_processed}个文件，但有{len(data_paths) - total_files_processed}个文件未能处理。")


if __name__ == "__main__":
    main()
