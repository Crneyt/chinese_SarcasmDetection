# -*- coding: utf-8 -*-
import hanlp
import pandas as pd
import pickle
import numpy as np
import itertools
import networkx as nx
import torch
import stanza
from tqdm import tqdm

tokenizer = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_BASE)

BATCH_SIZE = 16
window_size = 5


def preprocess_text(text):
    # 直接使用 tokenizer 分词，返回分词后的列表
    words = tokenizer(text)
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


def process_texts_in_batches(texts, batch_size):
    cooccu_matrices = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts"):
        batch_texts = [text for text in texts[i:i + batch_size] if isinstance(text, str)]

        for text in batch_texts:
            coocc_graph = build_cooccurrence_graph(text)
            G = nx.Graph(coocc_graph)
            adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
            cooccu_matrices.append(adj_matrix)
        torch.cuda.empty_cache()  # 清空未使用的缓存，仅当使用PyTorch GPU时需要
    return cooccu_matrices


def main():
    data_paths = [
        "/home/pptan/Codex/dual-channel-for-sarcasm-main/data/IAC-V1/test.csv",
        "/home/pptan/Codex/dual-channel-for-sarcasm-main/data/IAC-V1/train.csv",
        "/home/pptan/Codex/dual-channel-for-sarcasm-main/data/IAC-V1/valid.csv"
    ]
    encodings = ['utf-8', 'GBK', 'utf-8']

    for path, encoding in zip(data_paths, encodings):
        try:
            df = pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            print(f"Encoding error encountered while reading {path}")
            continue

        texts = df['text'].tolist()
        cooccu_matrices = process_texts_in_batches(texts, BATCH_SIZE)

        data_for_gnn = {
            'cooccu_matrices': cooccu_matrices
        }

        save_path = path.replace('.csv', '_cooccurrence_graph.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data_for_gnn, f)

        print(f"Processed and saved {len(texts)} texts from {path}.")


if __name__ == "__main__":
    main()
