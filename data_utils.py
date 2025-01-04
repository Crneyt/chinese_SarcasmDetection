# 原始讽刺数据集类

import torch
import pandas as pd
import pickle
from torch.utils.data import Dataset
from ast import literal_eval


class SarcasmDataset(Dataset):
    def __init__(self, csv_file, combine_file, sent_file):

        try:
            # 尝试使用不同的编码方式读取文件
            self.data_frame = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            try:
                self.data_frame = pd.read_csv(csv_file, encoding='ISO-8859-1', on_bad_lines='skip')
            except UnicodeDecodeError:
                self.data_frame = pd.read_csv(csv_file, encoding='GBK', on_bad_lines='skip')

        with open(combine_file, 'rb') as f:
            matrices = pickle.load(f)
            self.combine_adjacency_matrices = matrices['combine_adjacencies']
            self.feature_matrices = matrices['features']

        with open(sent_file, 'rb') as f:
            matrices = pickle.load(f)
            self.sent_adjacency_matrices = matrices['adjacency_matrices']
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            label = self.data_frame.iloc[idx,2]
            combine_adjacency_matrices = torch.tensor(self.combine_adjacency_matrices[idx], dtype=torch.float)
            sent_adjacency_matrix = torch.tensor(self.sent_adjacency_matrices[idx], dtype=torch.float)
            feature_matrix = torch.tensor(self.feature_matrices[idx], dtype=torch.float)

            # 打印每个矩阵的维度
            # print(f"combine_adjacency_matrix shape: {combine_adjacency_matrices.shape}")
            # print(f"sent_adjacency_matrix shape: {sent_adjacency_matrix.shape}")
            # print(f"feature_matrix shape: {feature_matrix.shape}")

            sample = {
                'label': label,
                'combine_adjacency_matrix': combine_adjacency_matrices,
                'sent_adjacency_matrix': sent_adjacency_matrix,
                'feature_matrix': feature_matrix
            }
            return sample
        except IndexError as e:
            print(f"IndexError caught! Attempted to access index {idx} which is out of range.")
            raise e  # Re-raise the exception to handle it according to the script's design