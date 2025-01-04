import torch
import pandas as pd
import pickle
from torch.utils.data import Dataset
from ast import literal_eval


class SarcasmDataset(Dataset):
    def __init__(self, csv_file, combine_file,sent_file):

        self.data_frame = pd.read_csv(csv_file, on_bad_lines='skip',encoding='utf-8')

        with open(combine_file, 'rb') as f:
            matrices = pickle.load(f)
            self.syntax_adjacency_matrices = matrices['combine_adjacencies']
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
            label = self.data_frame.iloc[idx, 2]
            # label = self.data_frame.iloc[idx, 1]  # 修改为读取第二列的标签
            syntax_adjacency_matrices = torch.tensor(self.syntax_adjacency_matrices[idx], dtype=torch.float)
            sent_adjacency_matrix = torch.tensor(self.sent_adjacency_matrices[idx], dtype=torch.float)
            feature_matrix = torch.tensor(self.feature_matrices[idx], dtype=torch.float)

            sample = {
                'label': label,
                'combine_adjacency_matrix': syntax_adjacency_matrices,
                'sent_adjacency_matrix': sent_adjacency_matrix,
                'feature_matrix': feature_matrix
            }
            return sample
        except IndexError as e:
            print(f"IndexError caught! Attempted to access index {idx} which is out of range.")
            raise e  # Re-raise the exception to handle it according to the script's design