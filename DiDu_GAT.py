# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# 使用kaiming初始化

class GatedInteractionUnit(nn.Module):
    def __init__(self, feature_size):
        super(GatedInteractionUnit, self).__init__()
        self.feature_size = feature_size
        # print(f"feature_size:{feature_size}")
        # 交互层的转换
        self.transform = nn.Linear(feature_size, feature_size)
        # 门控机制
        self.gate_self = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.Sigmoid()
        )
        self.gate_other = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.Sigmoid()
        )

    def forward(self, features_self, features_other):
        # print(f"features_self:{features_self.shape}")
        # print(f"features_other:{features_other.shape}")
        # 初步交互调整
        interaction_self = F.relu(self.transform(features_other))
        # print(f"interaction_self:{interaction_self.shape}")
        interaction_other = F.relu(self.transform(features_self))
        # print(f"interaction_other:{interaction_other.shape}")
        # 更新特征
        features_self = features_self + interaction_self
        features_other = features_other + interaction_other

        # 组合特征以应用门控
        combined_features = torch.cat([features_self, features_other], dim=-1)
        gate_self = self.gate_self(combined_features)
        gate_other = self.gate_other(combined_features)

        # 应用门控以更新特征
        updated_self = gate_self * features_self + (1 - gate_self) * features_other
        updated_other = gate_other * features_other + (1 - gate_other) * features_self
        return updated_self, updated_other

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, device,concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.device = device
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # 调整为正确的输入维度
        self.W = nn.Linear(in_features, out_features, bias=False).to(device)
        # print(f"out_fea:{out_features}")
        self.bn=nn.BatchNorm1d(out_features).to(device)
        self.a = nn.Parameter(torch.zeros(size=(out_features, 1)), requires_grad=True).to(device)

        # # 使用Kaiming初始化替换原有的Xavier初始化
        nn.init.kaiming_uniform_(self.W.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.a.data, mode='fan_in', nonlinearity='leaky_relu')


    def forward(self, input, adj):
        input = input.to(self.device)
        # print("Input shape:", input.shape)
        # print("Input type:", input.type())
        adj = adj.to(self.device)
        h = self.W(input)
        # print("H shape after W:", h.shape)
        h = h.permute(0, 2, 1)
        h = self.bn(h)  # 应用批量归一化
        h = h.permute(0, 2, 1)

        h = self.leakyrelu(h)   # 先应用非线性激活
        a_input = torch.matmul(h, self.a)   # 注意力得分
        e = self.leakyrelu(a_input.squeeze(2))

        # print("Raw attention scores:", e.detach().cpu().numpy())
        zero_vec = -9e15 * torch.ones_like(e).to(self.device)
        e_expanded = e.unsqueeze(2)
        zero_vec_expanded = zero_vec.unsqueeze(2)

        attention = torch.where(adj > 0, e_expanded, zero_vec_expanded)
        attention = F.softmax(attention, dim=1)
        # print("Raw attention scores after softmax:", attention.detach().cpu().numpy())

        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)

        if self.concat:
            return self.leakyrelu(h_prime)

        else:
            return h_prime


class DualChannelGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, device):
        super(DualChannelGAT, self).__init__()
        self.dropout = dropout
        self.nclass = nclass
        self.device = device
        # Multi-head attention for each channel
        self.attentions_channel1 = [GraphAttentionLayer(nfeat, nhid, dropout, alpha, device) for _ in range(nheads)]
        self.reduction_layer1 = nn.Linear(nheads * nhid * 2, 768, device=device)
        self.attentions_channel2 = [GraphAttentionLayer(nfeat, nhid, dropout, alpha, device) for _ in range(nheads)]
        self.reduction_layer2 = nn.Linear(nheads * nhid * 2, 768, device=device)

        # 初始化门控交互单元
        self.gated_interaction = GatedInteractionUnit(nhid*nheads)  # 注意特征维度应匹配两个通道的总维度

        self.out_att = GraphAttentionLayer(nhid * nheads * 2, nhid, dropout, alpha, device)

        self.pool = nn.AdaptiveAvgPool1d(1).to(device)

        self.classifier = nn.Linear(nhid, nclass).to(device)  # 注意更新分类器的输入维度匹配最终特征维度

    def forward(self, fea, adj1, adj2):
        # print(f"fea:{fea}")
        # print(f"adj1:{adj1}")
        # print(f"adj2:{adj2}")
        fea = fea.to(self.device)
        # print(f"fea shape: {fea.shape}")
        adj1 = adj1.to(self.device)
        # print(f"adj1 shape: {adj1.shape}")
        adj2 = adj2.to(self.device)
        # print(f"adj2 shape: {adj2.shape}")
        x1 = torch.cat([att(fea, adj1) for att in self.attentions_channel1], dim=2)
        # print(f"x1 post shape: {x1.shape}")
        x2 = torch.cat([att(fea, adj2) for att in self.attentions_channel2], dim=2)
        # print(f"x2 post shape: {x2.shape}")

        # 使用门控交互层
        x1, x2 = self.gated_interaction(x1, x2)
        # print(f"x1 gated_interaction shape: {x1.shape}")
        # print(f"x2 gated_interaction shape: {x2.shape}")
        x = torch.cat((x1, x2), dim=2)
        x = self.out_att(x, adj1 + adj2)
        x = x.transpose(1, 2)  # Prepare for pooling
        x = self.pool(x).squeeze(-1)

        logits = self.classifier(x)
        # print(f"模型内的输出：{logits}")

        return logits

