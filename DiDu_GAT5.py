import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GatedInteractionUnit(nn.Module):
    def __init__(self, feature_size):
        super(GatedInteractionUnit, self).__init__()
        self.feature_size = feature_size
        self.transform = nn.Linear(feature_size, feature_size)
        self.gate_self = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size),
            nn.Sigmoid()
        )
        self.gate_other = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size),
            nn.Sigmoid()
        )

    def forward(self, features_self, features_other):
        interaction_self = F.relu(self.transform(features_other))
        interaction_other = F.relu(self.transform(features_self))
        features_self = features_self + interaction_self
        features_other = features_other + interaction_other
        combined_features = torch.cat([features_self, features_other], dim=-1)
        gate_self = self.gate_self(combined_features)
        gate_other = self.gate_other(combined_features)
        updated_self = gate_self * features_self + (1 - gate_self) * features_other
        updated_other = gate_other * features_other + (1 - gate_other) * features_self
        return updated_self, updated_other


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, device, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.device = device
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.W = nn.Linear(in_features, out_features, bias=False).to(device)
        self.a_src = nn.Parameter(torch.zeros(size=(out_features, 1)), requires_grad=True).to(device)
        self.a_dst = nn.Parameter(torch.zeros(size=(out_features, 1)), requires_grad=True).to(device)

        nn.init.kaiming_uniform_(self.W.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.a_src, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.a_dst, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, input, adj, return_attention_scores=False):
        h = self.W(input)  # [batch_size, num_nodes, out_features]

        # Compute attention scores
        a_input_src = torch.matmul(h, self.a_src)  # [batch_size, num_nodes, 1]
        a_input_dst = torch.matmul(h, self.a_dst)  # [batch_size, num_nodes, 1]
        e = a_input_src + a_input_dst.transpose(1, 2)  # [batch_size, num_nodes, num_nodes]
        e = self.leakyrelu(e)

        # Masked attention
        zero_vec = -9e15 * torch.ones_like(e).to(self.device)
        attention = torch.where(adj > 0, e, zero_vec)

        # Softmax over neighbors
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Weighted sum
        h_prime = torch.matmul(attention, h)  # [batch_size, num_nodes, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=1)  # [batch_size, channel]
        y = self.fc1(y)  # [batch_size, channel // reduction]
        y = self.relu(y)
        y = self.fc2(y)  # [batch_size, channel]
        y = self.sigmoid(y)
        y = y.unsqueeze(1)  # [batch_size, 1, channel]
        return x * y  # [batch_size, num_nodes, channel]


class DualChannelGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, num_layers, se_reduction, device):
        super(DualChannelGAT, self).__init__()
        self.dropout = dropout
        self.nclass = nclass
        self.device = device
        self.alpha = alpha
        self.nheads = nheads
        self.num_layers = num_layers

        # 通道1
        self.attentions_channel1 = nn.ModuleList(
            [GraphAttentionLayer(nfeat if l == 0 and h == 0 else nheads * nhid, nhid, dropout, alpha, device)
             for l in range(num_layers) for h in range(nheads)]
        )
        self.reduction_layer1 = nn.Linear(nheads * nhid, nfeat, device=device)

        # 通道2
        self.attentions_channel2 = nn.ModuleList(
            [GraphAttentionLayer(nfeat if l == 0 and h == 0 else nheads * nhid, nhid, dropout, alpha, device)
             for l in range(num_layers) for h in range(nheads)]
        )
        self.reduction_layer2 = nn.Linear(nheads * nhid, nfeat, device=device)

        # Gated Interaction Unit
        self.gated_interaction = GatedInteractionUnit(nfeat)

        # SE Blocks
        self.se1 = SEBlock(nfeat, reduction=se_reduction)
        self.se2 = SEBlock(nfeat, reduction=se_reduction)

        # Final Reduction Layers
        self.final_reduction_layer = nn.Sequential(
            nn.Linear(2 * nfeat, 512),
            nn.ReLU(),
            nn.Linear(512, nhid)
        ).to(device)

        # Output Attention Layer
        self.out_att = GraphAttentionLayer(nhid, nhid, dropout, alpha, device, concat=False)

        # Pooling and Classification
        self.pool = nn.AdaptiveAvgPool1d(1).to(device)
        self.additional_linear = nn.Linear(nhid, nhid).to(device)
        self.bn_before_classifier = nn.BatchNorm1d(nhid).to(device)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.classifier = nn.Linear(nhid, nclass).to(device)

    def forward_channel(self, features, adj, attentions, num_heads, num_layers, reduction_layer):
        x = features
        for layer in range(num_layers):
            # print(f"Layer {layer} input shape: {x.shape}")
            h = []
            for head in range(num_heads):
                idx = layer * num_heads + head
                att_layer = attentions[idx]
                out = att_layer(x, adj)
                h.append(out)
            h = torch.cat(h, dim=2)  # [batch_size, num_nodes, 768]
            # print(f"Layer {layer} concatenated output shape: {h.shape}")
            x_prev = x
            x = F.relu(reduction_layer(h))  # [batch_size, num_nodes, 768]
            # print(f"Layer {layer} after reduction and ReLU: {x.shape}")
            x = x + x_prev  # [batch_size, num_nodes, 768]
            # print(f"Layer {layer} after skip connection: {x.shape}")
        return x

    def forward(self, fea, adj1, adj2):
        fea = fea.to(self.device)
        adj1 = adj1.to(self.device)
        adj2 = adj2.to(self.device)

        # Channel 1
        x1 = self.forward_channel(fea, adj1, self.attentions_channel1, self.nheads, self.num_layers,
                                  self.reduction_layer1)

        # Channel 2
        x2 = self.forward_channel(fea, adj2, self.attentions_channel2, self.nheads, self.num_layers,
                                  self.reduction_layer2)

        # SE Blocks
        x1 = self.se1(x1)
        x2 = self.se2(x2)

        # Gated Interaction
        x1, x2 = self.gated_interaction(x1, x2)

        # Concatenate Channels
        x = torch.cat((x1, x2), dim=2)  # [batch_size, num_nodes, 2 * nfeat]

        # Final Reduction
        x = self.final_reduction_layer(x)  # [batch_size, num_nodes, nhid]

        # Output Attention
        x = self.out_att(x, adj1 + adj2)  # [batch_size, num_nodes, nhid]

        # Pooling
        x = x.transpose(1, 2)  # [batch_size, nhid, num_nodes]
        x = self.pool(x).squeeze(-1)  # [batch_size, nhid]
        # Classification Layers
        x = self.additional_linear(x)  # [batch_size, nhid]
        x = self.bn_before_classifier(x)
        x = self.leakyrelu(x)
        # 这里x已经通过了LeakyReLU激活函数
        features_before_classifier_final = x.detach()  # 保留这个状态作为特征
        logits = self.classifier(x)  # [batch_size, nclass]

        return logits,features_before_classifier_final
