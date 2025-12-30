import math

import torch
from matplotlib.scale import scale_factory
from torch import nn
import torch.nn.functional as F

from get_clip_feature import ImageClassifier, TextClassifier


class AbsActivation(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class MyReLU(nn.Module):
    def __init__(self):
        super(MyReLU, self).__init__()

    def forward(self, x):
        return torch.maximum(x, torch.tensor(0.0, dtype=x.dtype))


class MyGeLU(nn.Module):
    def __init__(self):
        super(MyGeLU, self).__init__()

    def forward(self, x):
        return torch.maximum(x, torch.tensor(0.0, dtype=x.dtype))


class MySigmoid(nn.Module):
    def __init__(self):
        super(MySigmoid, self).__init__()

    def forward(self, x):
        return torch.maximum(x, torch.tensor(0.0, dtype=x.dtype))


class MySquare(nn.Module):
    def __init__(self):
        super(MySquare, self).__init__()

    def forward(self, x):
        return x * x


def batch_cosine_similarity_manual(query, key):
    # query: [N, D], key: [M, D], 其中 N 是 query 的数量，M 是 key 的数量，D 是向量的维度（128）

    # 计算点积 [N, M]
    dot_product = torch.matmul(query, key.T)  # query [64, 128] 和 key [M, 128] 的点积

    # 计算 query 和 key 的范数 [N, 1] 和 [M, 1]
    query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [64, 1]
    key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [M, 1]

    # 计算余弦相似度 [N, M]
    similarity = dot_product / (query_norm * key_norm.T)  # 广播规则，将 key_norm 转置为 [1, M]
    return similarity


class CombinedAttentionModel(nn.Module):
    def __init__(self, num_classes=512, combined_dim=256):
        super(CombinedAttentionModel, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)

        # Cross-attention mechanism for image features
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.query_dim)
        self.value_layer = nn.Linear(self.key_value_dim, self.query_dim)

        # Cross-attention mechanism for text features
        self.text_query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.text_key_layer = nn.Linear(self.key_value_dim, self.query_dim)
        self.text_value_layer = nn.Linear(self.key_value_dim, self.query_dim)

        # Final output layer
        self.linear1 = nn.Linear(self.query_dim * 2, combined_dim)
        self.output_fc = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(img_features)  # Image query
        key = self.key_layer(img_features)  # Image key
        value = self.value_layer(img_features)  # Image value

        text_query = self.text_query_layer(text_features)  # Text query
        text_key = self.text_key_layer(text_features)  # Text key
        text_value = self.text_value_layer(text_features)  # Text value

        # Combine keys (average of image and text keys)
        key_avg = (key + text_key) / 2

        # Compute attention scores and weights for image features
        attention_scores = torch.matmul(query, key_avg.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.query_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Compute attention scores and weights for text features
        attention_text_scores = torch.matmul(text_query, key_avg.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.query_dim, dtype=torch.float32))
        attention_text_weights = F.softmax(attention_text_scores, dim=-1)
        attention_text_output = torch.matmul(attention_text_weights, text_value)

        # Concatenate the attention outputs
        combined_output = torch.cat((attention_output, attention_text_output), dim=-1)

        # Final linear layers
        output = self.linear1(combined_output)
        output = self.relu(output)
        output = self.output_fc(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedCrossAttentionModel(nn.Module):
    def __init__(self, num_classes=512, hid_dim=128, combined_dim=32):
        super(CombinedCrossAttentionModel, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)

        # Cross-attention mechanism
        self.query_layer = nn.Linear(self.query_dim, hid_dim, bias=False)
        self.key_layer = nn.Linear(self.key_value_dim, hid_dim, bias=False)
        self.value_layer = nn.Linear(self.key_value_dim, hid_dim)

        # Final output layer
        # self.linear1 = nn.Linear(hid_dim, combined_dim)
        self.output_fc = nn.Linear(hid_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # For matching the dimensions in residual connections
        # self.img_projection = nn.Linear(self.query_dim, self.query_dim)  # Project img features to same dimension

    def forward(self, text_features, img_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)
        key = self.key_layer(img_features)
        value = self.value_layer(img_features)

        # Compute attention scores and weights
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.query_dim, dtype=torch.float32))
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        output = attention_output * attention_output
        output = self.output_fc(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedCrossAttentionCatModelsqmax(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(CombinedCrossAttentionCatModelsqmax, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)

        # Cross-attention mechanism
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.w_b_layer = torch.diag(torch.randn(64))
        self.bias = nn.Parameter(torch.randn(64))

        # Final output layer
        # self.linear1 = nn.Linear(self.query_dim * 2, combined_dim)  # Adjusted for concatenation
        self.output_fc = nn.Linear(combined_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        diag_matrix = torch.diag(torch.diagonal(self.w_b_layer)).to('cuda')
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]

        qk = (query @ key.T) / key.size(0) + self.bias
        # Concatenate query and attention output
        score = torch.matmul(qk, diag_matrix)
        score = score * score
        attention_output = score @ value
        combined_output = torch.cat((query, attention_output), dim=-1)  # [batch_size, query_dim * 2]
        # combined_output = query+attention_output  # [batch_size, query_dim * 2]
        combined_output = self.gelu(combined_output)

        # Process combined output through linear layers
        # combined_output = self.linear1(combined_output)  # [batch_size, combined_dim]
        output = self.output_fc(combined_output)
        output = self.dropout(output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedCrossAttentionModelsqmax(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(CombinedCrossAttentionModelsqmax, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)

        # Cross-attention mechanism
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.w_b_layer = torch.diag(torch.randn(64))
        self.bias = nn.Parameter(torch.randn(64))

        # Final output layer
        # self.linear1 = nn.Linear(self.query_dim * 2, combined_dim)  # Adjusted for concatenation
        self.output_fc = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        diag_matrix = torch.diag(torch.diagonal(self.w_b_layer)).to('cuda')
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]

        qk = (query @ key.T) / key.size(0) + self.bias
        # Concatenate query and attention output
        score = torch.matmul(qk, diag_matrix)
        score = score * score
        attention_output = score @ value
        # combined_output = query+attention_output  # [batch_size, query_dim * 2]

        # Process combined output through linear layers
        # combined_output = self.linear1(combined_output)  # [batch_size, combined_dim]
        output = self.output_fc(attention_output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedCrossAttentionModelCos(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(CombinedCrossAttentionModelCos, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)

        # Cross-attention mechanism
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)

        # Final output layer
        # self.linear1 = nn.Linear(self.query_dim * 2, combined_dim)  # Adjusted for concatenation
        self.output_fc = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]

        # Compute attention scores and weights
        cos_sim = F.cosine_similarity(query.unsqueeze(1), key.unsqueeze(0), dim=-1)  # Q和K的余弦相似度
        # 将余弦相似度转化到 [0, 1] 范围内
        cos_sim = (cos_sim + 1) / 2  # 余弦相似度调整到 [0, 1]
        # 计算加权的值 V
        output = torch.matmul(cos_sim, value)
        output = self.relu(output)
        # Concatenate query and attention output
        # combined_output = torch.cat((query, output), dim=-1)  # [batch_size, query_dim * 2]
        # combined_output = query+attention_output  # [batch_size, query_dim * 2]

        # Process combined output through linear layers
        # combined_output = self.linear1(combined_output)  # [batch_size, combined_dim]
        output = self.output_fc(output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedCrossAttentionCatModelCos(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64, plus=False):
        super(CombinedCrossAttentionCatModelCos, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)
        self.plus = plus
        # Cross-attention mechanism
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)

        # Final output layer
        # self.linear1 = nn.Linear(self.query_dim * 2, combined_dim)  # Adjusted for concatenation
        self.output_fc = nn.Linear(combined_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]

        cos_sim = F.cosine_similarity(query.unsqueeze(1), key.unsqueeze(0), dim=-1)  # Q和K的余弦相似度
        if self.plus:
            cos_sim = cos_sim + 1  # 余弦相似度调整到 [0, 1]
        # 计算加权的值 V
        output = torch.matmul(cos_sim, value)

        combined_output = torch.cat((query, output), dim=-1)  # [batch_size, query_dim * 2]
        # combined_output = self.relu(combined_output)
        combined_output = self.gelu(combined_output)
        # combined_output = query+attention_output  # [batch_size, query_dim * 2]

        # Process combined output through linear layers
        # combined_output = self.linear1(combined_output)  # [batch_size, combined_dim]
        output = self.output_fc(combined_output)
        output = self.dropout(output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedCrossAttentionCatModelCosSimSquare(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(CombinedCrossAttentionCatModelCosSimSquare, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)

        # Cross-attention mechanism
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)

        # Final output layer
        # self.linear1 = nn.Linear(combined_dim* 2, combined_dim)  # Adjusted for concatenation
        self.output_fc = nn.Linear(combined_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]

        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]

        qk_ = torch.einsum("nl,nm->nlm", query,
                           key) / (query_norm * key_norm).unsqueeze(1)

        # qk_ = qk_ * qk_
        # kv_ = kv_+1# [batch_size, key_dim, value_dim, key_dim]

        # Compute query attention with the key-value relationship
        qkv = torch.einsum("nlm,nl->nm", qk_, value)
        qkv = (qkv * qkv)
        combined_output = torch.cat((query, qkv), dim=-1)

        output = self.output_fc(combined_output)
        output = self.dropout(output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedCrossAttentionCatModelCosSim(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64, act=None, cat_act="gelu"):
        super(CombinedCrossAttentionCatModelCosSim, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)

        # Cross-attention mechanism
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.dropout = nn.Dropout(0.2)
        # Final output layer
        # self.linear1 = nn.Linear(combined_dim* 2, combined_dim)  # Adjusted for concatenation
        self.output_fc = nn.Linear(combined_dim * 2, 1)
        if cat_act == "gelu":
            self.cat_act = nn.GELU()
        if cat_act == "relu":
            self.cat_act = nn.ReLU()
        if cat_act == "sigmoid":
            self.cat_act = nn.Sigmoid()
        else:
            self.act = None
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        if act == "gelu":
            self.act = nn.GELU()
        if act == "relu":
            self.act = nn.ReLU()
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]

        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]

        qk_ = torch.einsum("nl,nm->nlm", query,
                           key) / (query_norm * key_norm).unsqueeze(1)
        # if self.act:
        #     qk_ = self.act(qk_)
        # kv_ = kv_+1# [batch_size, key_dim, value_dim, key_dim]

        # Compute query attention with the key-value relationship
        qkv = torch.einsum("nlm,nl->nm", qk_, value)
        combined_output = torch.cat((query, qkv), dim=-1)
        if self.cat_act:
            combined_output = self.cat_act(combined_output)
        # combined_output = self.linear1(combined_output)  # [batch_size, combined_dim]
        output = self.output_fc(combined_output)
        output = self.dropout(output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedCrossAttentionCatModelCosSimTimeline(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(CombinedCrossAttentionCatModelCosSimTimeline, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)

        # Cross-attention mechanism
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)

        # Final output layer
        # self.linear1 = nn.Linear(combined_dim* 2, combined_dim)  # Adjusted for concatenation
        self.output_fc = nn.Linear(combined_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]

        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]

        qk_ = torch.einsum("nl,nm->nlm", query,
                           key) / (query_norm * key_norm).unsqueeze(1)
        qk_ = torch.cumsum(qk_, dim=1)

        qk_ = self.relu(qk_)
        # kv_ = kv_+1# [batch_size, key_dim, value_dim, key_dim]

        # Compute query attention with the key-value relationship
        qkv = torch.einsum("nlm,nl->nm", qk_, value)
        combined_output = torch.cat((query, qkv), dim=-1)  # [batch_size, query_dim * 2]
        combined_output = self.relu(combined_output)
        # combined_output = query+attention_output  # [batch_size, query_dim * 2]

        # Process combined output through linear layers
        # combined_output = self.linear1(combined_output)  # [batch_size, combined_dim]
        output = self.output_fc(combined_output)
        output = self.dropout(output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class SimpleCombinedCrossAttentionWithoutCatModelCosSim(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(SimpleCombinedCrossAttentionWithoutCatModelCosSim, self).__init__()
        self.img = ImageClassifier()
        self.text = TextClassifier()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)

        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)

        self.output_fc = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img, text, id):
        # Extract features from image and text models
        img_features = self.img(img)
        text_features = self.text(text, id)
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]
        qk_ = torch.einsum("nl,nm->nlm", query,
                           key) / (query_norm * key_norm).unsqueeze(1)

        qkv = torch.einsum("nlm,nl->nm", qk_, value)
        combined_output = self.gelu(qkv)
        output = self.output_fc(combined_output)
        output = self.dropout(output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedCrossAttentionWithoutCatModelCosSim(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(CombinedCrossAttentionWithoutCatModelCosSim, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)
        # Cross-attention mechanism
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)

        self.output_fc = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]
        qk_ = torch.einsum("nl,nm->nlm", query,
                           key) / (query_norm * key_norm).unsqueeze(1)
        qkv = torch.einsum("nlm,nl->nm", qk_, value)
        # combined_output = torch.cat((query, qkv), dim=-1)  # [batch_size, query_dim * 2]
        combined_output = qkv * qkv
        output = self.output_fc(combined_output)
        output = self.dropout(output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedCrossAttentionWithoutCatModelCosSimScale(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64, scale=64):
        super(CombinedCrossAttentionWithoutCatModelCosSimScale, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)
        self.scale = scale
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.output_fc = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]
        qk_ = torch.einsum("nl,nm->nlm", query,
                           key) / self.scale
        qkv = torch.einsum("nlm,nl->nm", qk_, value)
        combined_output = self.gelu(qkv)
        output = self.output_fc(combined_output)
        output = self.dropout(output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedCrossAttentionWithoutCatModelCosSimTimeLine(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(CombinedCrossAttentionWithoutCatModelCosSimTimeLine, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)

        # Cross-attention mechanism
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)

        # Final output layer
        # self.linear1 = nn.Linear(combined_dim* 2, combined_dim)  # Adjusted for concatenation
        self.output_fc = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]

        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]

        qk_ = torch.einsum("nl,nm->nlm", query,
                           key) / (query_norm * key_norm).unsqueeze(1)
        qk_ = torch.cumsum(qk_, dim=1)
        qk_ = self.relu(qk_)
        # kv_ = kv_+1# [batch_size, key_dim, value_dim, key_dim]

        # Compute query attention with the key-value relationship
        qkv = torch.einsum("nlm,nl->nm", qk_, value)
        # combined_output = torch.cat((query, qkv), dim=-1)  # [batch_size, query_dim * 2]
        # combined_output = self.relu(qkv)
        # combined_output = query+attention_output  # [batch_size, query_dim * 2]
        combined_output = qkv * qkv
        # Process combined output through linear layers
        # combined_output = self.linear1(combined_output)  # [batch_size, combined_dim]
        output = self.output_fc(combined_output)
        output = self.dropout(output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedCrossAttentionCatModelCosSimTimeLine(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(CombinedCrossAttentionCatModelCosSimTimeLine, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)

        # Cross-attention mechanism
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)

        # Final output layer
        # self.linear1 = nn.Linear(combined_dim* 2, combined_dim)  # Adjusted for concatenation
        self.output_fc = nn.Linear(combined_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]

        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]

        qk_ = torch.einsum("nl,nm->nlm", query,
                           key) / (query_norm * key_norm).unsqueeze(1)
        qk_ = self.relu(qk_)
        qk_ = torch.cumsum(qk_, dim=1)

        # kv_ = kv_+1# [batch_size, key_dim, value_dim, key_dim]

        # Compute query attention with the key-value relationship
        qkv = torch.einsum("nlm,nl->nm", qk_, value)
        combined_output = torch.cat((query, qkv), dim=-1)  # [batch_size, query_dim * 2]
        # combined_output = self.relu(combined_output)
        # combined_output = query+attention_output  # [batch_size, query_dim * 2]
        combined_output = self.relu(qk_)
        # Process combined output through linear layers
        # combined_output = self.linear1(combined_output)  # [batch_size, combined_dim]
        output = self.output_fc(combined_output)
        output = self.dropout(output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_heads=8, num_classes=512, combined_dim=256):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)

        # Cross-attention mechanism with multi-heads
        assert self.query_dim % self.num_heads == 0, "Query dimension must be divisible by num_heads"
        self.head_dim = self.query_dim // self.num_heads

        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.query_dim)
        self.value_layer = nn.Linear(self.key_value_dim, self.query_dim)

        # Output projection layer
        self.output_fc = nn.Linear(self.query_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, key_value_dim]
        value = self.value_layer(img_features)  # [batch_size, key_value_dim]

        # Reshape for multi-head attention
        batch_size = query.size(0)
        query = query.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        key = key.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        value = value.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))  # [batch_size, num_heads, 1, 1]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, 1, 1]

        # Compute attention output
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, num_heads, 1, head_dim]
        attention_output = attention_output.squeeze(dim=2)  # [batch_size, num_heads, head_dim]

        # Concatenate the heads and project the result
        attention_output = attention_output.view(batch_size, self.query_dim)  # [batch_size, query_dim]
        output = self.output_fc(attention_output)  # [batch_size, 1]
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CombinedModel(nn.Module):
    def __init__(self, num_classes=512, combined_dim=256):
        super(CombinedModel, self).__init__()

        self.linear1 = nn.Linear(num_classes * 2, 1)
        self.output_fc = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        output = torch.cat((img_features, text_features), dim=-1)

        output = self.linear1(output)
        # output = self.relu(output)
        # output = self.output_fc(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class TestCtModel(nn.Module):
    def __init__(self, num_classes=512, combined_dim=256):
        super(CombinedModel, self).__init__()

        self.linear1 = nn.Linear(num_classes * 2, 1)
        self.output_fc = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        output = torch.cat((img_features, text_features), dim=-1)

        output = self.linear1(output)
        # output = self.relu(output)
        # output = self.output_fc(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class TestModel(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(TestModel, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.output_fc = nn.Linear(combined_dim, int(combined_dim / 2))
        self.output_fc_2 = nn.Linear(int(combined_dim / 2), 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]
        qk_ = torch.einsum("nl,nm->nlm", query,
                           key) / (query_norm * key_norm).unsqueeze(1)
        # qk_ = torch.einsum("nl,nm->nlm", query,
        #                    key) / 128
        qkv = torch.einsum("nlm,nl->nm", qk_, value)
        # cat_data = torch.cat((qkv, query), dim=-1)
        combined_output = self.gelu(qkv)
        output = self.output_fc(combined_output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class TestdoubleModel(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(TestdoubleModel, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)
        self.query_layer = nn.Linear(self.query_dim, combined_dim)
        self.key_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer_img = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer_text = nn.Linear(self.key_value_dim, combined_dim)
        self.value_layer = nn.Linear(self.key_value_dim, combined_dim)
        self.output_fc = nn.Linear(combined_dim * 2, int(combined_dim / 2))
        self.output_fc_2 = nn.Linear(int(combined_dim / 2), 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value_img = self.value_layer_img(img_features)  # [batch_size, query_dim]
        value_text = self.value_layer_text(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value_img = self.dropout(value_img)
        value_text = self.dropout(value_text)
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]
        qk_ = torch.einsum("nl,nm->nlm", query,
                           key) / (query_norm * key_norm).unsqueeze(1)
        # qk_ = torch.einsum("nl,nm->nlm", query,
        #                    key) / 128
        img = torch.einsum("nlm,nl->nm", qk_, value_img)
        text = torch.einsum("nlm,nl->nm", qk_.permute(0, 2, 1), value_text)
        # cat_data = torch.cat((qkv, query), dim=-1)
        concat = torch.cat((img, text), dim=-1)
        # concat = concat * concat
        concat = self.gelu(concat)
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)

        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class TestCatModel(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(TestCatModel, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, combined_dim)
        self.output_fc_2 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        concat = torch.cat((img_features, text_features), dim=-1)
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)

        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class TestSoftmaxModel(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(TestSoftmaxModel, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.query_dim, self.query_dim)
        self.value_layer = nn.Linear(self.query_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        # self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        # query = self.dropout(query)
        # key = self.dropout(key)

        attention_scores = torch.matmul(query, key.T) / torch.sqrt(
            torch.tensor(self.query_dim, dtype=torch.float32))  # [batch_size, 1, 1]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1, 1]

        attention_output = torch.matmul(attention_weights, value)  # [batch_size, 1, query_dim]

        attention_output = attention_output.squeeze(dim=1)  # [batch_size, query_dim]

        # Concatenate query and attention output
        # combined_output = torch.cat((query, attention_output), dim=-1)

        # output = self.output_fc(combined_output)
        # output = self.dropout(output)
        # output = self.gelu(output)
        output = self.output_fc_2(attention_output)
        # output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        # output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class TestCosCatModel(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(TestCosCatModel, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]
        qk_ = torch.einsum("nl,nm->nlm", query,
                           key) / (query_norm * key_norm).unsqueeze(1)
        img = torch.einsum("nlm,nl->nm", qk_, img_features)
        text = torch.einsum("nlm,nl->nm", qk_.permute(0, 2, 1), text_features)
        concat = torch.cat((img, text), dim=-1)
        concat = concat * concat
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output, qk_[0], concat


class TestOptCosCatModel(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(TestOptCosCatModel, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]
        img_kv_ = torch.einsum("nm,nm->n", key, img_features).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query, img_features).unsqueeze(1)
        img = query * img_kv_ / (query_norm * key_norm)
        text = key * text_qv_ / (query_norm * key_norm)
        concat = torch.cat((img, text), dim=-1)
        concat = concat * concat
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification
        return output


class TestOptCosCatModel2(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(TestOptCosCatModel2, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        img_kv_ = torch.einsum("nm,nm->n", key, img_features).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query, text_features).unsqueeze(1)
        img = query * img_kv_
        text = key * text_qv_
        max_abs_img = torch.max(torch.abs(img))
        max_abs_text = torch.max(torch.abs(text))
        scale_img = img / max_abs_img
        scale_text = text / max_abs_text
        concat = torch.cat((scale_img, scale_text), dim=-1)
        concat = concat * concat
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output



class TestOptCosCatModel3(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(TestOptCosCatModel3, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        img_kv_ = torch.einsum("nm,nm->n", key * key, img_features).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query * query, text_features).unsqueeze(1)
        img = query * query * img_kv_
        text = key * key * text_qv_
        max_abs_query = torch.max(torch.abs(query))
        max_abs_key = torch.max(torch.abs(key))

        scale_img = img / max_abs_query
        scale_text = text / max_abs_key

        concat = torch.cat((scale_img, scale_text), dim=-1)
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class BOPAFast(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(BOPAFast, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)

        text_map = self.key_layer(text_features)

        query = self.dropout(img_map)
        key = self.dropout(text_map)

        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]
        img_kv_ = (key * img_features).sum(dim=1).unsqueeze(1)
        text_qv_ = (query * text_features).sum(dim=1).unsqueeze(1)
        # img_kv_ = torch.einsum("nm,nm->n", key , img_features).unsqueeze(1)
        # text_qv_ = torch.einsum("nm,nm->n", query , text_features).unsqueeze(1)
        img = query * img_kv_ / (query_norm * key_norm)
        text = key * text_qv_ / (query_norm * key_norm)
        concat = torch.cat((img, text), dim=-1)
        concat = concat * concat
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class MultiHeadBOPA(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64, num_heads=4):
        super(MultiHeadBOPA, self).__init__()
        self.num_heads = num_heads
        self.query_layers = nn.ModuleList([nn.Linear(num_classes, num_classes) for _ in range(num_heads)])
        self.key_layers = nn.ModuleList([nn.Linear(num_classes, num_classes) for _ in range(num_heads)])

        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2 * num_heads, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)

        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features, text_features):
        head_outputs = []
        for i in range(self.num_heads):
            q = self.dropout(self.query_layers[i](img_features))
            k = self.dropout(self.key_layers[i](text_features))

            q_norm = torch.norm(q, p=2, dim=-1, keepdim=True)
            k_norm = torch.norm(k, p=2, dim=-1, keepdim=True)

            img_kv_ = (k * img_features).sum(dim=1, keepdim=True)
            text_qv_ = (q * text_features).sum(dim=1, keepdim=True)

            img = q * img_kv_ / (q_norm * k_norm + 1e-8)
            text = k * text_qv_ / (q_norm * k_norm + 1e-8)

            head_outputs.append(torch.cat((img, text), dim=-1))  # [B, 2d]

        concat = torch.cat(head_outputs, dim=-1)  # [B, 2d * num_heads]
        concat = concat * concat

        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)

        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)

        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)

        return output
class BOPACt(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(BOPACt, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.dropout = nn.Dropout(0.2)
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.output_fc = nn.Linear(num_classes * 2, combined_dim)
        self.output_fc_2 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)

        text_map = self.key_layer(text_features)

        query = self.dropout(img_map)
        key = self.dropout(text_map)
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]
        img_kv_ = (key * img_features).sum(dim=1).unsqueeze(1)
        text_qv_ = (query * text_features).sum(dim=1).unsqueeze(1)
        # img_kv_ = torch.einsum("nm,nm->n", key , img_features).unsqueeze(1)
        # text_qv_ = torch.einsum("nm,nm->n", query , text_features).unsqueeze(1)
        img = query * img_kv_ / (query_norm * key_norm)
        text = key * text_qv_ / (query_norm * key_norm)
        concat = torch.cat((img, text), dim=-1)
        concat = concat * concat
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = torch.clip(output, -6, 6)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class HateCLIP(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(HateCLIP, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes, combined_dim * 2)
        self.output_fc_2 = nn.Linear(combined_dim * 2, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        qk_flat = img_features * text_features  # 逐元素相乘 Q_i * K_i
        output = self.output_fc(qk_flat)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output

class CrossAttentionTransformerTokens(nn.Module):
    """
    扁平2D输入 -> token级 cross-attention -> 你的原MLP头（二分类）。
    - 文本(text)作 Query，图像(image)作 Key/Value（与原先一致）。
    - 输入必须是 [B, T*D] 形式；token 数写死在类里（TQ/TK/D）。
    """
    def __init__(self, combined_dim=64):
        super().__init__()
        # ======= 写死的 token 规格：改成你实际的 =======
        self.TQ = 16   # 文本 token 数
        self.TK = 16   # 图像 token 数
        self.D  = 64   # 每个 token 的特征维度
        # ============================================

        # 保留你的结构：线性投影（按 token 的最后一维做）、dropout/激活、MLP 头
        self.query_layer = nn.Linear(self.D, self.D)
        self.key_layer   = nn.Linear(self.D, self.D)
        self.value_layer = nn.Linear(self.D, self.D)

        self.dropout = nn.Dropout(0.2)

        # 你的 MLP 头（保持一致）
        self.output_fc   = nn.Linear(self.D, combined_dim)
        self.output_fc_2 = nn.Linear(combined_dim, combined_dim // 2)
        self.output_fc_3 = nn.Linear(combined_dim // 2, 1)

        self.gelu    = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features_flat, text_features_flat):
        """
        img_features_flat:  [B, TK*D]  -> Key/Value 来源（图像）
        text_features_flat: [B, TQ*D]  -> Query 来源（文本）
        返回:
          prob: [B, 1]
        """
        B = text_features_flat.size(0)

        # [B, T*D] -> [B, T, D]
        q = text_features_flat.view(B, self.TQ, self.D)
        k = img_features_flat.view(B, self.TK, self.D)
        v = img_features_flat.view(B, self.TK, self.D)

        # 线性投影（逐 token）
        q = self.dropout(self.query_layer(q))  # [B, TQ, D]
        k = self.dropout(self.key_layer(k))    # [B, TK, D]
        v = self.dropout(self.value_layer(v))  # [B, TK, D]

        # 单头缩放点积注意力：softmax(QK^T / sqrt(D)) V
        scale = math.sqrt(self.D)
        attn_scores  = torch.bmm(q, k.transpose(1, 2)) / scale      # [B, TQ, TK]
        attn_weights = F.softmax(attn_scores, dim=-1)                # [B, TQ, TK]
        attn_output  = torch.bmm(attn_weights, v)                    # [B, TQ, D]

        # 池化得到样本级向量（保持简单，用 mean；如需[CLS]只取 attn_output[:,0,:]）
        pooled = attn_output.mean(dim=1)  # [B, D]

        # 你的原 MLP 头
        x = self.output_fc(pooled)
        x = self.dropout(self.gelu(x))
        x = self.output_fc_2(x)
        x = self.dropout(self.gelu(x))
        x = self.output_fc_3(x)
        x = self.dropout(x)
        prob = self.sigmoid(x)  # [B, 1]

        return prob,attn_weights

class CrossAttentionTransformer(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(CrossAttentionTransformer, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.value_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes, combined_dim)
        self.output_fc_2 = nn.Linear(combined_dim, int(combined_dim / 2))
        self.output_fc_3 = nn.Linear(int(combined_dim / 2), 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]

        # Compute attention scores and weights
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.query_dim, dtype=torch.float32))  # [batch_size, 1, 1]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1, 1]
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, 1, query_dim]
        attention_output = attention_output.squeeze(dim=1)  # [batch_size, query_dim]
        output = self.output_fc(attention_output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output,attention_weights


class CrossAttentionCosFormer(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(CrossAttentionCosFormer, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.value_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes, combined_dim)
        self.output_fc_2 = nn.Linear(combined_dim, int(combined_dim / 2))
        self.output_fc_3 = nn.Linear(int(combined_dim / 2), 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        value = self.value_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        value = self.dropout(value)
        query = self.relu(query)
        key = self.relu(key)
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (query_norm * key_norm)  # [batch_size, 1, 1]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1, 1]
        attention_output = torch.matmul(attention_weights, value)
        output = self.output_fc(attention_output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class CrossAttentionPlusCatTransformer(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(CrossAttentionPlusCatTransformer, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.value_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        query = self.query_layer(text_features)  # [batch_size, query_dim]
        key = self.key_layer(img_features)  # [batch_size, query_dim]
        value = self.value_layer(img_features)  # [batch_size, query_dim]
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)  # [batch_size, query_dim]

        # Compute attention scores and weights
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.query_dim, dtype=torch.float32))  # [batch_size, 1, 1]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1, 1]
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, 1, query_dim]
        attention_output = attention_output.squeeze(dim=1)  # [batch_size, query_dim]
        concat = torch.cat((attention_output, query), dim=-1)
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class TestCosCatModelForTextImg(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(TestCosCatModelForTextImg, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim * 2, self.key_value_dim)
        self.value_img_layer = nn.Linear(self.query_dim, self.query_dim)
        self.value_text_layer = nn.Linear(self.key_value_dim * 2, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        img_value_map = self.value_img_layer(img_features)
        text_value_map = self.value_text_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]
        img_kv_ = torch.einsum("nm,nm->n", key, img_value_map).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query, text_value_map).unsqueeze(1)
        img = query * img_kv_ / (query_norm * key_norm)
        text = key * text_qv_ / (query_norm * key_norm)
        concat = torch.cat((img, text), dim=-1)
        concat = concat * concat
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class SmallCosCatModel(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(SmallCosCatModel, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, combined_dim)
        self.output_fc_2 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]
        qk_ = torch.einsum("nl,nm->nlm", query,
                           key) / (query_norm * key_norm).unsqueeze(1)
        img = torch.einsum("nlm,nl->nm", qk_, img_features)
        text = torch.einsum("nlm,nl->nm", qk_.permute(0, 2, 1), text_features)
        concat = torch.cat((img, text), dim=-1)
        concat = concat * concat
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class MultiHeadCosineAttentionModel(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64, num_heads=8):
        super(MultiHeadCosineAttentionModel, self).__init__()
        self.num_heads = num_heads
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.head_dim = num_classes // num_heads  # 每个头的维度

        # Linear layers for each attention head
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)

        # Output layers
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)

        # Activation and Dropout
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def split_heads(self, x):
        # 将输入张量分成多个头（head），保留每个样本的结构
        batch_size = x.size(0)  # [batch_size, num_classes]
        x = x.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        return x

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)  # [batch_size, seq_len, num_classes]
        text_map = self.key_layer(text_features)  # [batch_size, seq_len, num_classes]
        img_map = self.dropout(img_map)
        text_map = self.dropout(text_map)
        # Split into multiple heads
        img_map = self.split_heads(img_map)  # [batch_size, num_heads, seq_len, head_dim]
        text_map = self.split_heads(text_map)  # [batch_size, num_heads, seq_len, head_dim]

        # Compute cosine similarity for each head
        query_norm = torch.norm(img_map, p=2, dim=-1, keepdim=True)  # [batch_size, num_heads, seq_len, 1]
        key_norm = torch.norm(text_map, p=2, dim=-1, keepdim=True)  # [batch_size, num_heads, seq_len, 1]

        # Compute attention scores (cosine similarity)
        attn_scores = torch.einsum("bhn,bhm->bhnm", img_map, text_map) / (query_norm * key_norm).unsqueeze(
            -1)  # [batch_size, num_heads, num_heads, num_heads]

        # Weighted sum of attention heads
        img_attention = torch.einsum("bhnm,bhm->bhn", attn_scores,
                                     img_map)  # [batch_size, num_heads, num_heads, head_dim]
        text_attention = torch.einsum("bhnm,bhm->bhn", attn_scores.permute(0, 1, 3, 2),
                                      text_map)  # [batch_size, num_heads, num_heads, head_dim]

        # Concatenate image and text features
        concat = torch.cat((img_attention, text_attention), dim=-1)  # [batch_size, num_heads, seq_len, 2*head_dim]

        # Flatten the attention output for final layers
        concat = concat.view(concat.size(0), -1)  # [batch_size, num_heads * 2 * head_dim]
        concat = concat * concat
        # Final output through fully connected layers
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)

        return output


class ImgFC(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(ImgFC, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes, int(num_classes / 2))
        self.output_fc_2 = nn.Linear(int(num_classes / 2), combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, _):
        # Extract features from image and text models
        output = self.output_fc(img_features)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class TextFc(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(TextFc, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, int(num_classes))
        self.output_fc_2 = nn.Linear(int(num_classes), combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, _, text):
        # Extract features from image and text models
        output = self.output_fc(text)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class Fc(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(Fc, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 3, int(num_classes))
        self.output_fc_2 = nn.Linear(int(num_classes), combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img, text):
        # Extract features from image and text models
        data = torch.cat((img, text), dim=-1)
        output = self.output_fc(data)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class AllFc(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(AllFc, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        concat = torch.cat((img_features, text_features), dim=-1)
        # concat = concat * concat
        output1 = self.output_fc(concat)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class BOPA_without_sq(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(BOPA_without_sq, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        img_kv_ = torch.einsum("nm,nm->n", key, img_features).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query, text_features).unsqueeze(1)
        img = query * img_kv_
        text = key * text_qv_
        max_abs_img = torch.max(torch.abs(img))
        max_abs_text = torch.max(torch.abs(text))
        scale_img = img / max_abs_img
        scale_text = text / max_abs_text
        concat = torch.cat((scale_img, scale_text), dim=-1)
        # concat = concat * concat
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class Dot_Product_Mapping(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(Dot_Product_Mapping, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        img_kv_ = torch.einsum("nm,nm->n", key, img_features).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query, text_features).unsqueeze(1)
        img = query * img_kv_
        text = key * text_qv_
        # scale_text = text/max_abs_text
        concat = torch.cat((img, text), dim=-1)
        # concat = concat * concat
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class Dot_Product_Mapping_Normalization(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(Dot_Product_Mapping_Normalization, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        img_kv_ = torch.einsum("nm,nm->n", key, img_features).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query, text_features).unsqueeze(1)
        img = query * img_kv_
        text = key * text_qv_
        max_abs_img = torch.max(torch.abs(img))
        max_abs_text = torch.max(torch.abs(text))
        scale_img = img / max_abs_img
        scale_text = text / max_abs_text
        # scale_text = text/max_abs_text
        concat = torch.cat((scale_img, scale_text), dim=-1)
        # concat = concat * concat
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class Dot_Product_Mapping_Normalization_Non_Lin(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(Dot_Product_Mapping_Normalization_Non_Lin, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        img_kv_ = torch.einsum("nm,nm->n", key, img_features).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query, text_features).unsqueeze(1)
        img = query * img_kv_
        text = key * text_qv_
        max_abs_img = torch.max(torch.abs(img))
        max_abs_text = torch.max(torch.abs(text))
        scale_img = img / max_abs_img
        scale_text = text / max_abs_text
        # scale_text = text/max_abs_text
        concat = torch.cat((scale_img, scale_text), dim=-1)
        # concat = concat * concat
        concat = self.relu(concat)
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class BOPA_without_scale(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(BOPA_without_scale, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        img_kv_ = torch.einsum("nm,nm->n", key, img_features).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query, text_features).unsqueeze(1)
        img = query * img_kv_
        text = key * text_qv_
        # max_abs_img = torch.max(torch.abs(img))
        # max_abs_text = torch.max(torch.abs(text))
        # scale_img = img/max_abs_img
        # scale_text = text/max_abs_text
        concat = torch.cat((img, text), dim=-1)
        concat = concat * concat
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class Dot_Product_Mapping_Normalization_Non_Lin_Abs(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(Dot_Product_Mapping_Normalization_Non_Lin_Abs, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.abs = AbsActivation()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        img_kv_ = torch.einsum("nm,nm->n", key, img_features).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query, text_features).unsqueeze(1)
        img = query * img_kv_
        text = key * text_qv_
        max_abs_img = torch.max(torch.abs(img))
        max_abs_text = torch.max(torch.abs(text))
        scale_img = img / max_abs_img
        scale_text = text / max_abs_text
        # scale_text = text/max_abs_text
        concat = torch.cat((scale_img, scale_text), dim=-1)
        # concat = concat * concat
        concat = self.abs(concat)
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class Dot_Product_Mapping_Normalization_Non_Lin_Sigmoid(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(Dot_Product_Mapping_Normalization_Non_Lin_Sigmoid, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        img_kv_ = torch.einsum("nm,nm->n", key, img_features).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query, text_features).unsqueeze(1)
        img = query * img_kv_
        text = key * text_qv_
        max_abs_img = torch.max(torch.abs(img))
        max_abs_text = torch.max(torch.abs(text))
        scale_img = img / max_abs_img
        scale_text = text / max_abs_text
        # scale_text = text/max_abs_text
        concat = torch.cat((scale_img, scale_text), dim=-1)
        # concat = concat * concat
        concat = self.sigmoid(concat)
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class Dot_Product_Mapping_Normalization_Non_Lin_Softmax(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(Dot_Product_Mapping_Normalization_Non_Lin_Softmax, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        img_kv_ = torch.einsum("nm,nm->n", key, img_features).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query, text_features).unsqueeze(1)
        img = query * img_kv_
        text = key * text_qv_
        max_abs_img = torch.max(torch.abs(img))
        max_abs_text = torch.max(torch.abs(text))
        scale_img = img / max_abs_img
        scale_text = text / max_abs_text
        # scale_text = text/max_abs_text
        concat = torch.cat((scale_img, scale_text), dim=-1)
        # concat = concat * concat
        concat = self.softmax(concat)
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class BOPAOpt(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(BOPAOpt, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(num_classes * 2, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        query_sq = query * query
        key_sq = key * key
        img_kv_ = (key_sq * img_features).sum(dim=1).unsqueeze(1)
        text_qv_ = (query_sq * text_features).sum(dim=1).unsqueeze(1)
        img = query_sq * img_kv_
        text = key_sq * text_qv_
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)  # [32, 1]
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True)  # [32, 1]
        scale_factory = query_norm * key_norm
        scale_factory = scale_factory * scale_factory

        scale_img = img / scale_factory
        scale_text = text / scale_factory
        concat = torch.cat((scale_img, scale_text), dim=-1)

        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class TFN(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(TFN, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear((num_classes + 1) * (num_classes + 1), num_classes)
        self.output_fc_2 = nn.Linear(int((num_classes + 1) * (num_classes + 1) / 8),
                                     int(((num_classes + 1) * (num_classes + 1)) / 16))
        self.output_fc_3 = nn.Linear(int(((num_classes + 1) * (num_classes + 1)) / 16),
                                     int(((num_classes + 1) * (num_classes + 1)) / 32))
        self.output_fc_4 = nn.Linear(int(((num_classes + 1) * (num_classes + 1)) / 32),
                                     int(((num_classes + 1) * (num_classes + 1)) / 64))
        self.output_fc_5 = nn.Linear(int(((num_classes + 1) * (num_classes + 1)) / 64), 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        # Extract features from image and text models
        img_features = torch.cat([img_features, torch.ones(img_features.size(0), 1, device=img_features.device)], dim=1)
        text_features = torch.cat([text_features, torch.ones(text_features.size(0), 1, device=text_features.device)],
                                  dim=1)
        outer = torch.einsum('bi,bj->bij', img_features, text_features)
        outer = outer.view(img_features.size(0), -1)
        output = self.output_fc(outer)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_4(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_5(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


class MCB(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64, scale_classes=64):
        super(MCB, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.scale_fc_1 = nn.Linear(num_classes, scale_classes, bias=False)
        identity_weight = torch.eye(scale_classes, num_classes)  # [scale_classes, num_classes]
        # self.scale_fc_1.weight = nn.Parameter(identity_weight, requires_grad=False)
        self.scale_fc_2 = nn.Linear(num_classes, scale_classes, bias=False)
        # self.scale_fc_2.weight = nn.Parameter(identity_weight, requires_grad=False)
        self.output_fc = nn.Linear(scale_classes, int(scale_classes / 2))
        self.output_fc_2 = nn.Linear(int(scale_classes / 2), int(scale_classes / 4))
        self.output_fc_3 = nn.Linear(int(scale_classes / 4), 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features, text_features):
        img_map = self.scale_fc_1(img_features)
        text_map = self.scale_fc_2(text_features)
        x_fft = torch.fft.fft(img_map, dim=-1)
        y_fft = torch.fft.fft(text_map, dim=-1)

        z_fft = x_fft * y_fft
        z = torch.fft.ifft(z_fft, dim=-1).real
        output = self.output_fc(z)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output


import torch
import torch.nn as nn
import torch.nn.functional as F


class XLinearBlockAttention(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64, scale_classes=64, num_blocks=8):
        super(XLinearBlockAttention, self).__init__()
        self.num_blocks = num_blocks
        self.block_dim = scale_classes // num_blocks

        # 映射到交互空间（图像 / 文本）
        self.scale_fc_1 = nn.Linear(num_classes, scale_classes, bias=False)
        self.scale_fc_2 = nn.Linear(num_classes, scale_classes, bias=False)

        # Q, K, V for each block attention
        self.q_proj = nn.Linear(self.block_dim, self.block_dim)
        self.k_proj = nn.Linear(self.block_dim, self.block_dim)
        self.v_proj = nn.Linear(self.block_dim, self.block_dim)

        self.output_fc = nn.Linear(scale_classes * scale_classes, num_classes)
        self.output_fc_2 = nn.Linear(num_classes, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)

        self.dropout = nn.Dropout(0.2)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features, text_features):
        B = img_features.size(0)

        # Step 1: Map input features
        img_map = self.scale_fc_1(img_features)  # [B, D]
        text_map = self.scale_fc_2(text_features)  # [B, D]
        img_map = self.relu(img_map)
        text_map = self.relu(text_map)
        # Step 2: Outer Product to form FLIM
        flim = torch.einsum('bi,bj->bij', img_map, text_map)  # [B, D, D]
        flim = flim.view(B, self.num_blocks, self.block_dim, self.num_blocks, self.block_dim)  # [B, n, d, n, d]

        # Step 3: Apply self-attention per block (on last two dims)
        attended_blocks = []
        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
                block = flim[:, i, :, j, :]  # [B, d, d]

                Q = self.q_proj(block)  # [B, d, d]
                K = self.k_proj(block)  # [B, d, d]
                V = self.v_proj(block)  # [B, d, d]

                attn_score = torch.matmul(Q, K.transpose(-1, -2)) / (self.block_dim ** 0.5)  # [B, d, d]
                attn_weight = F.softmax(attn_score, dim=-1)
                block_attn = torch.matmul(attn_weight, V)  # [B, d, d]

                attended_blocks.append(block_attn)

        # Step 4: Flatten back
        attended = torch.cat(attended_blocks, dim=-1).view(B, -1)  # [B, D*D]

        # Step 5: Project & output
        out = self.output_fc(attended)
        out = self.gelu(self.dropout(out))
        out = self.output_fc_2(out)
        out = self.gelu(self.dropout(out))
        out = self.output_fc_3(out)
        out = self.sigmoid(self.dropout(out))
        return out


class MUTAN(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64, scale_classes=64):
        super(MUTAN, self).__init__()
        self.query_dim = num_classes
        self.key_value_dim = num_classes
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(0.2)
        self.scale_fc_1 = nn.Linear(num_classes, scale_classes, bias=False)
        self.scale_fc_2 = nn.Linear(num_classes, scale_classes, bias=False)
        self.output_fc = nn.Linear(scale_classes, int(scale_classes / 2))
        self.output_fc_2 = nn.Linear(int(scale_classes / 2), int(scale_classes / 4))
        self.output_fc_3 = nn.Linear(int(scale_classes / 4), 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.rank = 5
        self.linear_x = nn.ModuleList([nn.Linear(scale_classes, scale_classes) for _ in range(self.rank)])
        self.linear_y = nn.ModuleList([nn.Linear(scale_classes, scale_classes) for _ in range(self.rank)])

    def forward(self, img_features, text_features):
        img_map = self.scale_fc_1(img_features)
        text_map = self.scale_fc_2(text_features)
        x1, x2 = img_map, text_map
        fusion = 0
        for r in range(self.rank):
            x_r = self.linear_x[r](x1)  # [B, feature]
            y_r = self.linear_y[r](x2)  # [B, feature]
            fusion += x_r * y_r  # element-wise multiplication
        output = self.output_fc(fusion)
        output = self.dropout(output)
        output = self.gelu(output)
        output1 = self.output_fc_2(output)
        output = self.dropout(output1)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # Sigmoid for binary classification

        return output

class PerformerFusionModel2(nn.Module):
    """
    Performer 风格融合头（核注意力），接口与 TestOptCosCatModel2 一致：
    forward(img_features, text_features) -> [B,1]
    """
    def __init__(self, num_classes=512, combined_dim=64,
                 n_tokens=8, m_features=64, eps=1e-6):
        super().__init__()
        assert num_classes % n_tokens == 0, \
            "num_classes 必须能被 n_tokens 整除，用于 [B,d]->[B,T,D_tok] 的 reshape"

        self.d = num_classes
        self.T = n_tokens
        self.D_tok = num_classes // n_tokens
        self.m = m_features
        self.eps = eps

        # token 级投影：对最后一维 D_tok 做线性变换
        self.q_proj = nn.Linear(self.D_tok, self.D_tok)
        self.k_proj = nn.Linear(self.D_tok, self.D_tok)
        self.v_proj = nn.Linear(self.D_tok, self.D_tok)

        # 简单的正特征映射 φ(x) = elu(Wx)+1，保证非负（Performer 要求）
        self.feature_map = nn.Linear(self.D_tok, self.m)

        # 把 pooled token 向量映射回 d 维（分别给 img/text 分支）
        self.out_proj_img = nn.Linear(self.D_tok, self.d)
        self.out_proj_txt = nn.Linear(self.D_tok, self.d)

        # 尾部和 TestOptCosCatModel2 保持一致
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(self.d * 2, self.d)
        self.output_fc_2 = nn.Linear(self.d, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def phi(self, x):
        """
        x: [B, T, D_tok] -> [B, T, m]，非负核特征
        """
        return F.elu(self.feature_map(x)) + 1.0

    def kernel_attn(self, q, k, v):
        """
        单向 FAVOR+ 核注意力：
        q,k,v: [B, T, D_tok]
        返回: [B, D_tok]，在 token 维度上平均池化之后的向量
        """
        Q_phi = self.phi(q)   # [B, T, m]
        K_phi = self.phi(k)   # [B, T, m]

        # K_phi^T V  -> [B, m, D_tok]
        KV = torch.einsum("btm,btd->bmd", K_phi, v)

        # K_phi^T 1_T -> sum over T: [B, m]
        K_sum = K_phi.sum(dim=1)  # [B, m]

        # 分母 D: [B, T, 1]
        denom = torch.einsum("btm,bm->bt", Q_phi, K_sum).unsqueeze(-1) + self.eps

        # 分子: [B, T, D_tok]
        num = torch.einsum("btm,bmd->btd", Q_phi, KV)

        out_tokens = num / denom              # [B, T, D_tok]
        pooled = out_tokens.mean(dim=1)       # [B, D_tok]
        return pooled

    def forward(self, img_features, text_features):
        """
        img_features, text_features: [B, d]
        """
        B, d = img_features.size()
        assert d == self.d

        # [B, d] -> [B, T, D_tok]
        img_seq = img_features.view(B, self.T, self.D_tok)
        txt_seq = text_features.view(B, self.T, self.D_tok)

        # 文本 -> 图像 方向的注意力
        q_txt = self.q_proj(txt_seq)  # [B, T, D_tok]
        k_img = self.k_proj(img_seq)
        v_img = self.v_proj(img_seq)

        # 图像 -> 文本 方向的注意力
        q_img = self.q_proj(img_seq)
        k_txt = self.k_proj(txt_seq)
        v_txt = self.v_proj(txt_seq)

        pooled_txt2img = self.kernel_attn(q_txt, k_img, v_img)  # [B, D_tok]
        pooled_img2txt = self.kernel_attn(q_img, k_txt, v_txt)  # [B, D_tok]

        fused_img = self.out_proj_img(pooled_txt2img)  # [B, d]
        fused_txt = self.out_proj_txt(pooled_img2txt)  # [B, d]

        concat = torch.cat([fused_img, fused_txt], dim=-1)  # [B, 2d]
        concat = concat * concat

        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # [B,1]

        return output
class LinformerFusionModel2(nn.Module):
    """
    Linformer 风格融合头：沿 token 维度做低秩投影的 cross-attention。
    接口与 TestOptCosCatModel2 一致。
    """
    def __init__(self, num_classes=512, combined_dim=64,
                 n_tokens=8, k_tokens=4):
        super().__init__()
        assert num_classes % n_tokens == 0, \
            "num_classes 必须能被 n_tokens 整除"

        self.d = num_classes
        self.T = n_tokens
        self.D_tok = num_classes // n_tokens
        self.k_tokens = k_tokens

        self.q_proj = nn.Linear(self.D_tok, self.D_tok)
        self.k_proj = nn.Linear(self.D_tok, self.D_tok)
        self.v_proj = nn.Linear(self.D_tok, self.D_tok)

        # 沿 token 维度的投影矩阵 E ∈ ℝ^{k×T}
        self.E = nn.Parameter(
            torch.randn(k_tokens, n_tokens) / (k_tokens ** 0.5)
        )

        # 把 pooled token 向量映回 d 维
        self.out_proj_img = nn.Linear(self.D_tok, self.d)
        self.out_proj_txt = nn.Linear(self.D_tok, self.d)

        # 尾部保持一致
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(self.d * 2, self.d)
        self.output_fc_2 = nn.Linear(self.d, combined_dim)
        self.output_fc_3 = nn.Linear(combined_dim, 1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def linformer_attn(self, q, k, v):
        """
        q,k,v: [B, T, D_tok]
        返回: [B, D_tok]
        """
        B, T, D = q.size()

        # 投影 K,V 到 [B, k_tokens, D_tok]
        # [k,T] @ [B,T,D] -> [B,k,D]
        Kp = torch.einsum("kt,btd->bkd", self.E, k)
        Vp = torch.einsum("kt,btd->bkd", self.E, v)

        # scores: [B, T, k]
        scores = torch.einsum("btd,bkd->btk", q, Kp) / math.sqrt(D)
        P = F.softmax(scores, dim=-1)       # [B, T, k]

        # out_tokens: [B, T, D]
        out_tokens = torch.einsum("btk,bkd->btd", P, Vp)

        pooled = out_tokens.mean(dim=1)     # [B, D_tok]
        return pooled

    def forward(self, img_features, text_features):
        """
        img_features, text_features: [B, d]
        """
        B, d = img_features.size()
        assert d == self.d

        img_seq = img_features.view(B, self.T, self.D_tok)
        txt_seq = text_features.view(B, self.T, self.D_tok)

        q_txt = self.q_proj(txt_seq)
        k_img = self.k_proj(img_seq)
        v_img = self.v_proj(img_seq)

        q_img = self.q_proj(img_seq)
        k_txt = self.k_proj(txt_seq)
        v_txt = self.v_proj(txt_seq)

        pooled_txt2img = self.linformer_attn(q_txt, k_img, v_img)
        pooled_img2txt = self.linformer_attn(q_img, k_txt, v_txt)

        fused_img = self.out_proj_img(pooled_txt2img)  # [B, d]
        fused_txt = self.out_proj_txt(pooled_img2txt)  # [B, d]

        concat = torch.cat([fused_img, fused_txt], dim=-1)  # [B, 2d]
        concat = concat * concat

        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        output = self.sigmoid(output)  # [B,1]

        return output

def choose_model(type_as, num_classes, combined):
    model = None
    if type_as == "fc":
        model = Fc(num_classes, combined)
    if type_as == "sqmax":
        model = CombinedCrossAttentionCatModelsqmax(num_classes, combined)
    if type_as == "cos_normalize":
        model = CombinedCrossAttentionCatModelCos(num_classes, combined)
    if type_as == "cos_normalize_plus":
        model = CombinedCrossAttentionCatModelCos(num_classes, combined, plus=True)
    if type_as == "cos_normalize_without_cat":
        model = CombinedCrossAttentionModelCos(num_classes, combined)
    if type_as == "softmax_without_cat":
        model = CombinedCrossAttentionModel(num_classes, combined)
    if type_as == "sqmax_without_cat":
        model = CombinedCrossAttentionModelsqmax(num_classes, combined)
    if type_as == "cos_sim_without_act":
        model = CombinedCrossAttentionCatModelCosSim(num_classes, combined, act=None)
    if type_as == "cos_sim_without_act_lin_relu":
        model = CombinedCrossAttentionCatModelCosSim(num_classes, combined, act=None, cat_act="relu")
    if type_as == "cos_sim_without_act_lin_gelu":
        model = CombinedCrossAttentionCatModelCosSim(num_classes, combined, act=None, cat_act="gelu")
    if type_as == "cos_sim_without_act_lin_sigmoid":
        model = CombinedCrossAttentionCatModelCosSim(num_classes, combined, act=None, cat_act="sigmoid")
    if type_as == "cos_sim_square":
        model = CombinedCrossAttentionCatModelCosSimSquare(num_classes, combined)
    if type_as == "cos_sim_gelu":
        model = CombinedCrossAttentionCatModelCosSim(num_classes, combined, act="gelu")
    if type_as == "cos_sim_relu":
        model = CombinedCrossAttentionCatModelCosSim(num_classes, combined, act="relu")
    if type_as == "cos_sim_sigmoid":
        model = CombinedCrossAttentionCatModelCosSim(num_classes, combined, act="relu")
    if type_as == "cos_sim_without_cat":
        model = CombinedCrossAttentionWithoutCatModelCosSim(num_classes, combined)
    if type_as == "cos_sim_without_cat_scale":
        model = CombinedCrossAttentionWithoutCatModelCosSimScale(num_classes, combined, combined)
    if type_as == "cos_sim_without_cat_timeline":
        model = CombinedCrossAttentionWithoutCatModelCosSimTimeLine(num_classes, combined)
    if type_as == "cos_sim_timeline":
        model = CombinedCrossAttentionCatModelCosSimTimeLine(num_classes, combined)
    if type_as == "test":
        model = TestModel(num_classes, combined)
    # if type_as == "test_ct_gelu":
    #     model = TestCtGeluModel(num_classes, combined)
    if type_as == "test_double_cos_model":
        model = TestdoubleModel(num_classes, combined)
    if type_as == "test_cat_model":
        model = TestCatModel(num_classes, combined)
    if type_as == "test_cos_cat_model":
        model = TestCosCatModel(num_classes, combined)
    if type_as == "test_cos_cat_for_imgtext_model":
        model = TestCosCatModelForTextImg(num_classes, combined)
    if type_as == "small_cos_cat_model":
        model = SmallCosCatModel(num_classes, combined)
    if type_as == "softmax_speech":
        model = TestSoftmaxModel(num_classes, combined)
    if type_as == "mutihead_cos_model":
        model = MultiHeadCosineAttentionModel(num_classes, combined, num_heads=2)
    if type_as == "TestOptCosCatModel":
        model = TestOptCosCatModel(num_classes, combined)
    if type_as == "img_fc":
        model = ImgFC(num_classes, combined)
    if type_as == "text_fc":
        model = TextFc(num_classes, combined)
    if type_as == "all_fc":
        model = AllFc(num_classes, combined)
    if type_as == "TestOptCosCatModel2":
        model = TestOptCosCatModel2(num_classes, combined)
    if type_as == "TestOptCosCatModel3":
        model = TestOptCosCatModel3(num_classes, combined)
    if type_as == "BOPAFast":
        model = BOPAFast(num_classes, combined)
    if type_as == "BOPAOpt":
        model = BOPAOpt(num_classes, combined)
    if type_as == "HateCLIP":
        model = HateCLIP(num_classes, combined)
    if type_as == "CrossAttentionTransformer":
        model = CrossAttentionTransformer(num_classes, combined)
    if type_as == "CrossAttentionCosFormer":
        model = CrossAttentionCosFormer(num_classes, combined)
    if type_as == "CrossAttentionPlusCatTransformer":
        model = CrossAttentionPlusCatTransformer(num_classes, combined)
    if type_as == "Dot_Product_Mapping_Normalization_Non_Lin":
        model = Dot_Product_Mapping_Normalization_Non_Lin(num_classes, combined)
    if type_as == "CA_token":
        model = CrossAttentionTransformerTokens(num_classes)

    if type_as == "Dot_Product_Mapping_Normalization_Non_Lin_Softmax":
        model = Dot_Product_Mapping_Normalization_Non_Lin_Softmax(num_classes, combined)
    if type_as == "Dot_Product_Mapping_Normalization_Non_Lin_Sigmoid":
        model = Dot_Product_Mapping_Normalization_Non_Lin_Sigmoid(num_classes, combined)
    if type_as == "Dot_Product_Mapping_Normalization_Non_Lin_Abs":
        model = Dot_Product_Mapping_Normalization_Non_Lin_Abs(num_classes, combined)
    if type_as == "Dot_Product_Mapping_Normalization":
        model = Dot_Product_Mapping_Normalization(num_classes, combined)
    if type_as == "Dot_Product_Mapping":
        model = Dot_Product_Mapping(num_classes, combined)
    if type_as == "BOPA_without_sq":
        model = BOPA_without_sq(num_classes, combined)
    if type_as == "BOPA_without_scale":
        model = BOPA_without_scale(num_classes, combined)
    if type_as == "TFN":
        model = TFN(num_classes, combined)
    if type_as == "MUTAN":
        model = MUTAN(num_classes, combined)
    if type_as == "MCB":
        model = MCB(num_classes, combined)
    if type_as == "XLinearBlockAttention":
        model = XLinearBlockAttention(num_classes, combined)
    if type_as == "BOPACt":
        model = BOPACt(num_classes, combined)
    if type_as == "MultiHeadBOPA":
        model = MultiHeadBOPA(num_classes, combined)
    if type_as == "Performer":
        model = PerformerFusionModel2(num_classes, combined)
    if type_as == "Linformer":
        model = LinformerFusionModel2(num_classes, combined)

    return model


if __name__ == '__main__':
    dummy_input_img = torch.randn(64, 3, 256, 256)  # 比如一个输入大小为 (1, 3, 224, 224) 的图像
    dummy_input_text = torch.randint(low=0, high=1, size=(64, 128))  # 比如一个输入大小为 (1, 3, 224, 224) 的图像
    dummy_input_id = torch.randn(64, 128)  # 比如一个输入大小为 (1, 3, 224, 224) 的图像
    model = SimpleCombinedCrossAttentionWithoutCatModelCosSim(num_classes=256, combined_dim=128)
    model.eval()
    from torchviz import make_dot

    # Assuming your model takes (dummy_input_img, dummy_input_text, dummy_input_id)
    output = model(dummy_input_img, dummy_input_text, dummy_input_id)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render("model_architecture", format="png")
    # dot_file = "model_architecture_figure"
    # source = Source.from_file(dot_file)
    # source.render("model_architecture_figure", format="png")
