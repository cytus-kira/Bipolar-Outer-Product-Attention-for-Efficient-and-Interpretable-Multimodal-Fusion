class BOPA(nn.Module):
    def __init__(self, num_classes=512, combined_dim=64):
        super(BOPA, self).__init__()
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
