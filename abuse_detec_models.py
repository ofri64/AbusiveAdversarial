import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class AbuseDetectNet(nn.Module):
    def __init__(self, label_dim, num_hidden_features=3):
        super().__init__()
        self.label_dim = label_dim
        self.bert_base = BertModel.from_pretrained('bert-base-uncased')
        self.num_hidden_features = num_hidden_features

        # compute number of features to classification layer
        bert_hidden_dim = 768
        features_dim = bert_hidden_dim * num_hidden_features  # concat last hidden layers
        self.cls_layer = nn.Linear(in_features=features_dim, out_features=label_dim)

    def forward(self, x):
        # switch bert model to eval mode (no fine tuning just features extraction)
        self.bert_base.eval()

        encoded_layers, _ = self.bert_base(x)
        feature_layers = encoded_layers[-self.num_hidden_features:]
        feature_tensor = torch.cat(feature_layers, dim=2)  # concatenate on feature index
        first_token_tensor = feature_tensor[:, 0, :]  # getting tensor with shape [batch, features]
        logits = self.cls_layer(first_token_tensor)

        return logits