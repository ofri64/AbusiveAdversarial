import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class AbuseDetectNet(nn.Module):
    def __init__(self, label_dim, num_hidden_features=3):
        super().__init__()
        self.label_dim = label_dim
        self.bert_base = BertModel.from_pretrained('bert-base-uncased')
        self.bert_base.eval()  # switch bert model to eval mode (no fine tuning just features extraction)
        self.num_hidden_features = num_hidden_features

        # compute number of features to classification layer
        bert_hidden_dim = 768
        features_dim = bert_hidden_dim * num_hidden_features  # concat last hidden layers
        self.cls_layer = nn.Linear(in_features=features_dim, out_features=label_dim)

    def forward(self, x):
        # reduce memory usage by not tracking gradients for bert model weights
        with torch.no_grad():
            encoded_layers, _ = self.bert_base(x)
            # features = encoded_layers[:, -self.num_hidden_features:, :, :]

        print(encoded_layers.size)
        # print(features.shape)
