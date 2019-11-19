import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class AbuseDetectNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_dim = config.label_dim
        self.bert_base = BertModel.from_pretrained(config.bert_model_name)
        self.num_hidden_features = config.num_hidden_features
        self.bert_hidden_size = self.bert_base.config.hidden_size
        self.bert_max_seq_length = self.bert_base.config.max_position_embeddings

        # compute number of features to classification layer
        features_dim = self.bert_hidden_size * self.num_hidden_features  # concat last hidden layers
        self.cls_layer = nn.Linear(in_features=features_dim, out_features=self.label_dim)
        self.train_params = self.cls_layer.parameters()

    def forward(self, x, attention_mask):
        # switch bert model to eval mode (no fine tuning just features extraction)
        self.bert_base.eval()

        encoded_layers, _ = self.bert_base(x, attention_mask=attention_mask)
        feature_layers = encoded_layers[-self.num_hidden_features:]
        feature_tensor = torch.cat(feature_layers, dim=2)  # concatenate on feature index
        first_token_tensor = feature_tensor[:, 0, :]  # getting tensor with shape [batch, features]
        logits = self.cls_layer(first_token_tensor)

        return logits
