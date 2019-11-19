import pandas as pd
import torch
import torch.utils.data as data
from pytorch_pretrained_bert import BertTokenizer


class CommentsDataset(data.Dataset):
    def __init__(self, filepath, seq_length):
        super().__init__()
        self.filepath = filepath
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.seq_length = seq_length
        self.data = None

    def _load_file(self):
        if self.data is None:
            self.data = pd.read_csv(self.filepath)

    def __getitem__(self, item):
        self._load_file()
        _, comment_text, y = self.data.iloc[item]

        # convert comment text to tokens and then to discrete token ids
        text_bert_format = "[CLS] " + comment_text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(text_bert_format)
        current_length = len(tokenized_text)
        tokens_sequence = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # enforce that the maximum sequence length
        # create also a tokens mask
        mask_dtype = torch.int64
        if len(tokens_sequence) >= self.seq_length:
            tokens_sequence = tokens_sequence[:self.seq_length]
            x = torch.tensor(tokens_sequence)
            tokens_mask = torch.ones(self.seq_length, dtype=mask_dtype)  # all ones mean we want to entire sequence
        else:
            padding_tensor = torch.zeros(self.seq_length - current_length, dtype=torch.int64)
            x = torch.cat([torch.tensor(tokens_sequence), padding_tensor])
            tokens_mask = torch.cat([torch.ones(current_length, dtype=mask_dtype), padding_tensor])

        return x, tokens_mask, y

    def __len__(self):
        self._load_file()
        return len(self.data)
