import pandas as pd
import torch.utils.data as data
from pytorch_pretrained_bert import BertTokenizer


class CommentsDataset(data.Dataset):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data = None

    def _load_file(self):
        if self.data is None:
            self.data = pd.read_csv(self.filepath)

    def __getitem__(self, item):
        self._load_file()
        id_index, comment_text, y = self.data.iloc[item]

        # convert comment text to tokens and then to discrete token ids
        text_bert_format = "[CLS] " + comment_text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(text_bert_format)
        x = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        return id_index, x, y

    def __len__(self):
        self._load_file()
        return len(self.data)
