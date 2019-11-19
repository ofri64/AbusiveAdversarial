import torch.utils.data as data
import pandas as pd


class CommentsDataset(data.Dataset):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.data = None

    def _load_file(self):
        if self.data is None:
            self.data = pd.read_csv(self.filepath)

    def __getitem__(self, item):
        self._load_file()
        id_index, x, y = self.data.iloc[item]
        return id_index, x, y

    def __len__(self):
        self._load_file()
        return len(self.data)
