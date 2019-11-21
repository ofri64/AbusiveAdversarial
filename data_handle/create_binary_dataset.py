from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 1


def create_binary_label(data_df: pd.DataFrame) -> pd.DataFrame:
    has_abuse_label = (data_df["toxic"] == 1) | (data_df["severe_toxic"] == 1) | (data_df["obscene"] == 1) | \
                      (data_df["threat"] == 1) | (data_df["insult"] == 1) | (data_df["identity_hate"] == 1)
    data_df["abuse"] = has_abuse_label.astype(np.int32)
    binary_df = data_df[["id", "comment_text", "abuse"]]

    return binary_df


def split_dataset(data_df: pd.DataFrame, split_ratios=(0.8, 0.15)):
    train_ratio = split_ratios[0]
    dev_ratio = split_ratios[1] / (1 - train_ratio)

    train_df, dev_test_df = train_test_split(data_df, train_size=train_ratio, random_state=RANDOM_STATE)
    dev_df, test_df = train_test_split(dev_test_df, train_size=dev_ratio, random_state=RANDOM_STATE)

    return train_df, dev_df, test_df


def save_dataset(data_df: pd.DataFrame, filepath):
    data_df.to_csv(filepath, index=False)


def create_data(filepath, path_dir: Path):
    data = pd.read_csv(filepath)
    binary_data = create_binary_label(data)
    train_data, dev_data, test_data = split_dataset(binary_data)

    save_dataset(train_data, path_dir / "binary_train.csv")
    save_dataset(dev_data, path_dir / "binary_dev.csv")
    save_dataset(test_data, path_dir / "binary_test.csv")


if __name__ == '__main__':
    base_dir = Path().resolve().parent
    data_dir = base_dir / "jigsaw-toxic-comment-classification-challenge"
    data_file = data_dir / "train.csv"
    create_data(data_file, data_dir)
