import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils import data
from models_and_training.configs import AbuseDetectConfig, TrainingConfig
from models_and_training.models import AbuseDetectNet
from data_handle.datasets import CommentsDataset

ROOT_DIR = Path("C:\\Users\\t-ofklei\\Documents\\University\\AbusiveAdversarial")


def train_model(root_dir):
    logging.basicConfig(level=logging.INFO)

    # load configurations
    configs_dir = root_dir / "config_files"
    train_config_json = configs_dir / "training_config.json"
    train_config = TrainingConfig().from_json_file(train_config_json)
    model_config_json = configs_dir / "abuse_detect_config.json"
    model_config = AbuseDetectConfig().from_json_file(model_config_json)

    # training hyper parameters and configuration
    num_epochs = train_config.num_epochs
    learning_rate = train_config.learning_rate
    print_batch_step = 500
    device = train_config.device

    # model, loss and optimizer
    model = AbuseDetectNet(model_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.train_params, lr=learning_rate)

    # create dataset objects for train and dev
    training_path = root_dir / "jigsaw-toxic-comment-classification-challenge" / "binary_train.csv"
    dev_path = root_dir / "jigsaw-toxic-comment-classification-challenge" / "binary_dev.csv"
    max_seq_length = model.bert_max_seq_length
    training_dataset = CommentsDataset(training_path, max_seq_length)
    dev_dataset = CommentsDataset(dev_path, max_seq_length)

    train_config_dict = {"batch_size": train_config.batch_size, "num_workers": train_config.num_workers}
    training_loader = data.DataLoader(training_dataset, **train_config_dict)
    dev_loader = data.DataLoader(dev_dataset, **train_config_dict)

    # Start training
    model = model.to(device)
    model.train(mode=True)
    for epoch in range(num_epochs):

        running_loss = 0
        epoch_loss = 0

        for i, sample in enumerate(training_loader, 1):

            x, mask, y = sample
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x, mask)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            # print inter epoch statistics
            running_loss += loss.item()
            epoch_loss += loss.item() * train_config.batch_size
            if i % print_batch_step == 0:
                print(f"Epoch: {epoch + 1}, Batch Number: {i}, Average Loss: {(running_loss / print_batch_step):.3f}")
                running_loss = 0

        epoch_num = epoch + 1
        average_epoch_loss = epoch_loss / len(training_dataset)
        print(f"Epoch {epoch_num}: average loss is {average_epoch_loss}")


if __name__ == '__main__':
    train_model(ROOT_DIR)
