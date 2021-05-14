import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

torch.multiprocessing.set_sharing_strategy("file_system")

NUM_USERS = 943
NUM_MOVIES = 1682
NUM_GENRES = 19


class FactorizationMachine(pl.LightningModule):
    def __init__(self, field_dims, num_factors):
        super(FactorizationMachine, self).__init__()
        num_inputs = int(sum(field_dims))
        # self.embedding = nn.Embedding(num_inputs, num_factors)
        # self.fc = nn.Embedding(num_inputs, 1)
        # self.linear_layer = nn.Embedding(num_inputs, 1)
        # self.embedding = nn.Linear(num_inputs, num_factors)
        self.embedding = nn.Parameter(
            torch.randn(num_inputs, num_factors), requires_grad=True
        )
        self.linear_layer = nn.Linear(num_inputs, 1, bias=True)

    def forward(self, x):
        out_1 = torch.matmul(x, self.embedding).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.embedding.pow(2)).sum(
            1, keepdim=True
        )  # S_2

        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.linear_layer(x)
        out = out_inter + out_lin
        # square_of_sum = torch.sum(self.embedding(x), axis=1) ** 2
        # sum_of_square = torch.sum(self.embedding(x) ** 2, axis=1)
        # # breakpoint()
        # tmp = torch.matmul(
        #     self.fc(x).reshape((1, -1)), self.linear_layer(x).reshape((-1, 1))
        # )
        # x = tmp + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)

        return torch.clip(out.squeeze(), min=1, max=5)

    def training_step(self, batch, batch_idx):
        inputs, rating = batch
        rating = rating.to(torch.float32)
        output = self.forward(inputs)
        loss = F.mse_loss(rating, output)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.5, momentum=0.5, weight_decay=1e-3
        )  # learning rate
        return optimizer


class MlDataset(Dataset):
    def __init__(self, file_path: str, field_dims=[943, 1682, 19]):
        self.df = pd.read_csv(file_path, delimiter="\t", header=None)
        self.genre_df = pd.read_csv(
            "data/ml-100k/u.item", delimiter="|", header=None, encoding="latin-1"
        )[range(5, 24)]
        self.field_dims = field_dims

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # user_onehot, movie_onehot, genre_onehot
        # return self.df[0][index] - 1, self.df[1][index] - 1, self.df[2][index]
        user_index = self.df[0][index] - 1
        item_index = self.df[1][index] - 1
        item_index_offset = NUM_USERS + item_index
        genre_osset = NUM_USERS + NUM_MOVIES
        genre_indices = [
            genre_osset + i
            for i, val in enumerate(self.genre_df.iloc[item_index].tolist())
            if val == 1
        ]
        inputs = [user_index, item_index_offset] + genre_indices

        sparse = torch.zeros([NUM_USERS + NUM_MOVIES + NUM_GENRES], dtype=torch.float)
        indices = torch.LongTensor([inputs])
        values = torch.ones_like(torch.tensor((len(inputs),)), dtype=torch.float)
        sparse[indices] = values

        return sparse, self.df[2][index]


def eval_model(model, train_dataloader):
    loss = 0
    for inputs, rating in train_dataloader:
        pred = model(inputs)
        loss += F.mse_loss(pred, rating) ** 0.5
    avg_loss = loss / len(train_dataloader)
    print(f"avg rmse: {avg_loss}")


def run_pipeline():
    training_data = MlDataset("data/ml-100k/u1.base")
    validation_data = MlDataset("data/ml-100k/u1.test")
    batch_size = 256
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, num_workers=10
    )
    validation_dataloader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, num_workers=10
    )
    # https://files.grouplens.org/datasets/movielens/ml-100k-README.txt
    n_factors = 30
    model = FactorizationMachine(
        field_dims=[NUM_USERS, NUM_MOVIES, NUM_GENRES], num_factors=n_factors
    )
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(model, train_dataloader, validation_dataloader)
    print("Train loss")
    eval_model(model, train_dataloader)
    print("Validation loss")
    eval_model(model, validation_dataloader)


if __name__ == "__main__":
    run_pipeline()
    # run_pipeline2()
