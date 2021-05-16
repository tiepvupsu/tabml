import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

torch.multiprocessing.set_sharing_strategy("file_system")


class MatrixFactorization(pl.LightningModule):
    def __init__(self, n_users, n_items, n_factors=40, dropout_p=0, sparse=False):
        """
        Parameters
        ----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        n_factors : int
            Number of latent factors (or embeddings or whatever you want to
            call it).
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        sparse : bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use
            weight decay on the optimizer if sparse=True. Also, can only use
            Adagrad.
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse)
        self.bias = nn.Parameter(torch.rand(1))
        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=sparse)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.sparse = sparse

    def forward(self, users, items):
        """
        Forward pass through the model. For a single user and item, this
        looks like:
        user_bias + item_bias + user_embeddings.dot(item_embeddings)
        Parameters
        ----------
        users : np.ndarray
            Array of user indices
        items : np.ndarray
            Array of item indices
        Returns
        -------
        preds : np.ndarray
            Predicted ratings.
        """
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users) + self.bias
        preds += self.item_biases(items)
        preds += torch.reshape(
            torch.diag(
                torch.matmul(
                    self.dropout(ues), torch.transpose(self.dropout(uis), 0, 1)
                )
            ),
            (-1, 1),
        )

        return torch.clip(preds.squeeze(), min=1, max=5)

    def training_step(self, batch, batch_idx):
        users, items, rating = batch
        rating = rating.to(torch.float32)
        output = self.forward(users, items)
        loss = F.mse_loss(rating, output)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.5, momentum=0.5, weight_decay=1e-3
        )  # learning rate
        return optimizer


class MF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=40, dropout_p=0, sparse=False):
        """
        Parameters
        ----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        n_factors : int
            Number of latent factors (or embeddings or whatever you want to
            call it).
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        sparse : bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use
            weight decay on the optimizer if sparse=True. Also, can only use
            Adagrad.
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse)
        self.bias = nn.Parameter(torch.rand(1))
        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=sparse)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.sparse = sparse

    def forward(self, users, items):
        """
        Forward pass through the model. For a single user and item, this
        looks like:
        user_bias + item_bias + user_embeddings.dot(item_embeddings)
        Parameters
        ----------
        users : np.ndarray
            Array of user indices
        items : np.ndarray
            Array of item indices
        Returns
        -------
        preds : np.ndarray
            Predicted ratings.
        """
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users) + self.bias
        preds += self.item_biases(items)
        preds += torch.reshape(
            torch.diag(
                torch.matmul(
                    self.dropout(ues), torch.transpose(self.dropout(uis), 0, 1)
                )
            ),
            (-1, 1),
        )

        return torch.clip(preds.squeeze(), min=1, max=5)


class MlDataset(Dataset):
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path, delimiter="\t", header=None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df[0][index] - 1, self.df[1][index] - 1, self.df[2][index]


def eval_model(model, train_dataloader):
    loss = 0
    for users, items, rating in train_dataloader:
        pred = model(users, items)
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
    n_users = 943
    n_movies = 1682
    n_factors = 30
    model = MatrixFactorization(n_users=n_users, n_items=n_movies, n_factors=n_factors)
    trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(model, train_dataloader, validation_dataloader)
    print("Train loss")
    eval_model(model, train_dataloader)
    print("Validation loss")
    eval_model(model, validation_dataloader)


if __name__ == "__main__":
    run_pipeline()
