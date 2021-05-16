import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from tabml.movielen import datasets

torch.multiprocessing.set_sharing_strategy("file_system")

NUM_USERS = 943
NUM_MOVIES = 1682
NUM_GENRES = 19


class FactorizationMachine(pl.LightningModule):
    def __init__(self, num_inputs, num_factors):
        super(FactorizationMachine, self).__init__()
        self.embedding = nn.Parameter(
            torch.randn(num_inputs, num_factors), requires_grad=True
        )
        self.linear_layer = nn.Linear(num_inputs, 1, bias=True)

    def forward(self, x):
        out_1 = torch.matmul(x, self.embedding).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.embedding.pow(2)).sum(1, keepdim=True)

        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.linear_layer(x)
        out = out_inter + out_lin

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


def eval_model(model, train_dataloader):
    loss = 0
    for inputs, rating in train_dataloader:
        pred = model(inputs)
        loss += F.mse_loss(pred, rating) ** 0.5
    avg_loss = loss / len(train_dataloader)
    print(f"avg rmse: {avg_loss}")


def run_pipeline():
    training_data, validation_data = datasets.get_ml_1m_dataset()
    batch_size = 256
    n_factors = 30
    num_workers = min(batch_size, 14)
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    validation_dataloader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    model = FactorizationMachine(
        num_inputs=training_data.input_dim, num_factors=n_factors
    )
    trainer = pl.Trainer(gpus=1, max_epochs=1)
    trainer.fit(model, train_dataloader)
    print("Validation loss")
    eval_model(model, validation_dataloader)


if __name__ == "__main__":
    run_pipeline()
