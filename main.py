import torch
import click

from common import train_loop, valid_loop
from data import MNIST
from model import Classifier


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--n_epochs", default=3, help="number of epochs to train")
def train(lr, n_epochs):

    # 1) create dataloader
    dataset = MNIST("train")
    indices = torch.randperm(len(dataset)).tolist()
    # train split
    train_dataset = torch.utils.data.Subset(dataset, indices[:-500])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    # valid split
    valid_dataset = torch.utils.data.Subset(dataset, indices[-500:])
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=64, shuffle=False
    )

    # 2) create model and optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = Classifier().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):

        train_loss = train_loop(model, optim, train_loader, device)
        valid_loss, valid_metric = valid_loop(model, valid_loader, device)

        log = (
            f"Epoch {epoch} "
            f"| Train loss: {train_loss:.3f} "
            f"| Valid loss: {valid_loss:.3f} "
            f"| Valid accuracy: {valid_metric:.3f}"
        )

        print(log)

    torch.save(model.state_dict(), "model.pt")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):

    # 1) create dataloader
    test_set = MNIST("test")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    # 2) load model
    model = Classifier()
    model.load_state_dict(torch.load(model_checkpoint, map_location="cpu"))

    test_loss, test_metric = valid_loop(model, test_loader, "cpu")
    log = f"| Test loss: {test_loss:.3f} | Test accuracy: {test_metric:.3f} |"

    print(log)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
