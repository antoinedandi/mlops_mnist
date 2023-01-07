import torch
import torch.nn.functional as F


def train_loop(model, optimizer, dataloader, device):

    model.train()
    running_loss, n_samples = 0, 0

    for data in dataloader:

        image = data["image"].to(device)
        label = data["label"].to(device)

        # compute loss
        pred = model(image)
        loss = F.cross_entropy(pred, label)

        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate losses
        running_loss += loss.item() * image.size(0)
        n_samples += image.size(0)

    epoch_loss = running_loss / n_samples
    return epoch_loss


def valid_loop(model, dataloader, device):

    model.eval()
    running_loss, running_metric, n_samples = 0, 0, 0

    with torch.no_grad():
        for data in dataloader:

            image = data["image"].to(device)
            label = data["label"].to(device)

            # compute loss
            pred = model(image)
            loss = F.cross_entropy(pred, label)

            # validation metric: classification accuracy
            correct_pred = (pred.argmax(-1) == label).sum()

            # accumulate losses
            running_loss += loss.item() * image.size(0)
            running_metric += correct_pred
            n_samples += image.size(0)

    epoch_loss = running_loss / n_samples
    epoch_metric = running_metric / n_samples
    return epoch_loss, epoch_metric
