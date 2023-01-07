import numpy as np
import torch
import os


class MNIST(torch.utils.data.Dataset):
    def __init__(
        self,
        partition="train",
        data_dir="/Users/antoinedandigne/PycharmProjects/mlops_mnist/data/corruptmnist",
    ):
        # Load data
        data = {
            k.replace(".npz", ""): np.load(data_dir + "/" + k)
            for k in os.listdir(data_dir)
            if partition in k
        }
        self.images = np.concatenate([data[k]["images"] for k in data.keys()])
        self.labels = np.concatenate([data[k]["labels"] for k in data.keys()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "image": torch.tensor(self.images[i][None], dtype=torch.float32),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }


if __name__ == "__main__":
    train_dataset = MNIST("train")
    test_dataset = MNIST("test")
    print(len(train_dataset), len(test_dataset))
    print(train_dataset[0]["image"].shape, train_dataset[0]["label"])
