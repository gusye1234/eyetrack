from load_data import dataloader
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from torch import cuda
import utils



if __name__ == "__main__":
    tran = transforms.Compose([utils.Scale() ,utils.ToTensor()])

    data = dataloader(transform=tran)
    data = DataLoader(data, batch_size = 4, shuffle=True, num_workers=2)

    data_test = dataloader(mode="test", transform=tran)
    data_test = DataLoader(data_test, batch_size = 4, shuffle=False, num_workers=2)


    for i, train_data in enumerate(data_test):
        print(i, train_data["img0"].size(), train_data["img1"].size(), train_data["label"].size())
        if i > 2:
            break
