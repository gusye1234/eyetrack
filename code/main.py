from load_data import dataloader
from model import ITrackerModel, ItrackerImageModel
import utils
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch
from torch.utils.tensorboard import SummaryWriter
import world


if __name__ == "__main__":
    w = SummaryWriter(comment="experiment")
    tran = transforms.Compose([utils.Scale() ,utils.ToTensor()])

    data = dataloader(transform=tran)
    data = DataLoader(data, batch_size = 4, shuffle=True, num_workers=2)

    data_test = dataloader(mode="test", transform=tran)
    data_test = DataLoader(data_test, batch_size = 4, shuffle=True, num_workers=2)


    for i, train_data in enumerate(data_test):
        print(i, train_data["img0"].size(), train_data["img1"].size(), train_data["label"].size())
        if i > 2:
            break
    toy_input = torch.rand(1,1, 576, 720).requires_grad_()
    m = ITrackerModel()
    m2 = ItrackerImageModel()
    # w.add_graph(m2, toy_input)
    w.add_scalar("1/test", 1,1)
    w.add_image("image", make_grid( (train_data["img0"])*world.std + world.mean ), 1)
    w.close()