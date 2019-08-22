from load_data import dataloader
from torch.utils.data import Dataset, DataLoader



data = dataloader()
data = DataLoader(data, batch_size = 4, shuffle=False, num_workers=2)

data_test = dataloader(mode="test")
data_test = DataLoader(data_test, batch_size = 4, shuffle=False, num_workers=2)


for i, train_data in enumerate(data_test):
    print(i, train_data["img0"].size(), train_data["img1"].size(), train_data["label"].size())
    if i > 2:
        break