import torch
import world

class ToTensor:
    def __call__(self, sample):
        for i in sample:
            if "img" in i:
                sample[i] = torch.from_numpy(sample[i].transpose(2,0,1))
            else:
                sample[i] = torch.from_numpy(sample[i])
        if world.useCuda:
            for i in sample:
                sample[i] = sample[i].cuda()
        return sample


class Scale:
    def __init__(self, mean=None, std=None):
        self.mean = world.mean if mean is None else mean
        self.std = world.std if std is None else std
    def __call__(self, sample):
        for i in sample:
            if "img" in i:
                sample[i] = (sample[i] - self.mean)/self.std
        return sample