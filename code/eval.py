import torch
from model import ITrackerModel
from load_data import dataloader
from torch.utils.data import DataLoader
import time
from utils import AverageMeter
import world


def eval(test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()
    for i, data in enumerate(test_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            output = model(data["img0"].float(), data["img1"].float(), data["img2"].float(), data["img3"].float())
        print("sample:", output[0]*world.label_scalar, data["label"][0]*world.label_scalar)
        loss = criterion(output, data["label"].float())
        losses.update(loss.data.item(), data["label"].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            print('Test :[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(test_loader), batch_time=batch_time,
                loss=losses))