import torch
from model import ITrackerModel
from load_data import dataloader
from torch.utils.data import DataLoader
import time
from utils import AverageMeter
import world
import pandas as pd
import os
import numpy as np
import cv2 as cv
# cv.namedWindow("ok", cv.WINDOW_AUTOSIZE)
# cv.namedWindow("ok", cv.WINDOW_AUTOSIZE)

def eval(test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()
    predict = None
    for i, data in enumerate(test_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            output = model(data["img0"].float(), data["img1"].float(), data["img2"].float(), data["img3"].float())
        pred = output.cpu().numpy()
        pred = pd.DataFrame({"x":pred[:,0], "y":pred[:,1]})
        if predict is None:
            predict = pred
        else:
            predict = predict.append(pred)
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
    predict = predict[["x", "y"]]
    predict.to_csv( os.path.join(world.CHECKPOINTS_PATH, "predict.csv"))

def generate_heatmap(test_loader, model, criterion):
    model.eval()
    for i, data in enumerate(test_loader):
        for name in data:
            if "img" in name:
                data[name] = data[name].requires_grad_()
        output = model(data["img0"].float(), data["img1"].float(), data["img2"].float(), data["img3"].float())
        # loss = criterion(output, data["label"].float())
        # loss.backward()
        output.backward(torch.Tensor(np.ones((1,2))*0.5))
        eyes = []
        print(i)
        for name in data:
            if "img" in name:
                try:
                    img = data[name].cpu().detach().numpy().squeeze(axis=0).transpose(1,2,0)
                    # print(img.shape)
                    grad = data[name].grad.cpu().numpy().squeeze(axis=0).transpose(1,2,0)
                    grad = np.abs(grad)
                    kernal = np.ones((15,15))/225
                    grad = cv.filter2D(grad, -1, kernal)
                    grad[grad>np.percentile(grad, 60)] = grad[grad>np.percentile(grad, 60)]*1000
                    # grad[grad<1e-5] = 0
                    # print(np.max(grad), np.min(grad), np.median(grad))
                    # print(grad.shape)
                    img = img*world.std + world.mean
                    heat = heat_map(grad, img.astype("uint8"))
                    eyes.append(heat)
                except:
                    img = data[name].cpu().detach().numpy().squeeze(axis=0).transpose(1,2,0)
                    img = img*world.std + world.mean
                    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
                    eyes.append(img)
        if len(eyes) != 4:
            print("wrong")
            continue
        else:
            img = np.concatenate([eyes[0], eyes[2]], axis=1)
            img2 = np.concatenate([eyes[1], eyes[3]], axis=1)
            img = np.concatenate([img, img2], axis=0)
            img = cv.resize(img, (720,576))
            cv.imshow("ok", img)
            cv.waitKey(10)


def heat_map(grad, image):
    norm = np.zeros(grad.shape)
    cv.normalize(grad, norm, 0,255, cv.NORM_MINMAX)
    norm = np.asarray(norm, dtype=np.uint8)
    heat_img = cv.applyColorMap(norm, cv.COLORMAP_JET)  
    heat_img = cv.cvtColor(heat_img, cv.COLOR_BGR2RGB)
    img = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    cv.imwrite("see.png", heat_img)
    img_add = cv.addWeighted(img, 0.7, heat_img, 0.3, 0)
    return img_add

# norm_img = np.zeros(gray_img.shape)        
# cv2.normalize(gray_img , norm_img, 0, 255, cv2.NORM_MINMAX)
# norm_img = np.asarray(norm_img, dtype=np.uint8)
# heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
# heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)# 将BGR图像转为RGB图像
# img_add = cv2.addWeighted(org_img, 0.3, heat_img, 0.7, 0)

