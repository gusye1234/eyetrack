import cv2 as cv
import numpy as np
import pandas as pd
from glob import glob
import os

folder = int(input("choose a data folder to generate demo(0-22)? \n>"))

folder = glob(os.path.join("../data", "*/%d"%folder))
if len(folder) == 0:
    raise TypeError("folder didn't exist")
else:
    folder = folder[0]

print(folder)

out = cv.VideoWriter( os.path.join(folder, "project.avi"), cv.VideoWriter_fourcc(*'DIVX'),15, (1680, 540))
cv.namedWindow("point", cv.WINDOW_AUTOSIZE)

count = 0
try:
    points = pd.read_csv(os.path.join(folder, "IMG1/Learn.csv"))
    points = points.to_numpy()
    pred = pd.read_csv(os.path.join(folder, "predict.csv"))
    pred = pred.to_numpy()[:, 1:3]
    pred *= 1000
    print(len(points))
    print(len(pred))
except:
    raise FileNotFoundError("please check if there has a predict.csv under %s \nuse eval.py to generate one" % folder)

for k in range(1, len(points)+1):
    ret, frame = [], []
    # for cap in cap_all:
    #     ret_one, frame_one = cap.read()
    #     ret.append(ret_one)
    #     frame.append(frame_one)
    for i in [str(j) for j in range(5)]:
        frame.append(cv.imread(os.path.join(folder, "IMG1/%d_%s.png" % (k, i) )))

    # if np.all(ret):
    if True:
        # print(count)
        # print(frame[0].shape)
        img = np.hstack([frame[0], frame[3]])
        img_bot = np.hstack([frame[1], frame[4]])
        img = np.vstack([img, img_bot])
        img = cv.resize(img, dsize=(720, 540))
        # cv.imshow("show",img)
        # print(frame[2].shape)
        board = cv.resize(frame[2], dsize=(960, 540))
        point = points[count].astype(np.int32)
        predict = pred[count].astype(np.int32)
        board[point[1]-5:point[1]+5, point[0]-5:point[0]+5] = 0
        board[predict[1]-5:predict[1]+5, predict[0]-5:predict[0]+5] = np.array([0,255,0])
        img = np.hstack([img, board])
        cv.imshow("point", img)
        out.write(img)
        count += 1
        if cv.waitKey(1) & 0xFF == ord("q") or count == len(points):
            break
print("total frame is", count)
out.release()
cv.destroyAllWindows()



