import world
from glob import glob
import os
import cv2 as cv
import pandas as pd


if __name__ == "__main__":
    folders = glob(os.path.join("/data/gusye/new/DATA", "*/*/*/Learn.csv"))
    for folder in folders:
        dirname = os.path.dirname(folder)
        print("HANDLE", dirname)
        length = len(pd.read_csv(folder).to_numpy())
        for i in range(1, length+1):
            for suffix in ["0", "1", "3", "4"]:
                # print(os.path.join(dirname, str(i)+"_" + suffix + ".png"))
                a = cv.imread(os.path.join(dirname, str(i)+"_" + suffix + ".png"))[...,0]
                a = cv.resize(a, dsize=(256,256))
                cv.imwrite(os.path.join(dirname, str(i)+"_" + suffix + "_256.png"), a)
            print("done", os.path.join(dirname, str(i)))
                