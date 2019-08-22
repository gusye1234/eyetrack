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

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image

# Histgram Equalization (HE)

def executeHE(img_name, filename):
    """
    :type img_name: str, the file name of the single channel image
    :type filename: str, the name for saving
    """
    img = cv.imread(img_name,0)
    equ = cv.equalizeHist(img)
    # res = np.hstack((img,equ)) #stacking images side-by-side
    cv.imwrite(filename,equ)

class Histgram:
    """
    :type img: the file name of a single channel image located at corrent directory
    :reference: https://docs.opencv.org/trunk/d5/daf/tutorial_py_histogram_equalization.html
    """
    def __init__(self, image, channel):
        """
        :type channel: int, the color channel of the image
        """
        if channel == 1:
            img = cv.imread('wiki.jpg')
        else:
            img = cv.imread('wiki.jpg',0)

        self.hist,self.bins = np.histogram(self.img.flatten(),256,[0,256])
        
        # cdf: Cumulative Distribution Function
        self.cdf = self.hist.cumsum()
        self.cdf_normalized = self.cdf * float(self.hist.max()) / self.cdf.max()
    
    def OriginalDistribution(self):
        """
        show the original distribution of the image
        """
        plt.plot(self.cdf_normalized, color = 'b')
        plt.hist(self.img.flatten(),256,[0,256], color = 'r')
        plt.xlim([0,256])
        plt.legend(('cdf','histogram'), loc = 'upper left')
        plt.show()
    

############################################################

# Gamma Correction (GC)
def executeGC(img_name, filename, gamma):
    """
    :type img_name: str, the file name of the single channel image
    :type filename: str, the name for saving
    :type gamma: float
    """
    img = cv.imread(img_name, 0)
    img = img.astype(np.float)
    r,c = img.shape
    for i in range(r):
        for j in range(c):
            img[i, j] = 255 * (img[i, j]/255.0)**(1/gamma)
    img = img.astype(np.int16)
    img = Image.fromarray(img)
    # img.show()
    img = img.convert('L')
    img.save(filename)


# Gamma Intensity Correction (GIC)
def executeGIC(im_converted_name, im_canonical_name, save_name):
    """
    :type img: single channel image
    :rtype: single channel image
    :resource: https://ieeexplore.ieee.org/document/1240838
    :function: to make the brightness of the converted image similar to the canonical image
    :attention: im_converted and im_canonical should have a same shape
    """
    ## the range of gamma: [0.1, 7.9]

    im_canonical = cv.imread(im_canonical_name, 0)
    im_converted = cv.imread(im_converted_name, 0)

    im_canonical = im_canonical.astype(np.float)
    im_converted = im_converted.astype(np.float)

    sum = 0
    sum_array = []
    r, c = im_canonical.shape
    for g in range(80):
        if g == 0:
            continue
        gamma = 0.1 * g
        for i in range(r):
            for j in range(c):
                sum = sum + (255 * (im_converted[i, j]/255.0)**(1/gamma)
                             -im_canonical[i, j])**2

        sum_array.append(sum)
        sum = 0
        # if g % 10 == 0:
        # print(str(g) + ' values have been tried!')

    # print(sum_array.index(min(sum_array)))
    # plt.plot(sum_array)
    # plt.show()
    im_converted = executeGC(im_converted_name, save_name, sum_array.index(min(sum_array))*0.1)
    
 # Quotient Illumination Relighting (QIR)
import glob
def executeQIR(imGroup_0_loc, imGroup_1_loc, group_loc, new_im_name):
    """
    :type imGroup_0_loc: str, the directory of the canonical group
    :type imGroup_1_loc: str,the directory of another group
    :type group_loc: str, the directory for saveD
    :type new_im_name: str, the pics will be named as new_im_name + str(index)
    :rtype: None
    :reference: https://ieeexplore.ieee.org/document/1240838
    """
    # 这个方法是通过比较一系列物体在两组不同光照条件下获得的照片，其中一组为标准组A，
    # 这个函数的作用是模拟标准组A的光照条件去relight另一组B，从而获得组B物体在标准组A光照条件下的照片
    # 需要指定一个标准组
    # 两组内所有图片的shape都应该相同,图片数目应该相同，系列中的每个物体在两组中应该一一对应，即序号相同
    # 如果我们的数据里面有不同光照条件下的多组数据，这个方法可能有用

    imGroup_0 = glob.glob(imGroup_0_loc)    # type: array
    imGroup_1 = glob.glob(imGroup_1_loc)    # type: array

    im_num = len(imGroup_0)
    R_mat = cv.imread(imGroup_0[0], 0)
    R_mat = np.zeros_like(R_mat, dtype=float)
    tmp_mat = np.zeros_like(R_mat, dtype=int)

    r, c = R_mat.shape
    im_1_dict = {}

    for t in range(im_num):
        im_0 = cv.imread(imGroup_0[t], 0)
        im_1_dict[t] = cv.imread(imGroup_1[t], 0)
        for i in range(r):
            for j in range(c):
                if im_0[i, j] != 0:
                    R_mat[i, j] = R_mat[i, j] + 1/im_num * im_1_dict[t][i, j] / im_0[i, j]

    for t in range(im_num):
        for i in range(r):
            for j in range(c):
                if R_mat[i, j] != 0:
                    tmp_mat[i, j] = int(im_1_dict[t][i, j] / R_mat[i, j])
                else:
                    tmp_mat[i, j] = int(im_1_dict[t][i, j])

        img = tmp_mat.astype(np.int16)
        img = Image.fromarray(img)
        # img.show()
        img = img.convert('L')
        img.save(group_loc + new_im_name + str(t))
        
        

