# Illumination Tools
* func: **executeHE(img_name, filename)**
   * 直方图均衡化，用于图像增强，处理太亮或者太暗的图像
   * :type img_name: str, the file name of the single channel image
   * :type filename: str, the name for saving
* class: **Histgram**
   * 用于分析单幅图像的灰度直方图
   * :type img: str, the file name of a single channel image located at corrent directory
   * [参考](https://docs.opencv.org/trunk/d5/daf/tutorial_py_histogram_equalization.html)
   * method: **OriginalDistribution()**
      * show the original distribution of the image
* func: **executeGC(img_name, filename, gamma)**
   * gamma校正，用于图像的非线性变换，大于一就变亮
   * :type img_name: str, the file name of the single channel image
   * :type filename: str, the name for saving
   * :type gamma: float
* func: **executeGIC(im_converted_name, im_canonical_name, save_name)**
   * [参考](https://ieeexplore.ieee.org/document/1240838)
   * 输入需要一份标准图像，用于将其他图像的亮度转换到标准图像的水平
   * **运行速度较慢**
   * :type im_converted_name: str, name of single channel image
   * :type im_canonical_name: str, name of single channel image
* func: **executeQIR(imGroup_0_loc, imGroup_1_loc, group_loc, new_im_name)**
   * :type imGroup_0_loc: str, the directory of the canonical group
   * :type imGroup_1_loc: str,the directory of another group
   * :type group_loc: str, the directory for save
   * :type new_im_name: str, the pics will be named as new_im_name + str(index)
   * :rtype: None
   * [参考](https://ieeexplore.ieee.org/document/1240838)
   * 这个方法是通过比较一系列物体在两组不同光照条件下获得的照片，其中一组为标准组A，
    这个函数的作用是模拟标准组A的光照条件去relight另一组B，从而获得组B物体在标准组A光照条件下的照片。
    需要指定一个标准组。
    两组内所有图片的shape都应该相同,图片数目应该相同，系列中的每个物体在两组中应该一一对应，即序号相同。
   * *如果我们的数据里面有不同光照条件下的多组数据，这个方法可能有用*
