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
* func: **GIC(im_converted_name, im_canonical_name, save_name)**
   * [参考](https://ieeexplore.ieee.org/document/1240838)
   * 输入需要一份标准图像，用于将其他图像的亮度转换到标准图像的水平
   * **运行速度较慢**
   * :type im_converted_name: str, name of single channel image
   * :type im_canonical_name: str, name of single channel image
