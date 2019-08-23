## Explain the eye-tracking model



### developing environment

* macOS Mojave version 10.14
* Python 3.6.3
* pytorch1.2.0
* torchvision0.4.0a0+6b959ee

### progress

* 2019.8.22: complete `load_data.py` , return  `dict` type data. 

  including keys:

  * `img0`(left eye 0),`img1`(left eye 1), `im2`(right eye 3), `img3`(right eye 4) , image data format is `CHW`(channel first) ,shape`(-1, 1, 576, 720)`, type `torch.Tensor`
  * `label` , gaze point in image coordinate , type `torch.Tensor`
*  2019.8.23: add some image normalize tools  by [dearmrlv](https://github.com/dearmrlv)

* 2019.8.24: complete pipeline, and run it on CPU device

![program](./imgs/program.png)





### model

*  iTracker from [CSALI MIT](https://github.com/CSAILVision/GazeCapture)







### install

* `pip install requirements.txt`
* **prepare data**

under `root` 

```bash
mkdir data && cd data
mkdir train && mkdir test
```

copy data following the paths

```bash
data
├── test
│   ├── 01
│   ├── 02
│   └── etc
└── train
    ├── 01
    ├── 02
    └── etc
```

* `cd code && python main.py`
* follow the error traceback and see what's going on 😁



### TODO

* pytorch's data format problems--solved

```python
# File "main.py"
m = m.float()
# File "utils.py", Function 'train' and 'validate'
output = model(data["img0"].float(), data["img1"].float(), data["img2"].float(), data["img3"].float())
```

maybe there are some more elegant way to do so.

- running on GPU platform
- HOW TO EXPLAIN?





