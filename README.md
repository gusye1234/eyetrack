## Explain the eye-tracking model



### progress

* 8.22 完成 data_loader, 返回数据为`dict` 类型. 

  其中 key 包括:

  * `img0`(左眼0),`img1`(左眼1), `im3`(右眼3), `img4`(右眼4) , 皆为 `CHW`(channel first)的图片数据,shape`(-1, 1, 576, 720)` ,类型为`torch.Tensor`
  * `label` , 表示的是 **凝视点** 在图像坐标系下的坐标, 类型为`torch.Tensor`

### model

1. ,目前打算使用MIT 的 iTracker 模型







### data

在 eyetrack 目录下创建 data 目录

```bash
mkdir data && cd data
mkdir train && mkdir test
```

按照格式将数据复制

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

