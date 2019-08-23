from torch import cuda




string = r"""
  _   _        _              __    __          
 | | | |  ___ | | _ __       /  \  /  \     ___
 | |_| | / _ \| || '_ \     /    \/    \   / _ \
 |  _  ||  __/| || |_) |   /    _____   \ |  __/
 |_| |_| \___||_|| .__/   /__ /      \ __\ \___|
                 |_|                                                                      
"""

print(string)



# ===============load data====================
mean = 98.54010552
std = 41.5086445

# ===============cuda====================
useCuda = cuda.is_available()
if useCuda:
    print(">USING CUDA")
else:
    print(">USING CPU")


CHECKPOINTS_PATH = "./checkpoints"
filename = "checkpoint.pth.tar"
batch_size = 0
doLoad = True
n_batch = 0
comment = "test"
tensorboard = False

momentum = 0.9
lr = 0.0001
epochs = 100
weight_decay = 1e-4
workers = 4
base_lr = 0.0001