from torch import cuda




string = r"""
  ___  __  __  ___
 / _ \ \ \/ / / _ \
|  __/  \  / |  __/
 \___|  / /   \___|
       /_/                         
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

label_scalar = 1000
scalar = [800, 600]
useSigmoid = True

verbose = False
CHECKPOINTS_PATH = "./checkpoints"
filename = "checkpoint.pth.tar"
weight_file = ""
batch_size = 0
doLoad = True
n_batch = 0
comment = "test"
tensorboard = False
momentum = 0.9
epochs = 240
weight_decay = 1e-4
workers = 0
base_lr = 0.0001
resize=False
collect=False
activation="none"

if verbose:
    print("world loading complete")