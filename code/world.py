from torch import cuda


# ===============load data====================
mean = 98.54010552
std = 41.5086445
print(">MEAN:", mean)
print(">STD:", std)

# ===============cuda====================
useCuda = cuda.is_available()
if useCuda:
    print("USING CUDA")
else:
    print("USING CPU")