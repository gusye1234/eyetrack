
import torch
import world
import pandas as pd
import numpy as np
from model import experiment
from utils import show_config

def getFcModel(activation="sigmoid"):
    world.activation = activation
    show_config()

    model = experiment()

    weight = torch.load("./checkpoints/best_see_720_sigmoid_132.pth.tar", map_location=torch.device("cpu"))
    weight = weight["state_dict"]
    weight = {i:weight[i] for i in weight if "fc" in i}
    print(weight.keys())

    print(model.load_state_dict(weight))
    print(model.eval())

    folder = "../data/test/3"
    middle = np.load(folder+"/middle.npy")
    points = pd.read_csv(folder+"/predict.csv")[["x", "y"]].to_numpy()

    print(middle.shape, points.shape)
    middle = torch.from_numpy(middle)

    out = model(middle[:10])
    # print(out)
    # print(points[:10])
    return model, middle, points

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    best = [ 38, 37, 36, 35, 34, 45, 127,91,19,82]
    model, middle, points = getFcModel()
    model.eval()
    out_true = model(middle[:1]).detach().numpy()
    test = middle[0:1].numpy()
    test_bed = np.zeros((100,128))
    test_bed[:] = test
    assert (test_bed[0] == test_bed[1]).all()
    x = np.arange(100)*0.01
    # plt.ion()
    dim = 0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    while True:
        # dim = int(input(">\n"))
        if dim >= 128:
            break
        this_test = np.copy(test_bed)
        this_test[:, dim] = x
        # this_test[0] = test
        # print(this_test.shape)
        this_test = torch.from_numpy(this_test).float()
        out = model(this_test)
        out = out.detach().numpy()
        if dim in best:
            # ax.scatter(out[1:,0], out[1:, 1],s=20, label=str(dim))    
            pass
        else:
            ax.scatter(out[1:,0], out[1:, 1],s=1, c="darkseagreen", alpha=0.5)
        dim += 1
        # plt.show()
    for dim in best:
        this_test = np.copy(test_bed)
        this_test[:, dim] = x
        # this_test[0] = test
        # print(this_test.shape)
        this_test = torch.from_numpy(this_test).float()
        out = model(this_test)
        out = out.detach().numpy()
        ax.scatter(out[1:,0], out[1:, 1],s=10, label=str(dim))    
        
    ax.scatter(out_true[0,0], out_true[0,1], s=80, marker="+",c="k", label="original prediction")
    # majorLocator   = MultipleLocator(20)
    # majorFormatter = FormatStrFormatter('%d')
    # minorLocator   = MultipleLocator(5)
    # major_ticks = np.arange(0, 1, 50)
    # minor_ticks = np.arange(0, 1, 20)

    # ax.set_xticks(major_ticks)
    # ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)
    # ax.grid(which="major")
    ax.set_facecolor("linen")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("the changing range of the human 2D fix point when one of 128 intermediate variables changes")
    plt.legend()
    plt.show()
    