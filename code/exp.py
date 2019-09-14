import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os
from glob import glob

# folder = int(input("choose a data folder to generate demo(0-22)? \n>"))

# folder = glob(os.path.join("../data", "*/%d"%folder))
# if len(folder) == 0:
#     raise TypeError("folder didn't exist")
# else:
#     folder = folder[0]
folders = glob(os.path.join("../data", "test/*"))


if __name__ == "__main__":
    mark = ["o", "+", "^"]
    c = ["b", "r", "g"]
    reg = RandomForestRegressor(max_depth=2, random_state=0, 
                                    n_estimators=200)
    X = None
    Y = None
    for i, folder in enumerate(folders):
        # os.chdir(folder)
        
        x = np.load( os.path.join(folder, "middle.npy"))
        y = pd.read_csv(os.path.join(folder, "predict.csv"))[["x", "y"]].to_numpy()
        folder = os.path.basename(folder)
        print(folder)
        print(x.shape, y.shape)
        if X is None:
            X = x
            Y = y
        else:
            X = np.vstack([X, x])
            Y = np.vstack([Y, y])
    Y = Y*1000
    reg.fit(X,Y)
    pred = reg.predict(X)
    loss = np.mean(np.sqrt(np.sum((Y-pred)**2, axis=1)))
    
    importance = np.array(reg.feature_importances_)*10000
    
    most_value = np.argsort(importance)[-10:]
    print(most_value)
    print(importance[most_value])
    
    print("LOSS", loss)
    plt.scatter(range(x.shape[1]), reg.feature_importances_*1000, c=c[i], marker=mark[i], s=20, label=folder)
    
    plt.legend()
        # plt.title(folder)
    plt.show()
    
 