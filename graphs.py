import os
files = os.listdir(".")

import matplotlib.pyplot as plt

def visualize(res, dataset, estimator):
    x = range(0, len(res)) 
    fig = plt.figure()
    plt.plot(x, res, 'r.', markersize=12)
    plt.title('Grid search - ' + estimator +', ' + dataset)
    plt.show()


estimator = "RandomForestClassifier"

f = open("res_grid_6.txt")
lines = f.readlines()

res = []
for line in lines:
        res += [float(line.strip())]                

visualize(res, "_54", estimator)
#print(res.minimum)
#print(res.maximum)
#print(res.avg)
#print(res.std)
print()
print()
