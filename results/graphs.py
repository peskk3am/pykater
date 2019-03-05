import sys
# import os
# files = os.listdir(".")

import matplotlib.pyplot as plt

def visualize(res, search, estimator, dataset):
    x = range(0, len(res)) 
    fig = plt.figure()
    plt.plot(x, res, 'r.', markersize=12)
    plt.title(search +' - ' + estimator +', dataset:' + dataset)
    plt.show()


def parse_file_name(file_name):
    # EvolutionarySearchCV_KNeighborsClassifier_61_0.res
    search, estimator, dataset, the_rest = file_name.split("_")
    experiment_number, extenstion = the_rest.split(".")
    return search, estimator, dataset, experiment_number


if sys.argv[1] == "?":
    print("Usage: graphs.py res_file_name"))
else:
    # argv 1 is the .res file name
    file_name = sys.argv[1]
    search, estimator, dataset, experiment_number = parse_file_name(file_name)    

    f = open(file_name)
    lines = f.readlines()

    res = []
    for line in lines:
        if line[0] != "#":
            try:
                r = float(line.split(" ")[0])
                res += [r]
            except: 
                pass
                                
    print("Min:", min(res))
    print("Max:", max(res))
    print("Avg:", sum(res)/float(len(res)))

    visualize(res, search, estimator, dataset)
    
    #print()
    #print()
