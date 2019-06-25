import sys
import os

from db_connect import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import numpy  


def plot_random_forest():
    #method = "RandomForestClassifier"
    method = "random_forest"    
    
    errors_params = get_data_from_db(method) # [(error, {param_name: value})]
    # error -> color
    # x -> random_forest__n_estimators
    # y -> random_forest__min_samples_leaf
    # z -> random_forest__max_depth

    x, y, z, err = [], [], [], []

    n = 0            
    for error, params in errors_params:
      # if n % 5 == 0:         
        if params["random_forest__n_estimators"] < 100 and params["random_forest__max_depth"] < 100:
            x += [params["random_forest__n_estimators"]]
            y += [params["random_forest__min_samples_leaf"]]
            z += [params["random_forest__max_depth"]]            
                    
            err += [error]
            n += 1    
    print(n)        
    x, y, z, err = numpy.array(x), numpy.array(y), numpy.array(z), numpy.array(err)        

            
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
                        
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
                
    scatter = ax.scatter(x, y, z, c=err, cmap='Oranges_r', marker=".", s=2)
            
    ax.set_xlabel("random_forest__n_estimators")
    
    ax.set_zlabel("\n random_forest__max_depth", linespacing=3.1)  

    ax.set_ylabel("\n random_forest__min_samples_leaf", linespacing=3.4)
    
    
    # Now adding the colorbar
    #  [left, bottom, width, height]
    cbaxes = fig.add_axes([0.05, 0.1, 0.025, 0.7]) 
    cb = plt.colorbar(scatter, cax = cbaxes)     
    cb.set_label('Error')
            
                         
    plt.savefig(method+'_params.png')
    plt.savefig(method+'_params.pdf')
    plt.close(fig)    



def plot_gradient_boosting():
    #method = "GradientBoostingClassifier"
    method = "gradient_boosting"    
    
    errors_params = get_data_from_db(method) # [(error, {param_name: value})]
    # error -> color
    # x -> gradient_boosting__n_estimators
    # y -> gradient_boosting__max_features
    # z -> gradient_boosting__max_depth

    x, y, z, err = [], [], [], []

    n = 0            
    for error, params in errors_params:
      # if n % 5 == 0:         
        if params["gradient_boosting__n_estimators"] < 500 and params["gradient_boosting__max_depth"] <= 10:
            x += [params["gradient_boosting__n_estimators"]]
            y += [params["gradient_boosting__max_features"]]
            z += [params["gradient_boosting__max_depth"]]            
                    
            err += [error]
            n += 1    
    print(n)        
    x, y, z, err = numpy.array(x), numpy.array(y), numpy.array(z), numpy.array(err)        

            
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
                        
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
                
    scatter = ax.scatter(x, y, z, c=err, cmap='Oranges_r', marker=".", s=2)
            
    ax.set_xlabel("gradient_boosting__n_estimators")
    
    ax.set_zlabel("\n gradient_boosting__max_depth", linespacing=3.1)  

    ax.set_ylabel("\n gradient_boosting__max_features", linespacing=3.4)
    
    
    # Now adding the colorbar
    #  [left, bottom, width, height]
    cbaxes = fig.add_axes([0.05, 0.1, 0.025, 0.7]) 
    cb = plt.colorbar(scatter, cax = cbaxes)     
    cb.set_label('Error')
            
                         
    plt.savefig(method+'_params.png')
    plt.savefig(method+'_params.pdf')
    plt.close(fig)    



def plot_knn():
    # method = "KNeighborsClassifier"
    method = "knn"
    
    knn__algorithm = ["ball_tree", "kd_tree", "brute"]
    knn__weights = ["uniform", "distance"]

            
    errors_params = get_data_from_db(method) # [(error, {param_name: value})]
    # error -> color
    # x -> knn__n_neighbors
    # y -> knn__weights
    # z -> knn__algorithm    
    
    x, y, z, err = [], [], [], []
            
    n = 0
    k = 0
    for error, params in errors_params:  
      if k % 3 == 0:    
        if params["knn__n_neighbors"] < 100 :
            if "knn__algorithm" not in params or "knn__weights" not in params or "knn__n_neighbors" not in params:
                print(params)            
            else:    
                x += [params["knn__n_neighbors"]]              
                _y = knn__algorithm.index(params["knn__algorithm"])+(random.random()/6)     
                y += [_y]
                _z = knn__weights.index(params["knn__weights"])+(random.random()/6)                                   
                z += [_z]            
                
                err += [error]
        n += 1
      k += 1
    print(n)        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
            
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
                
    scatter = ax.scatter(x, y, z, c=err, cmap='Oranges_r', marker=".", s=2)
    
    ax.set_xlabel("knn__n_neighbors")
    
    ax.set_zticks([0,1])
    ax.set_zticklabels(knn__weights)    
    ax.set_zlabel("\nknn__weights", linespacing=3.1)

    ax.set_yticks([0,1,2])
    ax.set_yticklabels(knn__algorithm)    
    ax.set_ylabel("\nknn__algorithm", linespacing=3.4)
    
    ax.dist = 10
    
    
    # Now adding the colorbar
    #  [left, bottom, width, height]
    cbaxes = fig.add_axes([0.05, 0.1, 0.025, 0.7]) 
    cb = plt.colorbar(scatter, cax = cbaxes)     
    cb.set_label('Error')
    
                 
    plt.savefig(method+'_params.png')
    plt.savefig(method+'_params.pdf')
    plt.close(fig)    
    
    

def get_data_from_db(method):
    
    if mydb:
        mycursor = mydb.cursor()                                          
        
        val = (method,)
        
        sql = "SELECT accuracy, parameters FROM results WHERE method=%s AND accuracy IS NOT NULL AND preprocessings='[]' " 
        mycursor.execute(sql, val)
        
        myresult = mycursor.fetchall()
        
        errors_params = []  # [(error, {param_name: value})]
        try:
            for x in myresult:        
                params = eval(x[1])    
                errors_params.append( (1-float(x[0]), params) )                                                
        except:
            pass
                
        return errors_params    

plot_gradient_boosting()
plot_knn()
plot_random_forest()