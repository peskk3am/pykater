import sys
import os

from db_connect import *

import matplotlib.pyplot as plt
import numpy  


path_default_folder = ".."+os.sep+"results_default"+os.sep



# read datasets_sorted_indexes from file
try:
    f = open("labels_indexes.txt") 
    list_str = f.read()        
    datasets_sorted_indexes = [int(n) for n in list_str.strip("[]").split(", ")]    
    print(datasets_sorted_indexes)    
    f.close()
except:
    print("Datasets order could not be read from the file labels_indexes.txt")
    exit()


datasets = [
    12,
    14,
    16,
    18,
    20,
    54,
    60,
    36,
    46,
    22,
    23,
    181,
    183,
    311,
    313,
    307,
    1038,
    40670,
    1041,    
    53,
    62,
    39,
    40,
    41,
    43,
    1104,
    212,
    61
]


labels_indexes = []
datasets_info = {}


def my_sort(data):
    # sort by datasets_sorted_indexes (same as all graph)
    global labels_indexes
    labels_indexes = []
        
    data_new = []
    for c in datasets_sorted_indexes:
        # print(c)
        data_new.append(data[c])
        labels_indexes.append(c)
    
    return data_new      


method_list = [ "BernoulliNB",
                "KNeighborsClassifier",
                "DecisionTreeClassifier",
                "SGDClassifier",
                "PassiveAggressiveClassifier",
                "LinearDiscriminantAnalysis",
                "QuadraticDiscriminantAnalysis",
                "LinearSVC",
                "AdaBoostClassifier",
                "GradientBoostingClassifier",
                "RandomForestClassifier"
]

methods_short = {"BernoulliNB": "bernoulli_nb",
                "KNeighborsClassifier": "knn",
                "DecisionTreeClassifier": "decision_tree",
                "SGDClassifier": "sgd",
                "PassiveAggressiveClassifier": "passive_aggressive",
                "LinearDiscriminantAnalysis": "lda",
                "QuadraticDiscriminantAnalysis": "qda",
                "LinearSVC": "linear_svc",
                "AdaBoostClassifier": "adaboost",
                "GradientBoostingClassifier": "gradient_boosting",
                "RandomForestClassifier": "random_forest"
		}


methods = {
    "AdaBoostClassifier": 10,
    "BernoulliNB": 1,      
    "DecisionTreeClassifier": 3,
    "GradientBoostingClassifier": 5,
    "KNeighborsClassifier": 2,
    "LinearDiscriminantAnalysis": 4,
    "LinearSVC": 7,               
    "PassiveAggressiveClassifier": 12,
    "QuadraticDiscriminantAnalysis": 6,   
    "RandomForestClassifier": 11, 
    "SGDClassifier": 9, 
    "SVC": 8
    }

number_of_graphs = 0
def plot_one(method):
    global number_of_graphs
            
    # data ... sequence of vectors
    data = []
    for d in datasets:
        data.append(get_data_from_db(method, d))

    data = my_sort(data)
    
    fig1, ax1 = plt.subplots(figsize=(12, 7))    
    ax1.set_title(method)
    ax1.boxplot(data)    

    # plot default values
    dots = []
    for li in labels_indexes:
        dots += [default.get((method, datasets[li]), None)] 
    
    
    plt.plot(range(1,len(labels_indexes)+1), dots, "bs")        
    
    ax1.set_ylabel('Error')

    # create labels
    labels = []
    for i in labels_indexes:
        name, instances, attributes, classes = datasets_info.get(datasets[i], ("???", "???", "???", "???"))        
        l = str(datasets[i])+" "+name+" ("+str(instances)+", "+str(attributes)+", "+str(classes)+")" 
        labels.append(l)               
        
    
    plt.xticks(range(1, len(labels)+1), labels, rotation='vertical')
    
    number_of_graphs += 1
    
    
    plt.subplots_adjust(bottom=0.40)
                 
    plt.savefig(method+'.png')
    plt.savefig(method+'.pdf')
    plt.close(fig1)    
    

def set_box_color(bp, color):    
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)    
    plt.setp(bp["fliers"], markeredgecolor=color)
    
def plot(method_list, file_name, title):        
    # colors are from http://colorbrewer2.org/
    colors = ['#D7191C', '#2C7BB6', '#31a354']  
    fig1, ax1 = plt.subplots(figsize=(25, 15))
    ax1.set_title(title)
    ax1.set_ylabel('Error')
                
    i = -1            
    for method in method_list:
        # data ... sequence of vectors            
        data = []
        for d in datasets:
            data.append(get_data_from_db(method, d))
    
        data = my_sort(data)
        
        bp = ax1.boxplot(data, positions=numpy.array(range(len(data)))*3.0+(0.5*i)+1)
        set_box_color(bp, colors[i])
        i += 1        

    # create labels
    labels = []
    for i in labels_indexes:
        name, instances, attributes, classes = datasets_info.get(datasets[i], ("???", "???", "???", "???"))
        l = str(datasets[i])+" "+name+" ("+str(instances)+", "+str(attributes)+", "+str(classes)+")" 
        labels.append(l)               
    
    plt.xticks(range(1, len(labels)*3+1, 3), labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.35)
                 
    plt.savefig(file_name+'.png')
    plt.savefig(file_name+'.pdf')
    plt.close(fig1)    


def get_data_from_db(method, dataset):

    search = 'Randomized Search (scikit-learn)'
    
    if mydb:
        mycursor = mydb.cursor()                                          
        
        val = (dataset, methods_short[method], search)
        
        sql = "SELECT accuracy FROM results WHERE dataset=%s AND method=%s AND accuracy IS NOT NULL AND preprocessings='[]' AND search=%s"
        
        mycursor.execute(sql, val)
        
        myresult = mycursor.fetchall()
        
        errors = []
        try:
            for x in myresult:
                errors.append( 1-float(x[0]) )                
        except:
            pass
                
        if len(errors) == 0:   # add NULL as error 1, only if no other values are present (so that it would not spoil the avg)

            mycursor = mydb.cursor()                                          
        
            val = (dataset, methods_short[method], search)
            
            sql = "SELECT accuracy FROM results WHERE dataset=%s AND method=%s AND accuracy IS NULL AND preprocessings='[]' AND search=%s"
            
            mycursor.execute(sql, val)
        
            myresult = mycursor.fetchall()
        
            errors = []
            
            for x in myresult:
                errors.append(1)
        
        if len(errors) == 0:
            print("EMPTY:", method, datasets.index(dataset)+1, "open_ml_id:", dataset)          
            
        return errors    


def get_dataset_names_from_db():
    
    if mydb:
        mycursor = mydb.cursor()                                          
                
        
        sql = "SELECT openml_id, name, instances, attributes, classes FROM datasets"
        mycursor.execute(sql)
        
        myresult = mycursor.fetchall()
        
        datasets_info = {}  # [openml_id: (name, instances, attributes, classes)]
        try:
            for x in myresult:
                datasets_info[x[0]] = x[1:]
        except:
            pass
                
        return datasets_info    


def plot_all():
    for m in method_list:
        plot_one(m)


# read default data
default = {}  # (method, openml_id): error
for method in method_list:
    try:
        f = open(path_default_folder+method+".def.res")
        for line in f.readlines():            
            openml_id, accuracy, std, params = line.split(" ", maxsplit=3)
            default[(method, int(openml_id))] = 1-float(accuracy) 
        f.close()
    except:
        continue

# print(default)    

datasets_info = get_dataset_names_from_db()    
plot_all()

