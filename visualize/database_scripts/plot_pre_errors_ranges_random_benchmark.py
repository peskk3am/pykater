import sys
import os

from db_connect import *

import matplotlib.pyplot as plt
import numpy  

# ----- begin config ------

'''
    CONFIG:
    
        - print data to file "pre_random_benchmark_"+method+"_data.txt"
            print_data_table         True/False            
'''

print_data_table = True

# ----- end config ------



colors = ['#D7191C', '#2C7BB6', '#31a354']  # red, gree, blue

comparison = {} # {method: (number_of_better, number_of_worse)}
comparison_anneal = {} # {method: (number_of_better, number_of_worse)}


# read datasets_sorted_indexes from file
try:
    f = open("labels_indexes_pre.txt") 
    list_str = f.read()        
    datasets_sorted_indexes = [int(n) for n in list_str.strip("[]").split(", ")]    
    print(datasets_sorted_indexes)    
    f.close()
except:
    print("Datasets order could not be read from the file labels_indexes_pre.txt, run plot_pre_error_ranges_all_in_one.py first")
    exit()


datasets = [
    43,
    313,
    18,
    311,
    12,
    16,                
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
    print("sorting", labels_indexes)
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

searches = {"grid": 'Grid Search (scikit-learn)',
            "random": 'Randomized Search (scikit-learn)',
            "annealing": 'Simulated Annealing Search (scikit-learn)',
            "evolution": 'Evolutionary search (deap)'}

number_of_graphs = 0
def plot_one(method):
    global number_of_graphs
            
    # data ... sequence of vectors
    data_random = []
    for d in datasets:
        data_random.append(get_data_from_db(method, d, searches["random"]))

    data_evolution = []
    for d in datasets:
        data_evolution.append(get_data_from_db(method, d, searches["evolution"]))

    data_annealing = []
    for d in datasets:
        data_annealing.append(get_data_from_db(method, d, searches["annealing"]))

    
    data_random = my_sort(data_random)
    data_evolution = my_sort(data_evolution)
    data_annealing = my_sort(data_annealing)
    
    fig1, ax1 = plt.subplots(figsize=(3, 4.2))
        
    ax1.set_title(method)
    ax1.boxplot(data_random)    
    
            
    evolution_dots  = [min(values) if len(values)>0 else numpy.nan for values in data_evolution]
    annealing_dots  = [min(values) if len(values)>0 else numpy.nan for values in data_annealing]
    random_stats  = [(min(values), numpy.median(values)) if len(values)>0 else (numpy.nan, numpy.nan) for values in data_random ]
    

    if print_data_table:    
        f = open("pre_random_benchmark_"+method+"_data.txt", "w")
        i = 0
        for li in labels_indexes:        
            f.write(str(datasets[li])+"&"+str(round(evolution_dots[i],2))+"&"+str(round(random_stats[i][0],2))+"&"+str(round(random_stats[i][1],2))+"&"+str(round(random_stats[i][1]-evolution_dots[i],2))+"&"+str(round(random_stats[i][0]-evolution_dots[i],2))+"\n" )
            i += 1 
        f.close()
        
            
    for i in range(len(evolution_dots)):   # minimum benchmark
        print(method, random_stats[i][0]-annealing_dots[i]) 

        if random_stats[i][0]-annealing_dots[i] > 0:
            comparison_anneal[method][0] += 1
        else:
            comparison_anneal[method][1] -= 1
            
        if random_stats[i][0]-evolution_dots[i] > 0:
            comparison[method][0] += 1            
        else:
            comparison[method][1] -= 1            
    
    print(comparison[method])
    print(comparison_anneal[method])
            
    
    plt.plot(range(1,len(labels_indexes)+1), evolution_dots, color="xkcd:blood orange", linestyle='None', marker="s", markersize="4")
    
    plt.plot(range(1,len(labels_indexes)+1), annealing_dots, color="blue", linestyle='None', marker="o", fillstyle="none")
    
    ax1.set_ylabel('Error')

    # create labels
    labels = []
    simple_labels = []
    for i in labels_indexes:
        name, instances, attributes, classes = datasets_info.get(datasets[i], ("???", "???", "???", "???"))        
        l = str(datasets[i])+" "+name+" ("+str(instances)+", "+str(attributes)+", "+str(classes)+")" 
        simple_l = str(datasets[i])        
        labels.append(l)           
        simple_labels.append(simple_l)    
    
 
    # FULL LABELS:
    #plt.xticks(range(1, len(labels)+1), labels, rotation='vertical')
    
    # SIMPLE LABELS:
    plt.xticks(range(1, len(simple_labels)+1), simple_labels, rotation='vertical')
    
    number_of_graphs += 1
    
    
    plt.tight_layout()  
    
    plt.savefig('pre_random_benchmark_'+methods_short[method]+'.png')
    plt.savefig('pre_random_benchmark_'+methods_short[method]+'.pdf')

    plt.close(fig1)    
    

def set_box_color(bp, color):    
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)    
    plt.setp(bp["fliers"], markeredgecolor=color)
    
def plot(method_list, file_name, title):        
    # colors are from http://colorbrewer2.org/      
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


def get_data_from_db(method, dataset, search):
    
    if mydb:
        mycursor = mydb.cursor()                                          
        
        val = (dataset, methods_short[method], search)
        
        sql = "SELECT accuracy FROM results WHERE dataset=%s AND method=%s AND accuracy IS NOT NULL AND preprocessings!='[]' AND search=%s"
        
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
            
            sql = "SELECT accuracy FROM results WHERE dataset=%s AND method=%s AND accuracy IS NULL AND preprocessings!='[]' AND search=%s"
            
            mycursor.execute(sql, val)
            #print(mycursor.statement)
            
            myresult = mycursor.fetchall()
        
            errors = []
            
            for x in myresult:
                errors.append(1)
        
        if len(errors) == 0:
            print("ZERO LEN errors:", method, datasets.index(dataset)+1, "open_ml_id:", dataset, search)          
            
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


def plot_histogram():

    for m in method_list:
        print(comparison[m]) 
        print(comparison_anneal[m])
        print()        
        
    print("-------")    
        
    negative_data = []
    positive_data = []
    for m in method_list:        
        positive_data.append( comparison[m][0] )
        negative_data.append( comparison[m][1] )        

    negative_data_anneal = []
    positive_data_anneal = []
    for m in method_list:        
        positive_data_anneal.append( comparison_anneal[m][0] )
        negative_data_anneal.append( comparison_anneal[m][1] )        

    #positive_data = [2]*6+[20]*6
    #negative_data = [-4]*12
    
    #positive_data_anneal = [3]*6+[1]*6
    #negative_data_anneal = [-15]*12

                                                             
    print("evo", positive_data, sum(positive_data))
    print("evo", negative_data, sum(negative_data))

    print("an", positive_data_anneal, sum(positive_data_anneal))
    print("an", negative_data_anneal, sum(negative_data_anneal))


    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    x = [i-0.23 for i in range(len(method_list))]
    x1 = [i+0.23 for i in range(len(method_list))]
    
    ax.bar(x, positive_data, width=0.46, color="xkcd:blood orange") 
    ax.bar(x, negative_data, width=0.46, color="xkcd:sand brown")
        
    ax.bar(x1, positive_data_anneal, width=0.46, color="xkcd:clear blue")
    ax.bar(x1, negative_data_anneal, width=0.46, color="xkcd:greyish teal")
        
    for i, v in enumerate(positive_data):
        i = i - 0.23
        if v > 0:
            ax.text(i , v-1, str(v), ha='center', color="black")         
    
    for i, v in enumerate(negative_data):
        i = i - 0.23
        if v < 0:
            ax.text(i , v+0.5, str(-v), ha='center', color="black")         


    for i, v in enumerate(positive_data_anneal):
        i = i + 0.23
        if v > 0:
            ax.text(i , v-1, str(v), ha='center', color="black")         
    
    for i, v in enumerate(negative_data_anneal):
        i = i + 0.23
        if v < 0:
            ax.text(i , v+0.5, str(-v), ha='center', color="black")         

    plt.ylabel('Number of worse          Number of better')
    leg = [ 
            "Evolutionary Search - better", "Evolutionary Search - worse",
            "Simulated Annealing - better", "Simulated Annealing - worse"]
                                   
    ax.legend(leg, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
                                         
    plt.xticks(range(0, len(methods)), method_list, rotation='vertical')        
    
    plt.subplots_adjust(bottom=0.40)
    
    plt.savefig('pre_random_min_evolution_annealing.png')
    plt.savefig('pre_random_min_evolution_annealing.pdf')



def plot_all():
    for m in method_list:
        comparison[m] = [0,0]
        comparison_anneal[m] = [0,0]
        plot_one(m)
          

                                
datasets_info = get_dataset_names_from_db()    
plot_all()

plot_histogram()

