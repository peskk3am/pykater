import openml_api_key

import openml
import  os

import pandas as pd
import numpy as np

import database
import _config

def init():
    openml.config.apikey = openml_api_key.key

    # we also want to use the test server so as not to polute the production system
    # openml.config.server = "http://test.openml.org/api/v1/xml"
    openml.config.server = "http://www.openml.org/api/v1/xml"

    # datasets = openml.datasets.list_datasets()
    # most_likes = [40536, 61, 37, 1142, 1039, 1464, 1130, 1116, 1128, 54, 59, 186, 294, 28, 4538, 1471, 40477, 9, 1596, 1489, 43, 118, 310, 40478, 372, 23380, 182, 4541, 13, 44, 1113, 1462, 1397, 40, 21, 1176, 1398, 1514, 4532, 8, 38100, 38105, 38112, 38117, 38124, 38129, 38131, 38136, 38143, 38148, 38150, 38155, 38162, 38167, 38174, 38179, 38181, 38186, 38193, 38198, 38201, 38206, 38213, 38218, 38220, 38225, 38232, 38237, 38244, 38249, 38251, 38256, 38263, 38268, 38270, 38275, 38282, 38287, 38294, 38299, 38302, 38307, 38314, 38319, 38321, 38326, 38333, 38338, 38340, 38345, 38352, 38357, 38364, 38369, 38371, 38376 ]
    # most_likes = [61, 37, 1142, 1039, 1464, 1130, 1116, 1128, 54, 59, 186, 294, 28, 4538, 1471, 40477, 9, 1596, 1489, 43, 118, 310, 40478, 372, 23380, 182, 4541, 13, 44, 1113, 1462, 1397, 40, 21, 1176, 1398, 1514, 4532, 8, 38100, 38105, 38112, 38117, 38124, 38129, 38131, 38136, 38143, 38148, 38150, 38155, 38162, 38167, 38174, 38179, 38181, 38186, 38193, 38198, 38201, 38206, 38213, 38218, 38220, 38225, 38232, 38237, 38244, 38249, 38251, 38256, 38263, 38268, 38270, 38275, 38282, 38287, 38294, 38299, 38302, 38307, 38314, 38319, 38321, 38326, 38333, 38338, 38340, 38345, 38352, 38357, 38364, 38369, 38371, 38376 ]
    
    # irisy: openml.datasets.get_dataset(61)


def save_dataset_to_file(ID, X, y, attribute_names, name):
    ''' 
    save a dataset to datasets directory
    in arff format
    using openML ID as file name
    '''                  
    
    df = pd.DataFrame(X)
    attr_types = df.dtypes
    
    dfy = pd.DataFrame(y)
    class_type = dfy.dtypes
                
      
    f = open("datasets"+os.sep+str(ID)+".arff", "w")
    
    class_attr = set(y)
    class_attr = list(class_attr)
    class_attr[:] = [str(ca) for ca in class_attr]
    class_attr_str = ",".join(class_attr)
    
    f.write("@RELATION "+name+" "+str(ID)+" from OpenML\n")
    f.write("\n") 
    i = 0
    for attr in attribute_names:
        t = str(attr_types[i])
        f.write("@ATTRIBUTE "+attr+" "+t+"\n")
        i += 1
        
    f.write("@ATTRIBUTE class {"+class_attr_str+"}\n")
    f.write("\n") 
    
    f.write("@DATA\n") 
            
    data = zip(X, y)
    for x,y in data:
        x_str = [str(item) for item in x]
        x_str = ",".join(x_str)
        f.write(x_str+","+str(y)+"\n")

    f.write("\n% EOF")
    f.close()
    print("File", ID, "from openML saved to datasets folder.")
    
    # save dataset to db    
    (instances, attributes) = X.shape
    
    # (name, openml_id, instances, attributes, task)
    val = [(name, ID, instances, attributes, len(class_attr), "classification")]
    
    database.insert_datasets(val)
    
    

def _try_to_typecast(x, t): # t .. list of numpy types (strings)    
    for typ in t:
        dtype = eval("np."+typ)
        try:
            res = dtype(x)
            return res
        except:
            pass

    return x     


def load_dataset_from_file(ID):
    '''
    try to load dataset from a file first,
    if that doesn't work
    search openML online
    '''
    
    try:
        f = open("datasets"+os.sep+str(ID)+".arff")
    except:
        # read file form online openML repository
        return None
    
    print("Reading dataset from file")
    
    lines = f.readlines()  # TODO deal with large files - do we need it? It would be a problem in many places, not just here
    
    # we want to get: X, y, attribute_names
        
    attr_names = []
    X = []
    y = [] 
    read_data = False
    data_types = []
    for line in lines:
        if read_data:
            line = line.strip()
            if len(line) > 0 and line[0] != "%":
                instance = line.split(",")
            
                x = [_try_to_typecast(x, [data_types[instance.index(x)]]) for x in instance[:-1]]                
                #y.append(_try_to_typecast(instance[-1], ["int32", "float32"]))
                y.append(_try_to_typecast(instance[-1], ["int64", "float64"]))                
                X.append(np.array(x))
    
        if line[:10] == "@ATTRIBUTE":
            # read attribute types                        
            xx, attr, t = line.split(" ")
            data_types.append(t.strip())
            
            if attr != "class": 
                attr_names.append(attr)
        if line[:9] == "@RELATION":
            name = line.split(" ")[1]        
        if line[:5] == "@DATA":
            read_data = True

    X = np.array(X)
    
    return X, y, attr_names, name
    
                        
def get_dataset(ID):
    '''
        Returns list of tuples (X,y) 
    '''
    datasets_list = []
    
    print("Dataset: "+str(ID))    
                
    try:
        X, y, attribute_names, name = load_dataset_from_file(ID)
        print ("Dataset name", name)     
    except:
        # read file directly form openML
        try:
            init()
        except Exception as e:
            print(e)
            return
        
        print("Reading dataset from openML")                 
        dataset = openml.datasets.get_dataset(ID)    
        print ("Dataset name", dataset.name)
        
        X, y, attribute_names = dataset.get_data(
                                target=dataset.default_target_attribute, 
                                return_attribute_names=True)                                

        if _config.cache_opemml_datasets:
            # save dataset
            save_dataset_to_file(ID, X, y, attribute_names, dataset.name)

    # print(X)
    # print(y)
    
    datasets_list.append((X,y, ID))
        
    
    return datasets_list


def get_10_liked_datasets(n):
    '''
        Returns list of tuples (X,y) 
    '''
    datasets_list = []
    dataset = openml.datasets.get_dataset(most_likes[n])

    print("Dataset: "+str(most_likes[n]))

    X, y, attribute_names = dataset.get_data(
                            target=dataset.default_target_attribute, 
                            return_attribute_names=True)
    datasets_list.append((X,y, most_likes[n]))
        
    return datasets_list



def get_datasets(first_n=10):
    '''
        Returns list of tuples (X,y) 
    '''
    datasets_list = []
    for d in datasets[:first_n]:
        dataset = openml.datasets.get_dataset(d["did"])

        print("Dataset: "+d["name"])

        X, y, attribute_names = dataset.get_data(
                            target=dataset.default_target_attribute, 
                            return_attribute_names=True)
        datasets_list.append((X,y, d["name"]))
                
    return datasets_list


#get_datasets()
