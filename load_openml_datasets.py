import openml_api_key

import openml
#import pandas as pd

openml.config.apikey = openml_api_key.key

# we also want to use the test server so as not to polute the production system
# openml.config.server = "http://test.openml.org/api/v1/xml"
openml.config.server = "http://www.openml.org/api/v1/xml"


datasets = openml.datasets.list_datasets()
#most_likes = [40536, 61, 37, 1142, 1039, 1464, 1130, 1116, 1128, 54, 59, 186, 294, 28, 4538, 1471, 40477, 9, 1596, 1489, 43, 118, 310, 40478, 372, 23380, 182, 4541, 13, 44, 1113, 1462, 1397, 40, 21, 1176, 1398, 1514, 4532, 8, 38100, 38105, 38112, 38117, 38124, 38129, 38131, 38136, 38143, 38148, 38150, 38155, 38162, 38167, 38174, 38179, 38181, 38186, 38193, 38198, 38201, 38206, 38213, 38218, 38220, 38225, 38232, 38237, 38244, 38249, 38251, 38256, 38263, 38268, 38270, 38275, 38282, 38287, 38294, 38299, 38302, 38307, 38314, 38319, 38321, 38326, 38333, 38338, 38340, 38345, 38352, 38357, 38364, 38369, 38371, 38376 ]
most_likes = [61, 37, 1142, 1039, 1464, 1130, 1116, 1128, 54, 59, 186, 294, 28, 4538, 1471, 40477, 9, 1596, 1489, 43, 118, 310, 40478, 372, 23380, 182, 4541, 13, 44, 1113, 1462, 1397, 40, 21, 1176, 1398, 1514, 4532, 8, 38100, 38105, 38112, 38117, 38124, 38129, 38131, 38136, 38143, 38148, 38150, 38155, 38162, 38167, 38174, 38179, 38181, 38186, 38193, 38198, 38201, 38206, 38213, 38218, 38220, 38225, 38232, 38237, 38244, 38249, 38251, 38256, 38263, 38268, 38270, 38275, 38282, 38287, 38294, 38299, 38302, 38307, 38314, 38319, 38321, 38326, 38333, 38338, 38340, 38345, 38352, 38357, 38364, 38369, 38371, 38376 ]

# irisy: openml.datasets.get_dataset(61)


def get_dataset(ID):
    '''
        Returns list of tuples (X,y) 
    '''
    datasets_list = []
    dataset = openml.datasets.get_dataset(ID)

    print("Dataset: "+str(ID))

    X, y, attribute_names = dataset.get_data(
                            target=dataset.default_target_attribute, 
                            return_attribute_names=True)
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
