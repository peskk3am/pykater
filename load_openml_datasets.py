import openml_api_key

import openml
#import pandas as pd

openml.config.apikey = openml_api_key.key

# we also want to use the test server so as not to polute the production system
openml.config.server = "http://test.openml.org/api/v1/xml"

datasets = openml.datasets.list_datasets()

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
        datasets_list.append((X,y))
        
    return datasets_list


#get_datasets()
