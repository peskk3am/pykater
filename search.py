import os
import sklearn.model_selection as model_selection

from preprocessings import *
from methods import *

import database
import _config

class Search():
    def __init__(self):
        experiment_id = database.get_experiment_id()
        print("experiment_id:", experiment_id)
        self.experiment_id = experiment_id

    def get_chain_string(self):
        chain_list = [eval(e).get_model_class().__name__ for e in self.chain_names]
        chain = "-".join(chain_list)
        return chain
        
    def get_results_file_name(self):
        chain = self.get_chain_string()        
        dir_name = _config.results_dir
        if len(chain.split("-")) > 1:
            dir_name = "pre_"+dir_name 
         
        file_name = dir_name+os.sep+type(self).__name__+"_"
        file_name += chain+"_"+self.dataset_name
                                        
        extension = "res"
        return file_name, extension
        
    def results_file_open(self, flag):
        file_name, extension = self.get_results_file_name()
            
        # open() flag 'x'	open for exclusive creation,
        # failing if the file already exists
        experiment_number = 0
        success = False
        while(not success):
            try:
                name = file_name+"_"+str(experiment_number)+"."+extension
                f = open(name, "x")
                success = True
            except FileExistsError:
                experiment_number += 1
        
        return f, name
        
    def write_header(self, f):        
        f.write("# Experiment id: "+str(self.experiment_id)+"\n"+
                "# Parameter search: "+type(self).__name__+
                    " ("+self.get_name()+")\n"+ 
                "# Dataset: "+str(self.dataset_name)+"\n"+
                "# Estimator: "+self.get_chain_string()+"\n")                               
        try:                                            
            f.write("# Cross Validation: "+str(self.cv)+"\n")            
        except:
            pass

        f.write("#\n")        

    def split_dataset(self, X, y):
        # Split the dataset into testing and training data
        X_train, X_test, y_train, y_test = \
          model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
        
        # X_train = X
        # X_test = X
        # y_train = y
        # y_test = y

        return X_train, X_test, y_train, y_test 