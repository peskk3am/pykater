import os

class Search():
    def get_results_file_name(self):
        file_name = "results"+os.sep+type(self).__name__+"_"
        file_name += type(self.estimator).__name__+"_"+self.dataset_name
                                        
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
        
        f.write("# Parameter search: "+type(self).__name__+
                    " ("+self.get_name()+")\n"+ 
                "# Dataset: "+str(self.dataset_name)+"\n"+
                "# Estimator: "+type(self.estimator).__name__+"\n")                               
        try:                                            
            f.write("# Cross Validation: "+str(self.cv)+"\n")            
        except:
            pass

        f.write("#\n")        