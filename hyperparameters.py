'''
    Hyper parameters for a classificator
'''

import math
import _config_grid_search

try:
    grid_n = _config_grid_search.grid_n   # number of selected values for each param 
except:
    grid_n = None

try:
    max_number_of_evaluations = _config_grid_search.max_number_of_evaluations
except:
    max_number_of_evaluations = None   


class HyperparameterSpace:
    '''
        Parameters' names
        -----------------
        Scikit learn example:
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        param_grid = {
          'pca__n_components': [5, 20, 30, 40, 50, 64],
          'logistic__alpha': np.logspace(-4, 4, 5),
        }
        ==> edited param names in config files    
    '''

    def __init__(self):        
        self.grid_hyperparameters = {}
        
        '''
        param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of parameter
        settings to try as values, or a list of such dictionaries, in which case
        the grids spanned by each dictionary in the list are explored. This 
        enables searching over any sequence of parameter settings.
        '''
        self.hyperparameters = []    
    
    def add_hyperparameter_space(self, another_hs):
        for param in another_hs.hyperparameters:
            self.add_hyperparameter(param)           
    
    def evaluations_count(self):
        evaluations = 1
        for hp in self.hyperparameters:                                         
            evaluations *= hp.n
        return evaluations
    
    def get_grid_parameters(self):
        global grid_n
        global max_number_of_evaluations
         
        if max_number_of_evaluations:                                        
            # set all params' n to 1, increase untill smaller than max_value
            # count evaluations for all params after each param change            
            changed = True
            while changed:                
                changed = False
                for hp in self.hyperparameters:                                         
                    if hp.n < hp.max_values and self.evaluations_count() <= max_number_of_evaluations: 
                        hp.n += 1
                        changed = True                    
                        last_increased = hp

            # decrease last increased parameter - it exceded the max_number_of_evaluations 
            last_increased.n -= 1                                                                                                      

        else:   # use grid_n parameter
            for hp in self.hyperparameters:
                if hp.max_values < grid_n:
                    hp.n = max_values
                else:
                    hp.n = grid_n                    
        
        # add hyperparameters to the dictionary
        evaluations = 1
        for hp in self.hyperparameters:
            self.add_grid_hyperparameter(hp)            
            evaluations *= hp.n            
        
        print("Number of evaluations expected:", evaluations)
        
        # return the dictionary
        return self.grid_hyperparameters
        
     
    def add_grid_hyperparameter(self, param): 
        global grid_n                
        
        param.grid_n = grid_n
                           
        param.values = param.get_grid_values()   
       
        self.grid_hyperparameters[param.name] = param.values
                

    def add_hyperparameter(self, param):                               
        self.hyperparameters += [param]
                
    
    def get_hyperparameter_by_name(self, name):
        for h in self.hyperparameters:
            if h.name == name:
                return h
        return None
            

class NumericHyperparameter():
    def __init__(self, name, lower, upper, default,
                 logarithmic_scale=_config_grid_search.logarithmic_scale):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.default = default   
        self.logarithmic_scale = logarithmic_scale        
        self.max_values = float('inf')     
        self.n = 1  # to be increased   

    def get_grid_values(self):              
        values = [self.lower]
        if self.logarithmic_scale:         
            for i in range(1, self.n-1):        
                values += [self.lower + 
                       (self.upper-self.lower)**(1/(self.n-1)*i)]                            
        else:           
            if self.n == 1:
                # average value
                values = [self.upper-self.lower / 2]            
            else:    
                step = (self.upper-self.lower)/(self.n-1)            
                for i in range(1, self.n-1):        
                    values += [self.lower + step * i]
        values += [self.upper]                        
        
        return values
        
                                         
class IntegerHyperparameter(NumericHyperparameter):                            
    def get_grid_values(self):
        numbers = super(IntegerHyperparameter, self).get_grid_values()                                                
        return list(map(lambda x: int(round(x)), numbers))                 

class FloatHyperparameter(NumericHyperparameter):
    pass
    
class CategoricalHyperparameter:
    def __init__(self, name, choices, default):
        self.name = name
        self.value = choices  # []
        self.default = default
        self.max_values = len(self.value)
        self.n = 1 
    
    def get_grid_values(self):
        return self.value[:self.n]
    
class Constant:
    def __init__(self, name, value):
        self.name = name
        self.value = value        
        self.max_values = 1
        self.n = 1

    def get_grid_values(self):
        return [self.value]
        

# test
def test():
    hs = HyperparameterSpace()
    hs.add_hyperparameter(IntegerHyperparameter(name="n_neighbors", lower=1, 
                          upper=100, default=1))
    hs.add_hyperparameter(CategoricalHyperparameter(name="weights", 
                          choices=["uniform", "distance"], default="uniform"))
    hs.add_hyperparameter(CategoricalHyperparameter(name="p", choices=[1, 2], 
                          default=2))
    hs.add_hyperparameter(FloatHyperparameter("t", 0.5, 1.5, default=1))        
    print(hs.get_hyperparameters())

#test()   