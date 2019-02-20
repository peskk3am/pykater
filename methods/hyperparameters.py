'''
    Hyper parameters for a classificator
'''

import _config_grid_search
import math

grid_n = _config_grid_search.grid_n   # number of selected values 

class HyperparameterSpace:

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
     
    def add_grid_hyperparameter(self, param): 
        global grid_n
        param.grid_n = grid_n
        param.values = param.get_grid_values()   
       
        self.grid_hyperparameters[param.name] = param.values        

    def add_hyperparameter(self, param):              
        self.add_grid_hyperparameter(param)
        self.hyperparameters += [param]
            

class NumericHyperparameter():
    def __init__(self, name, lower, upper, default,
                 logarithmic_scale=_config_grid_search.logarithmic_scale):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.default = default   
        self.logarithmic_scale = logarithmic_scale

    def get_grid_values(self):              
        values = [self.lower]
        if self.logarithmic_scale:         
            for i in range(1,grid_n-1):        
                values += [self.lower + 
                       (self.upper-self.lower)**(1/(self.grid_n-1)*i)]                            
        else:
            step = (self.upper-self.lower)/(self.grid_n-1)            
            for i in range(1,grid_n-1):        
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
    
    def get_grid_values(self):
        return self.value
    
class Constant:
    def __init__(self, name, value):
        self.name = name
        self.value = value        

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