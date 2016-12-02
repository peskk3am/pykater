'''
    Hyper parameters for a classificator
'''

grid_n = 10   # number of selected values 

class HyperparameterSpace:

    def __init__(self):
        self.grid_hyperparameters = [{}]
        self.hyperparameters = []    
     
    def add_grid_hyperparameter(self, param): 
        global grid_n
        param.grid_n = grid_n
        param.value = param.get_grid_values()   
       
        self.grid_hyperparameters[0][param.name] = param.value        

    def add_hyperparameter(self, param):
        self.hyperparameters += [param]
            
        
class CategoricalHyperparameter:
    def __init__(self, name, choices, default):
        self.name = name
        self.value = choices  # []
        self.default = default
    
    def get_grid_values(self):
        return self.value
                         
class IntegerHyperparameter:
    def __init__(self, name, lower, upper, default):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.default = default   
    
    def get_grid_values(self):
        step = int(round((self.upper-self.lower)/(self.grid_n-1)))
        integers = list(range(self.lower,self.upper,step))+[self.upper]
        if len(integers) < self.grid_n:
            # TODO add more integers
            pass                                        
        
        return integers

class FloatHyperparameter:
    def __init__(self, name, lower, upper, default):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.default = default
    
    def get_grid_values(self):
        step = (self.upper-self.lower)/(self.grid_n-1)
        floats = []
        act = self.lower
        while act < self.upper:
            floats += [act]
            act += step            
        floats += [self.upper]                                      
        
        return floats

class Constant:
    def __init__(self, name, value):
        self.name = name
        self.value = [value]

    def get_grid_values(self):
        return self.value
        

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