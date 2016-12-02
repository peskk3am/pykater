import random

from deap import base
from deap import creator
from deap import tools

from methods import hyperparameters


class EvolutionSearchCV():

    def __init__(self, estimator, hyperparameter_space):
        self.estimator = estimator
        self.hyperparameter_space = hyperparameter_space
        
        self.evolution_init(hyperparameter_space)
        
        # TODO crossvalidation!!!
                
    
    def get_params(self, individual):
        params = {}
        for p in self.params:
            params[p] = individual[0]
            individual = individual[1:]
        return params        


    def evalFitness(self, individual):
        score = 0
        try:
            self.estimator.set_params(**self.get_params(individual))
            #print(self.estimator.get_params())
            self.estimator.fit(self.X_train, self.y_train)
            score = self.estimator.score(self.X_test, self.y_test)
        except Exception as e:      
            print("Exception: "+str(e))

        #print("SCORE: ", score)
        return score,

                     
    def evolution_init(self, hyperparameter_space):        

        #----------
        # Evolution parameters
        #----------
        
        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        #
        # NGEN  is the number of generations for which the
        #       evolution runs
        #
        # NPOP  us the number of individuals in the population
        
        self.CXPB, self.MUTPB, self.NGEN = 0.5, 0.2, 5
        self.NPOP = 10
        
        random.seed(64)  # TODO
        
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)        
                
        self.toolbox = base.Toolbox()
                        
        #----------
        # Define the population / individuals
        #----------
        
        # For each parameter register a 'attr'
        # e.g. self.toolbox.register("attr_bool", random.randint, 0, 1)
        
        # generate attrs automatically                       
        self.params = []        
        for param in self.hyperparameter_space.hyperparameters:                        
            if param.__class__.__name__ == "FloatHyperparameter":            
                self.toolbox.register("attr_"+param.name, random.uniform, 
                                      param.lower, param.upper)
            if param.__class__.__name__ == "CategoricalHyperparameter":
                self.toolbox.register("attr_"+param.name, random.choice, 
                                      param.value)
            if param.__class__.__name__ == "IntegerHyperparameter":
                self.toolbox.register("attr_"+param.name, random.randint,
                                      param.lower, param.upper)
            if param.__class__.__name__ == "Constant":
                continue # do not add the parameter to evolution attributes
            self.params += [param.name]
                    
                
        attrs = ["self.toolbox.attr_"+a for a in self.params]
        
        # we need a tuple of attrs
        attrs_tupple_str = "("+", ".join(attrs)+")"
        
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                         eval(attrs_tupple_str)) 
        
        
        # Define the population to be a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, 
                              self.toolbox.individual)
                
        #----------
        # Operator registration
        #----------
        
        # register the goal / fitness function
        self.toolbox.register("evaluate", self.evalFitness)
        
        # register the crossover operator
        self.toolbox.register("mate", tools.cxTwoPoint)
        
        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.05
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        
        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        #----------
        # Create an initial population of NPOP individuals
        #----------         
        self.pop = self.toolbox.population(n=self.NPOP)    
   

    def predict(self, X):
        """Call predict on the estimator with the best found parameters."""

        return self.best_estimator.predict(X)


    def fit(self, X_train, y_train, X_test=None, y_test=None):
        
        self.X_train, self.y_train  = X_train, y_train
        if not X_test: self.X_test = X_train
        if not y_test: self.y_test = y_train        
        
        print("Start of evolution")
        
        # Evaluate the entire self.population
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit
                
        
        print("  Evaluated %i individuals" % len(self.pop))
        
        # Begin the evolution
        for g in range(self.NGEN):
            print("-- Generation %i --" % g)
            
            # Select the next generation individuals
            offspring = self.toolbox.select(self.pop, len(self.pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
        
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
    
                # cross two individuals with probability CXPB
                if random.random() < self.CXPB:
                    self.toolbox.mate(child1, child2)
    
                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values
    
            for mutant in offspring:    
                # mutate an individual with probability MUTPB
                if random.random() < self.MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
        
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            print("  Evaluated %i individuals" % len(invalid_ind))
            
            # The self.population is entirely replaced by the offspring
            self.pop[:] = offspring
            
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in self.pop]
            
            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
        
        print("-- End of (successful) evolution --")
        
        best_ind = tools.selBest(self.pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        
        self.best_params_ = self.get_params(best_ind)       
        self.best_estimator = self.estimator.set_params(**self.best_params_)         
        

'''
# test
from sklearn.neighbors import KNeighborsClassifier

hs = hyperparameters.HyperparameterSpace()
hs.add_hyperparameter(hyperparameters.IntegerHyperparameter(name="n_neighbors",
                      lower=1, upper=100, default=1))
hs.add_hyperparameter(hyperparameters.CategoricalHyperparameter(name="weights",
                      choices=["uniform", "distance"], default="uniform"))
hs.add_hyperparameter(hyperparameters.CategoricalHyperparameter(name="p", 
                      choices=[1, 2], default=2))
hs.add_hyperparameter(hyperparameters.FloatHyperparameter("t", 0.5, 1.5, 
                      default=1))         

EvolutionSearchCV(KNeighborsClassifier, hs)
'''
