from deap import tools
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

CXPB, MUTPB, NGEN = 0.5, 0.3, 50
NPOP = 10        

# keep the (n) best individual(s) and put it in the next generation
elitism = 1   

cv_folds = 10

# Crossovers:
# - for details and more cross over operators see DEAP documentation
#
#   cxOnePoint()
#   cxTwoPoint()
#   cxUniform()
#   ...

crossover = tools.cxTwoPoint  

# Mutation operators:
#   List of operators: (mutation, probability)
#   mutation ... "gaussian", "uniform"
# probability - Independent probability for each attribute to be mutated.

#mutation_operators = [("gaussian", 0.05), ("uniform", 0.1)]
mutation_operators = [("gaussian", 0.5), ("uniform", 0.5)]
