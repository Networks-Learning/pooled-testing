import numpy as np

def generate_population(N, R0, k, seed):

    np.random.seed(seed=seed)
    num_of_infections = np.random.negative_binomial(5, 0.4)
    
    is_infected = np.full(N, False)
    infection_ids = np.random.choice(N, num_of_infections)
    is_infected[infection_ids] = True

    return is_infected

test= generate_population(10, 2, 1.3, 2)
print(test)