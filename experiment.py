import numpy as np
from scipy.special import comb
from scipy.stats import nbinom
import click
from joblib import Parallel, delayed
import json

def generate_population(N, r, k, seed):

    p = r/(k+r)
    num_of_infections = nbinom.rvs(k, p)
    
    is_infected = np.full(N, False)
    infection_ids = np.random.choice(N, num_of_infections)
    is_infected[infection_ids] = True

    return is_infected

def evaluate_population(lambda_1, lambda_2, se, sp, is_infected, groups, seed):

    num_of_tests = 0
    false_negatives = 0
    false_positives = 0
    
    examined = 0
    for group_size in groups:
        
        if group_size == 1:
            group_tests = 1
            if is_infected[examined] and np.random.binomial(1, se)==0:
                group_false_negatives = 1
                group_false_positives = 0
            elif (not is_infected[examined]) and np.random.binomial(1, sp)==0:
                group_false_negatives = 0
                group_false_positives = 1

        elif group_size > 1:

            group_is_infected = is_infected[examined : examined+group_size]
            num_of_infected = np.sum(group_is_infected)

            if np.random.binomial(num_of_infected, se)>0 or np.random.binomial(group_size - num_of_infected, 1-sp)>0:
            # Group test positive
                group_tests = 1+group_size
                group_false_negatives = np.random.binomial(num_of_infected, 1-se)
                group_false_positives = np.random.binomial(group_size-num_of_infected, 1-sp)
            else:
            # Group test negative
                group_tests = 1
                group_false_negatives = num_of_infected
                group_false_positives = 0

        num_of_tests += group_tests
        false_negatives += group_false_negatives
        false_positives += group_false_positives
        examined += group_size
    
    score = lambda_1 * false_negatives + lambda_2 * false_positives + (1 - lambda_1 - lambda_2) * num_of_tests

    return score, num_of_tests, false_negatives, false_positives


def dorfman_num_of_tests(size, se, sp, p):

    if size==1:
        result = 1

    elif size>1:
        result = 1 + size * (se - (se + sp -1) * (1-p)**size)

    return result    

def dorfman_false_negatives(size, se, sp, p):

    if size==1:
        result = (1-se) * p

    elif size>1:
        result = (1-se**2) * p * size

    return result

def dorfman_false_positives(size, se, sp, p):

    if size==1:
        result = (1-sp) * (1-p)

    elif size>1:
        result = (1-sp)*se*size*(1-p) - size*(1-sp)*(se+sp-1)*(1-p)**size

    return result

def compute_q_value(n, r, k, N):

    if n > N:
        result = 0
    else:
        p = r/(k+r)
        result = nbinom.pmf(n, k, p)/nbinom.cdf(N, k, p)
    
    return result

def compute_inner_sums(size, r, k, N):
    
    sum_array = np.zeros(N+1)
    for s in range(0, size+1):
        
        inner_sum = 0
        for n in range(s, N+1):
            inner_sum += (comb(n, s)*comb(N-n, size-s)/comb(N, size)) * compute_q_value(n, r, k, N)

        sum_array[s] = inner_sum

    return sum_array


def negbin_num_of_tests(size, se, sp, inner_sums):

    if size==1:
        result = 1

    elif size>1:

        double_sum = 0
        for s in range(0, size+1):
            double_sum += comb(size, s) * (1-se)**s * sp**(size-s) * inner_sums[s]

        result = 1 + size*(1-double_sum)

    return result

def negbin_false_negatives(size, se, sp, r, k, N, inner_sums):

    if size==1:
        
        simple_sum = 0
        for n in range(1, N+1):
            simple_sum += (n/N) * compute_q_value(n, r, k, N)
        
        result = (1-se)*simple_sum

    elif size>1:

        double_sum = 0
        for s in range(1, size+1):
            double_sum += inner_sums[s] * s * (1 - se*(1 - comb(size,s) * (1-se)**s * sp**(size-s)))

        result = double_sum

    return result


def negbin_false_positives(size, se, sp, r, k, N, inner_sums):

    if size==1:
        
        simple_sum = 0
        for n in range(0, N):
            simple_sum += ((N-n)/N) * compute_q_value(n, r, k, N)
        
        result = (1-sp)*simple_sum

    elif size>1:

        double_sum = 0
        for s in range(0, size):
            double_sum += (1 - comb(size,s) * (1-se)**s * sp**(size-s)) * inner_sums[s] * (size-s) * (1-sp)

        result = double_sum

    return result

def group_score(size, lambda_1, lambda_2, se, sp, r, k, N, p, method='negbin'):

    if method == 'negbin':
        
        inner_sums = compute_inner_sums(size, r, k, N)

        score = lambda_1 * negbin_false_negatives(size, se, sp, r, k, N, inner_sums) + \
                lambda_2 * negbin_false_positives(size, se, sp, r, k, N, inner_sums) + \
                (1 - lambda_1 - lambda_2) * negbin_num_of_tests(size, se, sp, inner_sums)
    
    elif method == 'dorfman':
        score = lambda_1 * dorfman_false_negatives(size, se, sp, p) + \
                lambda_2 * dorfman_false_positives(size, se, sp, p) + \
                (1 - lambda_1 - lambda_2) * dorfman_num_of_tests(size, se, sp, p)
    
    return score


def testing(lambda_1, lambda_2, se, sp, N, r=2.5, k=3, p=0.2, method='negbin'):

    if (method != 'negbin') and (method != 'dorfman'):
        print('Unspecified method')
        return None
    
    # Precompute the objective value for all group sizes
    g_fun = np.zeros(N)
    for size in range(1, N+1):
        g_fun[size-1] = group_score(size, lambda_1, lambda_2, se, sp, r, k, N, p, method=method)

    # Dynamic programming
    h_fun = np.zeros(N+1)
    splittings = np.zeros(N+1, dtype=int)
    for i in range(1, N+1):
        
        min_val = 3*N
        for j in range(1, i+1):
            val = g_fun[j-1] + h_fun[i-j]
            if val < min_val:
                min_val = val
                min_splitting=j

        h_fun[i] = min_val
        splittings[i] = min_splitting

    # print('Optimal value: ' + str(h_fun[N]))
    groups = []
    grouped = 0
    next_group_id = 0
    splittings = np.flip(splittings)
    while grouped < N:
        next_group_id = splittings[grouped]
        groups.append(next_group_id)
        grouped += next_group_id
    
    # print(str(len(groups)) + ' groups with sizes: ' + str(groups))

    return h_fun[N], groups

def gen_and_eval(N, r, k, lambda_1, lambda_2, se, sp, groups, seed):

    np.random.seed(seed) # Set random seed for reproducibility

    is_infected = generate_population(N, r, k, seed)

    score, num_of_tests, false_negatives, false_positives = evaluate_population(lambda_1, lambda_2, se, sp, is_infected, groups, seed)

    return score, num_of_tests, false_negatives, false_positives

def generate_summary(lambda_1, lambda_2, se, sp, N, r, k, p, method, exp_score, score, num_of_tests, false_negatives, false_positives):

    summary = {}
    summary['lambda_1'] = str(lambda_1)
    summary['lambda_2'] = str(lambda_2)
    summary['se'] = str(se)
    summary['sp'] = str(sp)
    summary['N'] = str(N)
    summary['r'] = str(r)
    summary['k'] = str(k)
    summary['p'] = str(p)
    summary['method'] = method
    summary['exp_score'] = str(exp_score)
    summary['score'] = str(score)
    summary['num_of_tests'] = str(num_of_tests)
    summary['false_negatives'] = str(false_negatives)
    summary['false_positives'] = str(false_positives)

    return summary

@click.command()
@click.option('--lambda_1', type=float, required=True, help="False Negative weight")
@click.option('--lambda_2', type=float, required=True, help="False Positive weight")
@click.option('--se', type=float, required=True, help="Test Sensitivity")
@click.option('--sp', type=float, required=True, help="Test Specificity")
@click.option('--n', type=int, required=True, help="Number of contacts")
@click.option('--r', type=float, default=2.0, help="Reproductive rate")
@click.option('--k', type=float, default=2.0, help="Dispersion")
@click.option('--p', type=float, default=0.2, help="Sensitivity")
@click.option('--method', type=str, required=True, help="Grouping method")
@click.option('--seeds', type=int, required=True, help="Number of contacts sets to be tested")
@click.option('--njobs', type=int, required=True, help="Number of parallel threads")
@click.option('--output', type=str, required=True, help="Output file name")
def experiment(lambda_1, lambda_2, se, sp, n, r, k, p, method, seeds, njobs, output):

    N = n # click doesn't accept upper case arguments
    exp_score, groups = testing(lambda_1, lambda_2, se, sp, N, r=r, k=k, p=p, method=method)

    results = Parallel(n_jobs=njobs)(delayed(gen_and_eval)(N, r, k, lambda_1, lambda_2, se, sp, groups, seed) for seed in range(1, seeds+1))

    score = [x[0] for x in results]
    num_of_tests = [x[1] for x in results]
    false_negatives = [x[2] for x in results]
    false_positives = [x[3] for x in results]

    summary = generate_summary(lambda_1, lambda_2, se, sp, N, r, k, p, method,
                                exp_score, score, num_of_tests, false_negatives, false_positives)

    with open(output, 'w') as outfile:
        json.dump(summary, outfile)

    print(exp_score, np.mean(score))
    return
# REMEMBER TO CHECK REPRODUCIBILITY WITH THE TWO METHODS

# return

if __name__ == '__main__':
    experiment()