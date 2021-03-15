import numpy as np
from scipy.special import comb
from scipy.stats import nbinom, binom
import click
from joblib import Parallel, delayed
import json
import multiprocessing as mp

def generate_num_of_contacts(num_of_days, rng, k=1.68, r=10.47):

    p = r/(k+r)

    home_contacts = 2.197
    work_contacts = 2.909
    school_contacts = 1.181
    other_contacts = 3.343

    overall_contacts = home_contacts + work_contacts + school_contacts + other_contacts
    p_repetitive_contact = (home_contacts + work_contacts + school_contacts)/overall_contacts

    daily_contacts = nbinom.rvs(n=k, p=1-p, random_state=rng)
    
    final_contacts = 0
    same_per_day = binom.rvs(n=daily_contacts, p=p_repetitive_contact, random_state=rng)
    new_per_day = daily_contacts - same_per_day
    final_contacts = same_per_day + new_per_day * num_of_days

    return final_contacts
    
def generate_individual_contacts(N, r, k, rng):
    
    p = r/(k+r)
    num_of_infections = min(N,nbinom.rvs(n=k, p=1-p, random_state=rng)) # min makes sure the infected are not more than N

    is_infected = np.full(N, False)
    infection_ids = rng.choice(N, size=num_of_infections)
    is_infected[infection_ids] = True

    return is_infected

def evaluate_population(lambda_1, lambda_2, se, sp, is_infected, groups, rng):

    num_of_tests = 0
    false_negatives = 0
    false_positives = 0
    
    examined = 0
    for group_size in groups:
        
        if group_size == 1:
            group_tests = 1
            if is_infected[examined]:
                if rng.binomial(1, se)==0:
                    group_false_negatives = 1
                    group_false_positives = 0    
                else:
                    group_false_negatives = 0
                    group_false_positives = 0
            else:
                if rng.binomial(1, sp)==0:
                    group_false_negatives = 0
                    group_false_positives = 1
                else:
                    group_false_negatives = 0
                    group_false_positives = 0

        elif group_size > 1:

            group_is_infected = is_infected[examined : examined+group_size]
            num_of_infected = np.sum(group_is_infected)

            if rng.binomial(num_of_infected, se)>0 or rng.binomial(group_size - num_of_infected, 1-sp)>0:
            # Group test positive
                group_tests = 1+group_size
                group_false_negatives = rng.binomial(num_of_infected, 1-se)
                group_false_positives = rng.binomial(group_size-num_of_infected, 1-sp)
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
        result = nbinom.pmf(n, k, 1-p)/nbinom.cdf(N, k, 1-p)
    
    assert result>=0 and result<=1, "Q value out of bounds"
    return result

def compute_inner_sums(size, r, k, N):
    
    sum_array = np.zeros(N+1, dtype=np.longdouble)
    for s in range(0, size+1):
        
        inner_sum = 0
        for n in range(s, N+1):
            inner_sum += (comb(n, s)*comb(N-n, size-s)/comb(N, size)) * compute_q_value(n, r, k, N)

        assert inner_sum>=0 and inner_sum<=1, "Inner sum out of bounds"
        sum_array[s] = inner_sum

    return sum_array


def negbin_num_of_tests(size, se, sp, inner_sums):

    if size==1:
        result = 1

    elif size>1:

        double_sum = np.longdouble(0)
        for s in range(0, size+1):
            double_sum += (1-se)**s * sp**(size-s) * inner_sums[s]

        assert double_sum>=0 and double_sum<=1, "Double sum out of bounds"
        result = 1 + size*(1-double_sum)

    return result

def negbin_false_negatives(size, se, sp, r, k, N, inner_sums):

    if size==1:
        
        simple_sum = 0
        for n in range(1, N+1):
            simple_sum += (n/N) * compute_q_value(n, r, k, N)
        
        result = (1-se)*simple_sum

    elif size>1:

        double_sum = np.longdouble(0)
        for s in range(1, size+1):
            double_sum += inner_sums[s] * s * (1 - se*(1 - (1-se)**s * sp**(size-s)))

        result = double_sum

    return result


def negbin_false_positives(size, se, sp, r, k, N, inner_sums):

    if size==1:
        
        simple_sum = 0
        for n in range(0, N):
            simple_sum += ((N-n)/N) * compute_q_value(n, r, k, N)
        
        result = (1-sp)*simple_sum

    elif size>1:

        double_sum = np.longdouble(0)
        for s in range(0, size):
            double_sum += (1 - (1-se)**s * sp**(size-s)) * inner_sums[s] * (size-s) * (1-sp)

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

    groups = []
    grouped = 0
    next_group_id = 0
    splittings = np.flip(splittings)
    while grouped < N:
        next_group_id = splittings[grouped]
        groups.append(next_group_id)
        grouped += next_group_id

    return groups

def gen_and_eval_nbinom(r, k, days, lambda_1, lambda_2, se, sp, seed, method):

    rng = np.random.default_rng(seed)
    
    N = generate_num_of_contacts(num_of_days=days, rng=rng, k=1.68, r=10.47)
    
    if N==0:
        score, num_of_tests, false_negatives, false_positives, num_of_infected = (0, 0, 0, 0, 0)
    else:
        if method != 'individual':
            p = r/N
            groups = testing(lambda_1, lambda_2, se, sp, N, r=r, k=k, p=p, method=method)
        else:
            groups = list(np.full(N,1,dtype=int))

        is_infected = generate_individual_contacts(N, r, k, rng)
        num_of_infected = np.sum(is_infected)

        score, num_of_tests, false_negatives, false_positives = evaluate_population(lambda_1, lambda_2, se, sp, is_infected, groups, rng)

    return score, num_of_tests, false_negatives, false_positives, num_of_infected

def gen_and_eval_fixed(N, r, k, lambda_1, lambda_2, se, sp, groups, seed):

    rng = np.random.default_rng(seed)

    is_infected = generate_individual_contacts(N, r, k, rng)

    num_of_infected = np.sum(is_infected)

    score, num_of_tests, false_negatives, false_positives = evaluate_population(lambda_1, lambda_2, se, sp, is_infected, groups, rng)

    return score, num_of_tests, false_negatives, false_positives, num_of_infected

def generate_summary(lambda_1, lambda_2, se, sp, N, r, k, p, method, days, seed,
                        score, num_of_tests, false_negatives, false_positives, num_of_infected):

    summary = {}
    summary['lambda_1'] = str(lambda_1)
    summary['lambda_2'] = str(lambda_2)
    summary['se'] = str(se)
    summary['sp'] = str(sp)
    summary['r'] = str(r)
    summary['k'] = str(k)
    summary['method'] = method
    summary['score'] = str(score)
    summary['num_of_tests'] = str(num_of_tests)
    summary['false_negatives'] = str(false_negatives)
    summary['false_positives'] = str(false_positives)
    summary['num_of_infected'] = str(num_of_infected)
    summary['seed'] = str(seed)
    if N is not None:
        summary['N'] = str(N)
        summary['p'] = str(p)
    elif days is not None:
        summary['days'] = str(days)
    
    return summary

@click.command()
@click.option('--r', type=float, default=2.0, help="Reproductive rate")
@click.option('--k', type=float, default=2.0, help="Dispersion")
@click.option('--n', type=int, default=None, help="Number of contacts")
@click.option('--days', type=int, default=None, help="Number of days for generating contacts")
@click.option('--lambda_1', type=float, required=True, help="False Negative weight")
@click.option('--lambda_2', type=float, required=True, help="False Positive weight")
@click.option('--se', type=np.longdouble, required=True, help="Test Sensitivity")
@click.option('--sp', type=np.longdouble, required=True, help="Test Specificity")
@click.option('--method', type=str, required=True, help="Grouping method")
@click.option('--seeds', type=int, required=True, help="Number of contacts sets to be tested")
@click.option('--njobs', type=int, required=True, help="Number of parallel threads")
@click.option('--output', type=str, required=True, help="Output file name")
def experiment(r, k, n, days, lambda_1, lambda_2, se, sp, method, seeds, njobs, output):

    N = n # click doesn't accept upper case arguments
    p = None

    if (n is None and days is None) or (n is not None and days is not None):
        print('Unclear experiment type')
    elif n is None and days is not None:
        print('Starting varying N ' + method + ' experiment')
        results = Parallel(n_jobs=njobs, backend='multiprocessing')(delayed(gen_and_eval_nbinom)(r, k, days, lambda_1, lambda_2, se, sp, seed, method) for seed in range(1, seeds+1))
    
    elif n is not None and days is None:
        print('Starting fixed N ' + method + ' experiment')
        p = r/N
        
        if method != 'individual':
            groups = testing(lambda_1, lambda_2, se, sp, N, r=r, k=k, p=p, method=method)
        else:
            groups = list(np.full(N,1,dtype=int))
        
        results = Parallel(n_jobs=njobs, backend='multiprocessing')(delayed(gen_and_eval_fixed)(N, r, k, lambda_1, lambda_2, se, sp, groups, seed) for seed in range(1, seeds+1))

    score = [x[0] for x in results]
    num_of_tests = [x[1] for x in results]
    false_negatives = [x[2] for x in results]
    false_positives = [x[3] for x in results]
    num_of_infected = [x[4] for x in results]

    for seed in range(1,seeds+1):
        summary = generate_summary(lambda_1=lambda_1, lambda_2=lambda_2, se=se, sp=sp, N=N, r=r, k=k, p=p, days=days, method=method,
                                    score=score[seed-1], num_of_tests=num_of_tests[seed-1], false_negatives=false_negatives[seed-1],
                                    false_positives=false_positives[seed-1], num_of_infected=num_of_infected[seed-1], seed=seed)
        
        with open('{output}_seed_{seed}.json'.format(output=output, seed=seed), 'w') as outfile:
            json.dump(summary, outfile)

    print(np.mean(score))
    return


if __name__ == '__main__':
    experiment()