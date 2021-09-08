import numpy as np
from scipy.special import comb
from scipy.stats import nbinom, binom, bernoulli, poisson
import click
from joblib import Parallel, delayed
import json
import multiprocessing as mp
    
def generate_infected_contacts(N, r, k, rng, N_untraced):
    
    p = r/(k+r)
    num_of_infections = nbinom.rvs(n=k, p=1-p, random_state=rng) # Sample from a negative binomial (Refer to numpy documentation reg. 1-p)

    while num_of_infections > N + N_untraced:
        num_of_infections = nbinom.rvs(n=k, p=1-p, random_state=rng) # Reject if it is larger than N

    is_infected = np.full(N + N_untraced, False)
    infection_ids = rng.choice(N + N_untraced, size=num_of_infections, replace=False)
    is_infected[infection_ids] = True

    is_observed_infected = rng.choice(is_infected, size=N, replace=False)
 
    return is_observed_infected

def evaluate_population(lambda_1, lambda_2, se, sp, d, is_infected, groups, rng):

    num_of_tests = 0
    false_negatives = 0
    false_positives = 0
    
    examined = 0
    for group_size in groups:
        
        if group_size == 1:
            # Individual testing
            group_tests = 1

            if is_infected[examined]:
                if rng.binomial(1, se)==0:
                    # False negative
                    group_false_negatives = 1
                    group_false_positives = 0    
                else:
                    group_false_negatives = 0
                    group_false_positives = 0
            else:
                if rng.binomial(1, sp)==0:
                    # False positive
                    group_false_negatives = 0
                    group_false_positives = 1
                else:
                    group_false_negatives = 0
                    group_false_positives = 0

        elif group_size > 1:
            # Group testing
            group_is_infected = is_infected[examined : examined+group_size]
            num_of_infected = np.sum(group_is_infected)

            # if (num_of_infected > 0 and rng.binomial(1, se)==1) or (num_of_infected == 0 and rng.binomial(1, sp)==0):
            if (num_of_infected > 0 and rng.binomial(1, 1-sp+(se+sp-1)*np.power(num_of_infected/group_size, d))==1) or (num_of_infected == 0 and rng.binomial(1, sp)==0):
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
    
    score = num_of_tests + lambda_1 * false_negatives + lambda_2 * false_positives

    return score, num_of_tests, false_negatives, false_positives

def compute_q_value(n, r, N, method, k=None):

    if method=='negbin':
        if n > N:
            result = 0
        else:
            p = r/(k+r)
            result = nbinom.pmf(n, k, 1-p)/nbinom.cdf(N, k, 1-p)

    elif method=='poisson':
        if n > N:
            result = 0
        else:
            result = poisson.pmf(n, r)/poisson.cdf(N, r)
    
    assert result>=0 and result<=1, "Q value out of bounds"
    return result

def compute_inner_sums(size, r, N, method, k=None):
    
    sum_array = np.zeros(N+1, dtype=np.longdouble)
    for s in range(0, size+1):
        
        inner_sum = 0
        for n in range(s, N+1):
            inner_sum += (comb(n, s)*comb(N-n, size-s)/comb(N, size)) * compute_q_value(n=n, r=r, N=N, method=method, k=k)

        assert inner_sum>=0 and inner_sum<=1, "Inner sum out of bounds"
        sum_array[s] = inner_sum

    return sum_array


def compute_num_of_tests(size, se, sp, d, method, inner_sums=None, p_bernoulli=None):

    if method=='poisson' or method=='negbin':
        
        if size==1:
            result = 1

        elif size>1:

            double_sum = np.longdouble(0)
            for s in range(1, size+1):
                double_sum += (sp - (se+sp-1) * np.power(s/size, d)) * inner_sums[s]

            assert double_sum>=0 and double_sum<=1, "Double sum out of bounds"
            result = 1 + size*(1 - double_sum - sp*inner_sums[0])

    elif method=='binomial':
         
        if size==1:
            result = 1
        
        elif size>1:

            double_sum = np.longdouble(0)
            for s in range(1, size+1):
                double_sum += (sp - (se+sp-1) * np.power(s/size, d)) * (comb(size, s) * np.power(p_bernoulli, s) * np.power(1-p_bernoulli, size-s))
            
            assert double_sum>=0 and double_sum<=1, "Double sum out of bounds"
            result = 1 + size * (1 - double_sum - sp*np.power(1-p_bernoulli, size))
        
    return result

def compute_false_negatives(size, se, sp, d, method, r=None, k=None, N=None, inner_sums=None, p_bernoulli=None):

    if method == 'poisson' or method=='negbin':

        if size==1:
            
            simple_sum = 0
            for n in range(1, N+1):
                simple_sum += (n/N) * compute_q_value(n=n, r=r, k=k, N=N, method=method)
            
            result = (1-se)*simple_sum

        elif size>1:

            double_sum = np.longdouble(0)
            for s in range(1, size+1):
                double_sum += s * (1 - se + se*(sp - (se+sp-1)*np.power(s/size,d))) * inner_sums[s]

            result = double_sum

    elif method=='binomial':

        if size==1:
            result = (1-se) * p_bernoulli

        elif size>1:

            double_sum = np.longdouble(0)
            for s in range(1, size+1):
                double_sum += s * (1 - se + se*(sp - (se+sp-1)*np.power(s/size,d))) * (comb(size, s) * np.power(p_bernoulli, s) * np.power(1-p_bernoulli, size-s))

            result = double_sum

    return result


def compute_false_positives(size, se, sp, d, method, r=None, k=None, N=None, inner_sums=None, p_bernoulli=None):

    if method == 'poisson' or method=='negbin':
    
        if size==1:
            
            simple_sum = 0
            for n in range(0, N):
                simple_sum += ((N-n)/N) * compute_q_value(n=n, r=r, k=k, N=N, method=method)
            
            result = (1-sp)*simple_sum

        elif size>1:

            double_sum = np.longdouble(0)
            for s in range(1, size):
                double_sum += (1 - sp + (se+sp-1) * np.power(s/size, d)) * (size-s) * (1-sp) * inner_sums[s]

            result = (1-sp)**2 * size * inner_sums[0] + double_sum

    elif method == 'binomial':

        if size==1:
            result = (1-sp) * (1-p_bernoulli)

        elif size>1:
            
            double_sum = np.longdouble(0)
            for s in range(1, size):
                double_sum += (1 - sp + (se+sp-1) * np.power(s/size, d)) * (size-s) * (1-sp) * (comb(size, s) * np.power(p_bernoulli, s) * np.power(1-p_bernoulli, size-s))

            result = (1-sp)**2 * size * np.power(1-p_bernoulli, size) + double_sum

    return result

# Computes the expected tests, expected false negatives & expected false positives
# for a given group size
def group_score(size, lambda_1, lambda_2, se, sp, d, method, r=None, k=None, N=None, p_bernoulli=None):

    if method == 'negbin' or method == 'poisson':
        inner_sums = compute_inner_sums(size=size, r=r, k=k, N=N, method=method)
    else:
        inner_sums = None
    
    expected_false_negatives = compute_false_negatives(size=size, se=se, sp=sp, d=d, r=r, k=k, N=N, inner_sums=inner_sums, p_bernoulli=p_bernoulli, method=method)
    expected_false_positives = compute_false_positives(size=size, se=se, sp=sp, d=d, r=r, k=k, N=N, inner_sums=inner_sums, p_bernoulli=p_bernoulli, method=method)
    expected_num_of_tests = compute_num_of_tests(size=size, se=se, sp=sp, d=d, inner_sums=inner_sums, p_bernoulli=p_bernoulli, method=method)
    
    score =  expected_num_of_tests + lambda_1*expected_false_negatives + lambda_2*expected_false_positives
    
    return score, expected_false_negatives, expected_false_positives, expected_num_of_tests

# Dynamic programming algorithm to split individuals into groups
def testing(lambda_1, lambda_2, se, sp, d, N, method, r=None, k=None, p_bernoulli=None):

    if (method != 'negbin') and (method != 'binomial') and (method != 'poisson'):
        print('Unspecified method')
        return None
    
    # Precompute the objective value for all group sizes
    g_fun = np.zeros(N)
    fn_fun = np.zeros(N)
    fp_fun = np.zeros(N)
    tests_fun = np.zeros(N)
    for size in range(1, N+1):
        g_fun[size-1], fn_fun[size-1], fp_fun[size-1], tests_fun[size-1] = \
            group_score(size=size, lambda_1=lambda_1, lambda_2=lambda_2,
                        se=se, sp=sp, d=d, r=r, k=k, N=N, p_bernoulli=p_bernoulli, method=method)

    # Dynamic programming
    h_fun = np.zeros(N+1)
    splittings = np.zeros(N+1, dtype=int)
    fn_total = np.zeros(N+1)
    fp_total = np.zeros(N+1)
    tests_total = np.zeros(N+1)
    for i in range(1, N+1):
        
        min_val = 3*N
        for j in range(1, i+1):
            val = g_fun[j-1] + h_fun[i-j]
            if val < min_val:
                min_val = val
                min_splitting=j

        h_fun[i] = min_val
        splittings[i] = min_splitting
        fn_total[i] = fn_fun[min_splitting-1] + fn_total[i-min_splitting]
        fp_total[i] = fp_fun[min_splitting-1] + fp_total[i-min_splitting]
        tests_total[i] = tests_fun[min_splitting-1] + tests_total[i-min_splitting]

    groups = []
    grouped = 0
    next_group_id = 0
    splittings = np.flip(splittings)
    while grouped < N:
        next_group_id = splittings[grouped]
        groups.append(next_group_id)
        grouped += next_group_id

    return groups, fn_total[N], fp_total[N], tests_total[N]

# Generates infected contacts, performs testing based on the given group sizes
# and returns the number of tests, false negatives and false positives
def gen_and_eval_fixed(N, r, k, lambda_1, lambda_2, se, sp, d, groups, seed, N_untraced):

    rng = np.random.default_rng(seed)
    
    is_infected = generate_infected_contacts(N, r, k, rng, N_untraced)

    num_of_infected = np.sum(is_infected)

    score, num_of_tests, false_negatives, false_positives = evaluate_population(lambda_1, lambda_2, se, sp, d, is_infected, groups, rng)

    return score, num_of_tests, false_negatives, false_positives, num_of_infected

# Saves configuration and results to a JSON file
def generate_summary(lambda_1, lambda_2, se, sp, d, N, untraced, r, k, method, seeds, groups,
                        exp_fn, exp_fp, exp_tests,
                        score, num_of_tests, false_negatives, false_positives, num_of_infected):

    summary = {}
    summary['lambda_1'] = str(lambda_1)
    summary['lambda_2'] = str(lambda_2)
    summary['se'] = str(se)
    summary['sp'] = str(sp)
    summary['d'] = str(d)
    summary['r'] = str(r)
    summary['k'] = str(k)
    summary['method'] = method
    summary['N'] = str(N)
    summary['untraced'] = str(untraced)
    summary['exp_fn'] = str(exp_fn)
    summary['exp_fp'] = str(exp_fp)
    summary['exp_tests'] = str(exp_tests)
    
    summary['groups'] = {}
    for ind, group_size in enumerate(groups):
        summary['groups'][ind+1] = str(group_size)
    
    summary['seeds'] = {}
    for seed in range(1, seeds+1):
        summary['seeds'][seed] = {}
        summary['seeds'][seed]['score'] = str(score[seed-1])
        summary['seeds'][seed]['num_of_tests'] = str(num_of_tests[seed-1])
        summary['seeds'][seed]['false_negatives'] = str(false_negatives[seed-1])
        summary['seeds'][seed]['false_positives'] = str(false_positives[seed-1])
        summary['seeds'][seed]['num_of_infected'] = str(num_of_infected[seed-1])
     
    return summary

@click.command() # Comment the click commands for testing
@click.option('--r', type=float, required=True, help="Reproductive rate")
@click.option('--k', type=float, required=True, help="Dispersion")
@click.option('--n', type=int, required=True, help="Number of traced contacts")
@click.option('--untraced', type=float, required=False, default=0.0, help="Percentage of total contacts who are untraced")
@click.option('--lambda_1', type=float, required=True, help="False Negative weight")
@click.option('--lambda_2', type=float, required=True, help="False Positive weight")
@click.option('--se', type=np.longdouble, required=True, help="Test Sensitivity")
@click.option('--sp', type=np.longdouble, required=True, help="Test Specificity")
@click.option('--d', type=np.longdouble, required=True, help="Dilution")
@click.option('--method', type=str, required=True, help="Grouping method")
@click.option('--seeds', type=int, required=True, help="Number of contacts sets to be tested")
@click.option('--njobs', type=int, required=True, help="Number of parallel threads")
@click.option('--output', type=str, required=True, help="Output file name")
def experiment(r, k, n, untraced, lambda_1, lambda_2, se, sp, d, method, seeds, njobs, output):

    N = n # click doesn't accept upper case arguments
    N_untraced = int(np.around(untraced*N / (1-untraced)))

    if method=='binomial':
        
        # Estimate the probability of infection for the Binomial giving the same expected number of infected as the Generalized Negative Binomial
        effective_mean = 0
        for n in range(0,N+1):
            effective_mean += n * compute_q_value(n, r, N, 'negbin', k)
        
        p_bernoulli = effective_mean/N

        print('Computing optimal groups under ' + method + ' assumption...')
        groups, exp_fn, exp_fp, exp_tests = testing(lambda_1=lambda_1, lambda_2=lambda_2, se=se, sp=sp, d=d, N=N, method=method, p_bernoulli=p_bernoulli)

    elif method == 'individual':

        print('Computing optimal groups under ' + method + ' assumption...')
        groups, exp_fn, exp_fp, exp_tests = (list(np.full(N,1,dtype=int)), None, None, None) # Expected numbers left undefined for individual testing

    else:

        print('Computing optimal groups under ' + method + ' assumption...')
        groups, exp_fn, exp_fp, exp_tests = testing(lambda_1=lambda_1, lambda_2=lambda_2, se=se, sp=sp, d=d, N=N, r=r, k=k, method=method)

    print('Evaluating...')
    results = Parallel(n_jobs=njobs, backend='multiprocessing')(delayed(gen_and_eval_fixed)(N, r, k, lambda_1, lambda_2, se, sp, d, groups, seed, N_untraced) for seed in range(1, seeds+1))

    score = [x[0] for x in results]
    num_of_tests = [x[1] for x in results]
    false_negatives = [x[2] for x in results]
    false_positives = [x[3] for x in results]
    num_of_infected = [x[4] for x in results]

    print('Saving results...')
    summary = generate_summary(lambda_1=lambda_1, lambda_2=lambda_2, se=se, sp=sp, d=d, N=N, untraced=untraced, r=r, k=k, method=method,
                                exp_fn=exp_fn, exp_fp=exp_fp, exp_tests=exp_tests,
                                score=score, num_of_tests=num_of_tests, false_negatives=false_negatives, false_positives=false_positives,
                                groups=groups, num_of_infected=num_of_infected, seeds=seeds)
        
    with open('{output}.json'.format(output=output), 'w') as outfile:
        json.dump(summary, outfile)

    return

# Temporary function to compare the q-values of Poisson and Negative Binomial 
def testing_q_values(N, r, k):
    
    poisson_q_values = []
    negbin_q_values = []
    for n in range(0, N+1):
        poisson_q_values.append(compute_q_value(n=n, r=r, N=N, method='poisson'))
        negbin_q_values.append(compute_q_value(n=n, r=r, k=k, N=N, method='negbin'))

    print(poisson_q_values)
    print(negbin_q_values)  
    return
    
# Temporary function to compare the expected number of tests, FNs and FPs of Poisson and Negative Binomial
def testing_exp_values(N, r, k, lambda_1, lambda_2, se, sp, seeds):

    results={}

    effective_mean = 0
    for n in range(0,N+1):
        effective_mean += n * compute_q_value(n, r, N, 'negbin', k)
    
    p_bernoulli = effective_mean/N

    for method in ['binomial', 'negbin']:
        
        results[method] = (np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N))
        for size in range(1, N+1):

            results[method][0][size-1], results[method][1][size-1], results[method][2][size-1], results[method][3][size-1] = \
                group_score(size=size, lambda_1=lambda_1, lambda_2=lambda_2,
                            se=se, sp=sp, r=r, k=k, N=N, p_bernoulli=p_bernoulli, method=method)

    print(results)
    return

if __name__ == '__main__':
    experiment()
    # testing_q_values(N=100, r=2.5, k=0.2)
    # testing_exp_values(N=100, r=2.5, k=0.2, lambda_1=0.0, lambda_2=0.0, se=0.95, sp=0.95, seeds=100000)
    # experiment(r = 2.5, k = 0.1, n = 50, untraced=0.0, lambda_1 = 0.0, lambda_2 = 0.0, se = 0.8, sp = 0.98, d=0.3,
    #             method = 'negbin', seeds = 10000, njobs = 1, output = 'outputs/test')