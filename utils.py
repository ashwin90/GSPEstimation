import time
from itertools import chain, combinations
from math import log

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from numpy import array

NLP_LOWER_BOUND_INF = -1e19
NLP_UPPER_BOUND_INF = 1e19
LINEAR_SOLVER_TIME_LIMIT = 300

ZERO_LOWER_BOUND = 1e-6
ONE_UPPER_BOUND = 1.0 - ZERO_LOWER_BOUND

FINITE_DIFFERENCE_DELTA = 1e-7
ESTIMATOR_TIME_LIMIT = 3600

RUNTIME = 'time'
NLL_METRIC = 'KL_div'
RMSE_METRIC = 'RMSE'
CHISQ_METRIC = 'chisq'
HIT_RATE_METRIC = 'hit_rate'
MAPE_METRIC = 'mape'
L1_METRIC = 'L1_error' # used in Jena et al.
PRODUCT_CATEGORIES = ['shamp', 'toothbr', 'hhclean', 'yogurt', 'coffee']
# METRICS = [NLL_METRIC, RMSE_METRIC, MAPE_METRIC, L1_METRIC]
METRICS = [NLL_METRIC, L1_METRIC]


WITH_NO_CHOICE = 'w_nc'
WITHOUT_NO_CHOICE = 'wo_nc'
GPT_VARIANT = 'gpt'


def compute_probs_under_HaloMNL_model(membership, prod_utils, lambdas, new=False):
    pred_probs = np.zeros_like(membership, dtype=float)
    for k in range(len(lambdas)):
        if new:
            prod_utils_k = np.dot(membership, prod_utils[k])
        else:
            prod_utils_k = np.diag(prod_utils[k]) + np.dot(1 - membership, prod_utils[k])
        prod_wts_by_os = np.exp(prod_utils_k)*membership
        row_sums = np.sum(prod_wts_by_os, 1)
        choice_probs = prod_wts_by_os/row_sums[:, np.newaxis]
        assert np.all(np.around(np.sum(choice_probs, 1) - np.ones(membership.shape[0]), 7) == 0), (print('Choice probs in Halo-MNL model do not add up to 1!'), embed())
        pred_probs += lambdas[k]*choice_probs

    return pred_probs


def compute_probs_under_HaloMNL_model_v2(membership, prod_utils, lambdas):
    pred_probs = np.zeros_like(membership, dtype=float)
    for k in range(len(lambdas)):
        prod_utils_k = np.diag(prod_utils[k]) + np.dot(1 - membership, prod_utils[k])
        pred_probs += lambdas[k]*np.exp(prod_utils_k)*membership

    row_sums = np.sum(pred_probs, 1)
    choice_probs = pred_probs / row_sums[:, np.newaxis]
    assert np.all(np.around(np.sum(choice_probs, 1) - np.ones(membership.shape[0]), 7) == 0), (
        print('Choice probs in Halo-MNL model do not add up to 1!'), embed())
    return choice_probs


def plot_metrics(GSP_metrics, RL_metrics, metric, dataset, savepath, train=False):
    rl_xaxis = range(1, len(RL_metrics) + 1)
    plt.plot(rl_xaxis, RL_metrics, 'o-', label='RL')
    gsp_xaxis = range(1, len(GSP_metrics) + 1)
    plt.plot(gsp_xaxis, GSP_metrics, 'o-', label='GSP')
    plt.legend(loc='best', fancybox=True)
    plt.ylabel(metric)
    plt.xlabel('Number of iterations')
    if train:
        plt.title('Dataset: {0} Training data'.format(dataset))
    else:
        plt.title('Dataset: {0} Test data'.format(dataset))

    plt.savefig('{3}/{0}_{1}_{2}.pdf'.format(dataset, 'train' if train else 'test', metric, savepath))
    plt.close()


def safe_log(x):
    # This is to deal with infeasible optimization methods (those who don't care about evaluating objective function
    # inside constraints, this could cause evaluating outside log domain)
    if x > ZERO_LOWER_BOUND:
        return log(x)
    log_lower_bound = log(ZERO_LOWER_BOUND)
    a = 1 / (3 * ZERO_LOWER_BOUND * (3 * log_lower_bound * ZERO_LOWER_BOUND)**2)
    b = ZERO_LOWER_BOUND * (1 - 3 * log_lower_bound)
    return a * (x - b) ** 3


def finite_difference(function):
    def derivative(x):
        h = FINITE_DIFFERENCE_DELTA
        gradient = []
        x = list(x)
        for i, parameter in enumerate(x):
            plus = function(x[:i] + [parameter + h] + x[i + 1:])
            minus = function(x[:i] + [parameter - h] + x[i + 1:])
            gradient.append((plus - minus) / (2 * h))
        return array(gradient)
    return derivative


def generate_n_random_numbers_that_sum_one(n):
    distribution = [np.random.uniform(0, 1) for _ in range(n)]
    total = sum(distribution)

    for i in range(len(distribution)):
        distribution[i] = distribution[i] / total

    return distribution


def generate_n_equal_numbers_that_sum_one(n):
    # Sometimes precision makes numbers not end up equal if one does just return [1.0 / n for _ in range(n)]
    head = [1.0 / n for _ in range(n - 1)]
    return head + [1.0 - sum(head)]


def generate_n_random_numbers_that_sum_m(n, m):
    return map(lambda x: x * m, generate_n_random_numbers_that_sum_one(n))


def rindex(a_list, a_value):
    return len(a_list) - a_list[::-1].index(a_value) - 1


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


MPRINT_ON = False


def verbose_mode(new_value):
    global MPRINT_ON
    MPRINT_ON = new_value


def mprint(string):
    if MPRINT_ON:
        print(string)


ACCEPTABLE_ITERATIONS = 5
ACCEPTABLE_OBJ_DIFFERENCE = 1e-8
LIKELIHOOD_RATIO_THRESH = 1e-6
KLDIV_CUTOFF = 0.0
KLDIV_REL_CHANGE = 1e-8


class ConvergenceCriteria(object):
    def would_stop_this(self, profiler):
        raise NotImplementedError('Subclass responsibility')

    def reset_for(self, profiler):
        pass


class ObjectiveValueCriteria(ConvergenceCriteria):
    def __init__(self, acceptable_iterations, acceptable_objective_difference):
        self._acceptable_iterations = acceptable_iterations
        self._acceptable_objective_difference = acceptable_objective_difference
        self._last_considered_iteration = 0

    def acceptable_iterations(self):
        return self._acceptable_iterations

    def acceptable_objective_difference(self):
        return self._acceptable_objective_difference

    def reset_for(self, profiler):
        self._last_considered_iteration = len(profiler.iterations())

    def would_stop_this(self, profiler):
        last_iterations = profiler.iterations()[self._last_considered_iteration:][-self.acceptable_iterations():]
        if len(last_iterations) == self.acceptable_iterations():
            differences = [abs(last_iterations[i].value() - last_iterations[i - 1].value()) for i in range(1, len(last_iterations))]
            return all([difference < self.acceptable_objective_difference() for difference in differences])
        return False


class Iteration(object):
    def __init__(self):
        self._start_time = time.time()
        self._stop_time = None
        self._value = None

    def is_finished(self):
        return self._value is not None

    def finish_with(self, value):
        if self.is_finished():
            raise Exception('Finishing already finished iteration.')
        self._value = value
        self._stop_time = time.time()

    def value(self):
        return self._value

    def start_time(self):
        return self._start_time

    def stop_time(self):
        return self._stop_time

    def duration(self):
        return self.stop_time() - self.start_time()

    def as_json(self):
        return {'start': self.start_time(),
                'stop': self.stop_time(),
                'value': self.value()}

    def __repr__(self):
        data = (self.start_time(), self.stop_time(), self.duration(), self.value())
        return '< Start: %s ; Stop: %s ; Duration %s ; Value: %s >' % data


class Profiler(object):
    def __init__(self, verbose=True):
        self._verbose = verbose
        self._iterations = []
        self._convergence_criteria = ObjectiveValueCriteria(ACCEPTABLE_ITERATIONS, ACCEPTABLE_OBJ_DIFFERENCE)

    def iterations(self):
        return self._iterations

    def convergence_criteria(self):
        return self._convergence_criteria

    def json_iterations(self):
        return map(lambda i: i.as_json(), self.iterations())

    def last_iteration(self):
        return self._iterations[-1]

    def first_iteration(self):
        return self._iterations[0]

    def start_iteration(self):
        self._iterations.append(Iteration())

    def stop_iteration(self, value):
        self.last_iteration().finish_with(value)
        self.show_progress()

    def show_progress(self):
        if self._verbose:
            if len(self.iterations()) % 1000 == 1:
                mprint('----------------------')
                mprint('N#  \tTIME \tOBJ VALUE')

            if len(self.iterations()) % 100 == 1:
                mprint('%s\t%ss\t%.15f' % (len(self.iterations()), int(self.duration()), self.last_iteration().value()))

    def duration(self):
        if len(self.iterations()) > 0:
            return self.last_iteration().stop_time() - self.first_iteration().start_time()
        return 0

    def should_stop(self):
        return self.convergence_criteria().would_stop_this(self)

    def reset_convergence_criteria(self):
        self.convergence_criteria().reset_for(self)

    def update_time(self):
        if len(self._iterations) > 2:
            self.start_iteration()
            self.stop_iteration(self._iterations[-2].value())


def convert_aggregated_to_individual_choices(membership, aggregated_choices):
    empirical_choice_probs = aggregated_choices / np.sum(aggregated_choices, 1)[:, np.newaxis]
    empirical_choice_probs = empirical_choice_probs[empirical_choice_probs > 0]
    prods_chosen = aggregated_choices.nonzero()[1]
    n_C = aggregated_choices[aggregated_choices > 0]
    n_obs_per_offerset = np.sum(aggregated_choices > 0, 1)
    X_expanded = np.repeat(membership, n_obs_per_offerset, axis=0)
    return empirical_choice_probs, X_expanded, n_C, prods_chosen
