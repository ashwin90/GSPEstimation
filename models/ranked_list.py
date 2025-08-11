from numpy import ones, zeros
from numpy.random import RandomState

from models import Model
from optimization.non_linear import Constraints
from utils import *


class RankedListModel(Model):
    @classmethod
    def code(cls):
        return 'rl'

    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['ranked_lists'], data['betas'])

    def populate(self, transactions):
        chosen_products = []
        offered_products = []
        observed_sales = []
        num_prods = len(self.products)
        offer_set_dict = {}
        num_uniq_tx = 0
        for tr in transactions:
            os = np.zeros(num_prods, dtype=np.int)
            os[tr.offered_products] = 1
            chosen_product = tr.product
            tr_info = (chosen_product, tuple(os))
            if tr_info not in offer_set_dict:
                offer_set_dict[tr_info] = num_uniq_tx
                num_uniq_tx += 1
                chosen_products.append(chosen_product)
                observed_sales.append(tr.sales)
                offered_products.append(os)
            else:
                observed_sales[offer_set_dict[tr_info]] += tr.sales

        self.offered_products = np.array(offered_products)
        self.chosen_products = np.array(chosen_products)
        self.observed_sales = np.array(observed_sales)
        self.num_transactions = self.offered_products.shape[0]
        self.init_observed_choice_indicators()

    def init_observed_choice_indicators(self):
        self.observed_choice_indicators = np.zeros((self.num_transactions, self.num_rankings), dtype=np.int)
        for r, ranking in enumerate(self.ranked_lists):
            self.observed_choice_indicators[:, r] = self.transactions_compatible_with(self.offered_products, self.chosen_products, ranking)

    def init_from_parsed_data(self, offered_products, chosen_products, observed_sales, empirical_choice_probs):
        self.offered_products = offered_products
        self.chosen_products = chosen_products
        self.observed_sales = observed_sales
        self.num_transactions = self.offered_products.shape[0]
        self.init_observed_choice_indicators()
        self.empirical_choice_probs = empirical_choice_probs

    @classmethod
    def default_naive(cls, products, variant=WITHOUT_NO_CHOICE, ranked_lists=None):
        if not ranked_lists:
            products_set = set(products) - {0}
            ranked_lists = [products]
            prng = RandomState(3)
            for first_product in products_set:
                remaining_products = list(products_set - {first_product})
                ranked_lists.append([first_product] + [0] + list(prng.permutation(remaining_products)))

        betas = generate_n_equal_numbers_that_sum_one(len(ranked_lists))
        return cls(products, ranked_lists, betas, variant)

    @classmethod
    def default_random(cls, products, variant, ranked_lists):
        betas = generate_n_random_numbers_that_sum_one(len(ranked_lists))
        return cls(products, ranked_lists, betas, variant)

    def __init__(self, products, ranked_lists, betas, variant=WITHOUT_NO_CHOICE):
        super(RankedListModel, self).__init__(products)
        if len(betas) != len(ranked_lists):
            info = (len(betas), len(ranked_lists))
            raise Exception('Amount of betas (%s) should be equal to that of ranked lists (%s).' % info)
        if any([len(ranked_list) != len(products) for ranked_list in ranked_lists]):
            info = (products, ranked_lists)
            raise Exception('All ranked list should have all products.\n Products: %s\n Ranked lists: %s\n' % info)

        self.rl_dict = dict()
        self.ranked_lists = np.array(ranked_lists)
        for ranking in ranked_lists:
            self.rl_dict[tuple(ranking)] = 1
        self.betas = np.array(betas)
        self.num_rankings = len(self.rl_dict)
        self.variant = variant

        # to be populated later
        self.offered_products = None
        self.chosen_products = None
        self.observed_sales = None
        self.num_transactions = None
        self.observed_choice_indicators = None
        self.empirical_choice_probs = None

    def predict_insample_proba(self):
        return np.dot(self.observed_choice_indicators, self.betas)

    def predict_proba(self, membership, chosen_products):
        probs = np.zeros(membership.shape[0])
        for tid, ranking in enumerate(self.ranked_lists):
            probs += self.betas[tid]*self.transactions_compatible_with(membership, chosen_products, ranking)
        return probs

    def log_likelihood_for(self, transactions):
        return np.dot(self.observed_sales, np.ma.log(self.predict_insample_proba()).filled(0.)) / np.sum(self.observed_sales)

    def amount_of_ranked_lists(self):
        return len(self.ranked_lists)

    def transactions_compatible_with(self, membership, chosen_products, ranked_list):
        return ranked_list[np.argmax(membership[:, ranked_list], axis=1)] == chosen_products

    def add_ranked_list(self, ranked_list):
        if tuple(ranked_list) not in self.rl_dict:
            percentage = 1.0 / (len(self.betas) + 1.0)
            new_beta = [percentage*np.sum(self.betas[1:])]
            self.betas = np.concatenate((self.betas[:1], (1.0 - percentage)*self.betas[1:], new_beta))
            self.ranked_lists = np.vstack((self.ranked_lists, ranked_list[np.newaxis]))
            self.observed_choice_indicators = np.hstack((self.observed_choice_indicators, self.transactions_compatible_with(self.offered_products, self.chosen_products, ranked_list)[:, np.newaxis]))
            self.rl_dict[tuple(ranked_list)] = 1
            self.num_rankings += 1

    def parameters_vector(self):
        return self.betas

    def update_parameters_from_vector(self, parameters):
        self.betas = list(parameters)

    def constraints(self):
        return RankedListModelConstraints(self)

    def print_stats(self, metric=NLL_METRIC):
        final_types = np.where(self.betas > 1e-15)[0]
        mprint('RL estimated {0} types as follows {1}'.format(len(final_types), self.ranked_lists[final_types]))
        if metric == NLL_METRIC:
            final_kldiv = np.dot(self.observed_sales, np.ma.log(self.empirical_choice_probs).filled(0.) - np.ma.log(self.predict_insample_proba()).filled(0.))/np.sum(self.observed_sales)
            print(f'RL model KL divergence = {final_kldiv}')
            return final_kldiv
        elif metric == L1_METRIC:
            final_abs_error = np.sum(np.abs(self.empirical_choice_probs - self.predict_insample_proba()))
            print(f'RL model L1 error = {final_abs_error}')
            return final_abs_error

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'betas': self.betas,
            'ranked_lists': self.ranked_lists,
        }

    def __repr__(self):
        return '<Products: %s ; Ranked Lists: %s ; Betas: %s >' % (self.products, self.ranked_lists, self.betas)


class RankedListModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return ones(len(self.model.parameters_vector()))

    def amount_of_constraints(self):
        return 1

    def lower_bounds_over_constraints_vector(self):
        return array([ZERO_LOWER_BOUND])

    def upper_bounds_over_constraints_vector(self):
        return array([ONE_UPPER_BOUND])

    def non_zero_parameters_on_constraints_jacobian(self):
        return len(self.model.betas)

    def constraints_evaluator(self):
        def evaluator(x):
            return array([sum(x)])
        return evaluator

    def constraints_jacobian_evaluator(self):
        def jacobian_evaluator(x, flag):
            if flag:
                return zeros(len(self.model.betas)), array(range(len(self.model.betas)))
            else:
                return ones(len(self.model.betas))
        return jacobian_evaluator
