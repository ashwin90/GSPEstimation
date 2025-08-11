from collections import defaultdict

from numpy import ones, zeros
from numpy.random import RandomState

from models import Model
from optimization.non_linear import Constraints
from utils import *


class GeneralizedStochasticPreferenceModel(Model):
    @classmethod
    def code(cls):
        return 'gsp'

    def populate(self, transactions):
        chosen_products = []
        offered_products = []
        observed_sales = []
        total_os_sales = defaultdict(int)
        num_prods = len(self.products)
        offer_set_dict = {}
        num_uniq_tx = 0
        for tr in transactions:
            os = np.zeros(num_prods, dtype=np.int)
            os[tr.offered_products] = 1
            chosen_product = tr.product
            product_sales = tr.sales
            tr_info = (chosen_product, tuple(os))
            total_os_sales[tuple(os)] += product_sales
            if tr_info not in offer_set_dict:
                offer_set_dict[tr_info] = num_uniq_tx
                num_uniq_tx += 1
                chosen_products.append(chosen_product)
                observed_sales.append(product_sales)
                offered_products.append(os)
            else:
                observed_sales[offer_set_dict[tr_info]] += product_sales

        self.offered_products = np.array(offered_products)
        self.chosen_products = np.array(chosen_products)
        self.observed_sales = np.array(observed_sales)
        self.num_transactions = self.offered_products.shape[0]
        self.init_observed_choice_indicators()
        self.empirical_choice_probs = np.zeros(self.num_transactions)
        for os_id in range(self.num_transactions):
            self.empirical_choice_probs[os_id] = self.observed_sales[os_id]/total_os_sales[tuple(self.offered_products[os_id])]

    def init_observed_choice_indicators(self):
        self.observed_choice_indicators = np.zeros((self.num_transactions, self.num_customer_types), dtype=np.int)
        for tidx, (ranking, cidx) in enumerate(self.customer_types):
            self.observed_choice_indicators[:, tidx] = self.transactions_compatible_with(self.offered_products, self.chosen_products, ranking, cidx)

    def init_from_parsed_data(self, offered_products, chosen_products, observed_sales, empirical_choice_probs):
        self.offered_products = offered_products
        self.chosen_products = chosen_products
        self.observed_sales = observed_sales
        self.num_transactions = self.offered_products.shape[0]
        self.init_observed_choice_indicators()
        self.empirical_choice_probs = empirical_choice_probs

    @classmethod
    def default_naive(cls, products, variant=WITHOUT_NO_CHOICE, ranked_lists=None, choice_indices=None):
        if not ranked_lists:
            prng = RandomState(3)
            products_set = set(products) - {0}
            if variant == WITH_NO_CHOICE:
                ranked_lists = [[0]]
            else:
                ranked_lists = [products]
            for first_product in products_set:
                remaining_products = list(products_set - {first_product})
                if variant == WITH_NO_CHOICE:
                    new_ranking = [first_product] + [0]
                else:
                    new_ranking = [first_product] + [0] + list(prng.permutation(remaining_products))
                ranked_lists.append(new_ranking)

        if not choice_indices:
            choice_indices = [1]*len(ranked_lists)

        customer_types = []
        for rid, r in enumerate(ranked_lists):
            customer_types.append((np.array(r), choice_indices[rid]))

        betas = generate_n_equal_numbers_that_sum_one(len(ranked_lists))
        return cls(products, customer_types, betas, variant)

    @classmethod
    def default_random(cls, products, variant, ranked_lists, choice_indices):
        betas = generate_n_random_numbers_that_sum_one(len(ranked_lists))
        customer_types = []
        for rid, r in enumerate(ranked_lists):
            customer_types.append((np.array(r), choice_indices[rid]))
        return cls(products, customer_types, betas, variant)

    def __init__(self, products, customer_types, betas, variant=WITHOUT_NO_CHOICE):
        super(GeneralizedStochasticPreferenceModel, self).__init__(products)
        if len(betas) != len(customer_types):
            info = (len(betas), len(customer_types))
            raise Exception('Amount of betas (%s) should be equal to that of customer types (%s).' % info)
        self.betas = np.array(betas)
        self.customer_types = customer_types
        self.ranked_list_to_choice_index_mapping = defaultdict(list)
        self.rational_type_indicators = []
        for r, cid in customer_types:
            self.ranked_list_to_choice_index_mapping[tuple(r)].append(cid)
            self.rational_type_indicators.append(cid == 1)
        self.num_customer_types = len(self.customer_types)
        self.variant = variant
        self.rational_type_indicators = np.array(self.rational_type_indicators)
        # to be populated later
        self.offered_products = None
        self.chosen_products = None
        self.observed_sales = None
        self.num_transactions = None
        self.observed_choice_indicators = None
        self.empirical_choice_probs = None

    def set_variant(self, variant):
        self.variant = variant

    def predict_insample_proba(self):
        return np.dot(self.observed_choice_indicators, self.betas)

    def predict_proba(self, membership, chosen_products):
        probs = np.zeros(membership.shape[0])
        for tid, gsp_type in enumerate(self.customer_types):
            # gsp_type[0] is the ranked list, gsp_type[1] is the choice_index
            probs += self.betas[tid]*self.transactions_compatible_with(membership, chosen_products, gsp_type[0], gsp_type[1])
        return probs

    def log_likelihood_for(self, transactions):
        return np.dot(self.observed_sales, np.ma.log(self.predict_insample_proba()).filled(0.))

    def get_chosen_prod(self, consid_set_binary, k, ranking):
        cid = k - 1
        consid_set = consid_set_binary.nonzero()[0]
        assert len(consid_set) > 0, embed()
        if self.variant == GPT_VARIANT:
            return consid_set[cid] if cid < len(consid_set) else np.where(ranking == 0)[0][0]
        else:
            return consid_set[cid] if cid < len(consid_set) else consid_set[-1]

    def transactions_compatible_with(self, membership, chosen_products, ranked_list, choice_index):
        return ranked_list[np.apply_along_axis(self.get_chosen_prod, 1, membership[:, ranked_list], k=choice_index, ranking=ranked_list)] == chosen_products
        # return ranked_list[np.argmax(self.offered_products[:, ranked_list], axis=1)] == self.chosen_products

    def products_chosen_by_type(self, membership, ranked_list, choice_index):
        return ranked_list[np.apply_along_axis(self.get_chosen_prod, 1, membership[:, ranked_list], k=choice_index)]

    def add_ranked_list_and_index(self, ranked_list, choice_index, max_irr_mass):
        rl_key = tuple(ranked_list)
        if rl_key in self.ranked_list_to_choice_index_mapping and choice_index in self.ranked_list_to_choice_index_mapping[rl_key]:
            return

        percentage = 1.0 / (self.num_customer_types + 1.0)
        new_beta = [percentage]
        self.betas = np.concatenate(((1.0 - percentage)*self.betas, new_beta))
        self.ranked_list_to_choice_index_mapping[rl_key].append(choice_index)
        self.customer_types.append((ranked_list, choice_index))
        self.num_customer_types += 1
        self.observed_choice_indicators = np.hstack((self.observed_choice_indicators, self.transactions_compatible_with(self.offered_products, self.chosen_products, ranked_list, choice_index)[:, np.newaxis]))
        self.rational_type_indicators = np.append(self.rational_type_indicators, choice_index == 1)
        self.reduce_irrational_mass(max_irr_mass)

    def reduce_irrational_mass(self, max_irr_mass):
        irrational_sum = np.sum(self.betas[~self.rational_type_indicators])
        if irrational_sum > max_irr_mass:
            self.betas[~self.rational_type_indicators] *= max_irr_mass / irrational_sum
            self.betas[self.rational_type_indicators] *= (1 - max_irr_mass) / (1 - irrational_sum)

    def print_stats(self, metric=NLL_METRIC):
        final_types = np.where(self.betas > 1e-15)[0]
        num_irr_types = 0.
        irr_types_mass = 0.
        for tid in final_types:
            if self.customer_types[tid][1] > 1:
                num_irr_types += 1
                irr_types_mass += self.betas[tid]
            mprint('GSP recovered type {0} with mass {1}'.format(self.customer_types[tid], self.betas[tid]))

        mprint('GSP estimated {0} irrational types with mass {1:.3f} out of {2} total types'.format(num_irr_types, irr_types_mass, len(final_types)))
        predicted_probs = self.predict_insample_proba()
        # print('Predicted Probs: {0}'.format(predicted_probs))
        if metric == NLL_METRIC:
            final_kldiv = np.dot(self.observed_sales, np.ma.log(self.empirical_choice_probs).filled(0.) - np.ma.log(predicted_probs).filled(0.))/np.sum(self.observed_sales)
            print(f'GSP({len(self.betas)}) KL divergence = {final_kldiv}')
            return final_kldiv, irr_types_mass
        elif metric == L1_METRIC:
            final_abs_error = np.sum(np.abs(self.empirical_choice_probs - self.predict_insample_proba()))
            print(f'GSP({len(self.betas)}) L1 error = {final_abs_error}')
            return final_abs_error, irr_types_mass

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'betas': self.betas,
            'ranked_list_to_choice_index_mapping': self.ranked_list_to_choice_index_mapping,
            'num_customer_types': self.num_customer_types
        }


class GeneralizedStochasticPreferenceModelConstraints(Constraints):
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
