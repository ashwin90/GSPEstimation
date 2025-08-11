from copy import deepcopy

import numpy as np

from models import Model
from utils import generate_n_equal_numbers_that_sum_one, embed, WITH_NO_CHOICE, WITHOUT_NO_CHOICE


class GeneralizedMultinomialLogitModel(Model):
    @classmethod
    def code(cls):
        return 'k_mnl'

    def add_dummy_transactions_for_estimation(self):
        self.em_offered_products = deepcopy(self.offered_products)
        self.em_observed_sales = deepcopy(self.observed_sales)
        self.em_chosen_products = deepcopy(self.chosen_products)
        num_prods = len(self.products)
        self.em_transaction_indices = []
        for os_id, os in enumerate(self.offered_products):
            os_prods = os.nonzero()[0]
            not_chosen_prods = np.delete(os_prods, np.argwhere(os_prods == self.chosen_products[os_id]))
            os_to_add = np.zeros(num_prods, dtype=np.int)
            os_to_add[not_chosen_prods] = 1
            self.em_transaction_indices.append(os_to_add)
            for prod_id in not_chosen_prods:
                dummy_os = np.zeros(num_prods)
                dummy_os[os_prods] = 1
                self.em_offered_products = np.append(self.em_offered_products, np.atleast_2d(dummy_os), 0)
                self.em_observed_sales = np.append(self.em_observed_sales, self.observed_sales[os_id])
                self.em_chosen_products = np.append(self.em_chosen_products, prod_id)

        for os_id, os in enumerate(self.offered_products):
            os_prods = os.nonzero()[0]
            not_chosen_prods = np.delete(os_prods, np.argwhere(os_prods == self.chosen_products[os_id]))
            for prod_id in not_chosen_prods:
                dummy_os = np.zeros(num_prods)
                dummy_os[os_prods] = 1
                dummy_os[prod_id] = 0
                self.em_offered_products = np.append(self.em_offered_products, np.atleast_2d(dummy_os), 0)
                self.em_observed_sales = np.append(self.em_observed_sales, self.observed_sales[os_id])
                self.em_chosen_products = np.append(self.em_chosen_products, self.chosen_products[os_id])

        self.em_transaction_indices = np.array(self.em_transaction_indices)

    def init_from_parsed_data(self, offered_products, chosen_products, observed_sales, empirical_choice_probs):
        self.offered_products = offered_products
        self.chosen_products = chosen_products
        self.observed_sales = observed_sales
        self.num_transactions = self.offered_products.shape[0]
        self.add_dummy_transactions_for_estimation()
        self.empirical_choice_probs = empirical_choice_probs

    @classmethod
    def default_random(cls, products, is_mnl=False, irrat_prop_ub=1.0, variant=WITHOUT_NO_CHOICE):
        prod_utils = np.zeros(len(products))
        if is_mnl:
            betas = [1., 0.]
        else:
            betas = generate_n_equal_numbers_that_sum_one(2)
            # betas = generate_n_random_numbers_that_sum_one(2)
        return cls(products, prod_utils, betas, irrat_prop_ub, variant)

    def __init__(self, products, prod_utils, betas, irrat_prop_ub, variant):
        super(GeneralizedMultinomialLogitModel, self).__init__(products)
        if len(prod_utils) != len(products):
            info = (len(prod_utils), len(products))
            raise Exception('Amount of prod_utils (%s) should be equal to that of products (%s).' % info)
        self.betas = np.array(betas)
        self.prod_utils = prod_utils
        # to be populated later
        self.offered_products = None
        self.chosen_products = None
        self.observed_sales = None
        self.num_transactions = None
        self.em_observed_sales = None
        self.em_offered_products = None
        self.em_chosen_products = None
        self.em_transaction_indices = None
        self.empirical_choice_probs = None
        self.variant = variant
        # UB on proportion of irrational type (=0.5 to enforce regularity on the GMNL model)
        self.irrat_prop_ub = irrat_prop_ub

    def predict_proba(self, membership, chosen_products):
        return self.predict_insample_proba(False, membership, chosen_products)

    def predict_insample_proba(self, in_sample, membership=None, chosen_products=None):
        assert len(self.betas) == 2, 'Estimation supported for GMNL(2) model only!'
        X = membership if membership is not None else self.offered_products
        C = chosen_products if chosen_products is not None else self.chosen_products
        num_os = X.shape[0]
        prod_wts = self.prod_utils - np.max(self.prod_utils)
        prod_wts_by_os = np.exp(prod_wts)*X
        # chosen_prod_wts = prod_wts_by_os[range(num_os), chosen_products]
        row_sums = np.sum(prod_wts_by_os, 1) # w(S_t) in the paper
        subtracted_row_sums = row_sums[:, np.newaxis] - prod_wts_by_os
        mnl_probs = prod_wts_by_os/row_sums[:, np.newaxis]
        chosen_mnl_probs = mnl_probs[range(num_os), C]
        product_matrix = prod_wts_by_os/subtracted_row_sums
        product_matrix[range(num_os), C] = 0
        # mnl_probs_2 = np.sum(product_matrices, 1) - chosen_prod_wts/(row_sums-chosen_prod_wts)
        pred_probs = self.betas[0]*chosen_mnl_probs + self.betas[1]*chosen_mnl_probs*np.sum(product_matrix, 1)
        if in_sample:
            return chosen_mnl_probs, pred_probs, product_matrix
        else:
            return pred_probs

    def predict_mnl_proba(self, membership):
        prod_wts = self.prod_utils - np.max(self.prod_utils)
        prod_wts_by_os = np.exp(prod_wts)*membership
        row_sums = np.sum(prod_wts_by_os, 1)  # w(S_t) in the paper
        choice_probs = prod_wts_by_os/row_sums[:, np.newaxis]
        assert np.all(np.around(np.sum(choice_probs, 1) - np.ones(membership.shape[0]), 7) == 0), embed()
        # 'Choice probs do not add to 1'
        return choice_probs

    def predict_2mnl_proba(self, membership):
        prod_wts = self.prod_utils - np.max(self.prod_utils)
        prod_wts_by_os = np.exp(prod_wts) * membership
        row_sums = np.sum(prod_wts_by_os, 1)  # w(S_t) in the paper
        mnl_probs = prod_wts_by_os / row_sums[:, np.newaxis]
        if self.variant == WITH_NO_CHOICE:
            subtracted_row_sums = row_sums[:, np.newaxis] - prod_wts_by_os[:, 1:]  # w(S_t) - e^{v_i} in paper
            product_matrix = prod_wts_by_os[:, 1:] / subtracted_row_sums  # e^{v_i}/(w(S_t) - e^{v_i}) in paper
            type_2_probs = mnl_probs[:, 1:] * (np.sum(product_matrix, 1, keepdims=True) - product_matrix)
            # compute no-purchase probs
            type_2_probs = np.hstack((1 - np.sum(type_2_probs, 1, keepdims=True), type_2_probs))
            assert np.all(type_2_probs >= 0), (print('Choice probs do not sum to 1'), embed())
        else:
            valid_rows = np.round(np.abs(np.max(mnl_probs, 1) - 1), 7) > 0
            type_2_probs = np.zeros_like(mnl_probs)
            second_stage_membership = membership[~valid_rows].copy()
            second_stage_membership[np.arange(second_stage_membership.shape[0]), np.argmax(mnl_probs[~valid_rows], 1)] = 0
            type_2_probs[~valid_rows] = self.predict_mnl_proba(second_stage_membership)
            subtracted_row_sums = row_sums[valid_rows][:, np.newaxis] - prod_wts_by_os[valid_rows]  # w(S_t) - e^{v_i} in paper
            # product_matrix = np.where(subtracted_row_sums > 0, prod_wts_by_os / subtracted_row_sums, 1)  # e^{v_i}/(w(S_t) - e^{v_i}) in paper
            product_matrix = prod_wts_by_os[valid_rows] / subtracted_row_sums  # e^{v_i}/(w(S_t) - e^{v_i}) in paper
            type_2_probs[valid_rows] = mnl_probs[valid_rows] * (np.sum(product_matrix, 1, keepdims=True) - product_matrix)
            assert np.all(np.around(np.sum(type_2_probs, 1) - np.ones(membership.shape[0]), 7) == 0), (print('Choice probs do not sum to 1'), embed())

        return type_2_probs

    @staticmethod
    def array_to_dict(arr):
        # Create an array of row indices
        row_indices = np.arange(arr.shape[0])

        # Convert each row to a tuple and zip with row indices
        row_tuples = [tuple(row) for row in arr]
        result_dict = dict(zip(row_tuples, row_indices))

        return result_dict

    # add function that computes choice probabilities given only membership matrix
    # TODO: assumes no-purchase option is present and membership contains all possible offer-sets
    def predict_gmnl_proba(self, membership):
        max_k = len(self.betas)
        choice_prob_dict = dict()
        choice_prob_dict[1] = self.predict_mnl_proba(membership)
        choice_prob_dict[2] = self.predict_2mnl_proba(membership)
        choice_probs = self.betas[0] * choice_prob_dict[1] + self.betas[1] * choice_prob_dict[2]
        if max_k == 2:
            return choice_probs
        else:
            assert self.variant == WITH_NO_CHOICE, 'TODO: update for without no-choice'
            membership_dict = self.array_to_dict(membership)
            for k in range(3, max_k+1):
                choice_prob_dict[k] = np.zeros_like(membership, dtype=float)
                for os_id, os in enumerate(membership):
                    if k >= np.sum(os):
                        continue
                    for offered_prod in os[1:].nonzero()[0]:
                        choice_prob_dict[k][os_id, offered_prod+1] = self.compute_gmnl_choice_probs_recursively(os, offered_prod+1, k, choice_prob_dict, membership_dict)

                choice_prob_dict[k][:, 0] = 1 - np.sum(choice_prob_dict[k], 1)
                assert np.all(choice_prob_dict[k] >= 0), 'Choice probs do not add to 1'
                choice_probs += self.betas[k-1]*choice_prob_dict[k]

            return choice_probs

    def compute_gmnl_choice_probs_recursively(self, offer_set, i, type_id, choice_prob_dict, membership_dict):
        # using Proposition 6
        offered_prods = offer_set.nonzero()[0]
        mnl_prob_index = membership_dict[tuple(offer_set)]
        choice_prob = 0
        # print(f"offer-set={offer_set}")
        for j in offered_prods:
                if j != i and j != 0:
                    os_copy = np.copy(offer_set)
                    os_copy[j] = 0
                    gmnl_prob_index = membership_dict[tuple(os_copy)]
                    choice_prob += choice_prob_dict[1][mnl_prob_index][j]*choice_prob_dict[type_id-1][gmnl_prob_index][i]

        return choice_prob

    def log_likelihood_for(self, transactions):
        return np.dot(self.observed_sales, np.ma.log(self.predict_insample_proba(False)).filled(0.)) / np.sum(self.observed_sales)

    def print_stats(self):
        print('Type proportions {0}'.format(self.betas))
        print('Product utils {0}'.format(self.prod_utils))

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'betas': self.betas,
            'product utils': self.prod_utils
        }

