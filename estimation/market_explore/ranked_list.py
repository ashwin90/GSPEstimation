import itertools
from copy import deepcopy

import numpy as np

from estimation.market_explore import MarketExplorer
from optimization.linear import LinearProblem, LinearSolver
from utils import mprint, WITH_NO_CHOICE


class RankedListMarketExplorer(MarketExplorer):
    @classmethod
    def code(cls):
        raise NotImplementedError('Subclass responsibility')

    def explore_for(self, estimator, model, transactions):
        raise NotImplementedError('Subclass responsibility')


class NullMarketExplorer(MarketExplorer):
    @classmethod
    def code(cls):
        return 'null'

    def explore_for(self, estimator, model, transactions):
        return model.ranked_lists[0]


class RationalHeuristicMarketExplorer(RankedListMarketExplorer):
    @classmethod
    def code(cls):
        return 'hst'

    def explore_for(self, estimator, model, transactions):
        transactions_probs = model.predict_insample_proba()
        ranked_list = self.construction_phase(model, transactions, transactions_probs)  # Static variant
        # print('..Done with construction phase...')
        ranked_list = self.improvement_phase(model, transactions, ranked_list, transactions_probs)
        return ranked_list

    def improvement_phase(self, model, transactions, ranked_list, transactions_probs):
        curr_reward = self.reward_for(model, transactions, transactions_probs, ranked_list)
        for tid in range(model.num_transactions):
            head = model.chosen_products[tid]
            candidate = ranked_list[np.argmax(model.offered_products[tid, ranked_list])]
            if candidate != head:
                swapped = deepcopy(ranked_list)
                swapped[np.where(ranked_list == head)[0][0]] = candidate
                swapped[np.where(ranked_list == candidate)[0][0]] = head
                swapped_reward = self.reward_for(model, transactions, transactions_probs, swapped)
                if swapped_reward > curr_reward:
                    curr_reward = swapped_reward
                    ranked_list = swapped

        return ranked_list

    def reward_for(self, model, transactions, transactions_probs, candidate_list):
        return np.dot(model.observed_sales/transactions_probs, model.transactions_compatible_with(model.offered_products, model.chosen_products, candidate_list))

    def construction_phase(self, model, transactions, transactions_probs):
        sorted_transactions = np.argsort(transactions_probs)
        new_ranked_list = []
        while len(sorted_transactions):
            i = 0
            while model.chosen_products[sorted_transactions[i]] == 0 and len(new_ranked_list) < 1:
                i += 1
                if i >= len(sorted_transactions):
                    return []
            new_product = model.chosen_products[sorted_transactions[i]]
            new_ranked_list.append(new_product)
            vector_lambda = np.frompyfunc(lambda w: model.offered_products[w, new_product] != 1, 1, 1)
            sorted_transactions = sorted_transactions[vector_lambda(sorted_transactions).astype(np.bool)]

        new_ranked_list += ([] if 0 in new_ranked_list else [0])
        for product in filter(lambda x: x not in new_ranked_list, model.products):
            new_ranked_list.append(product)

        return np.array(new_ranked_list)


class MIPMarketExplorer(MarketExplorer):
    @classmethod
    def code(cls):
        return 'mip'

    def __init__(self):
        self.prev_soln = None

    def explore_for(self, estimator, model, transactions):
        problem = MIPMarketExploreLinearProblem(model, transactions)
        optimum, optimum_values_provider = LinearSolver().solve(problem, estimator.profiler(), self.prev_soln)
        # CHANGED to account for numerical errors
        obj_bound = 1
        if optimum > obj_bound or np.abs(optimum - obj_bound) < 1e-6:
            new_ranked_list = [0] * len(model.products)
            for j in model.products:
                position = sum(optimum_values_provider(['x_%s_%s' % (i, j) for i in model.products if i != j]))
                new_ranked_list[round(position)] = j
            mprint('Obj: {1} Needed: {2} New ranking: {0}'.format(new_ranked_list, optimum, obj_bound))
            self.prev_soln = optimum_values_provider(problem.variable_names())
            return np.array(new_ranked_list)

        print('MIP failed to find improving solution, obj: {0} needed: {1}'.format(optimum, obj_bound))
        return NullMarketExplorer().explore_for(estimator, model, transactions)


class MIPMarketExploreLinearProblem(LinearProblem):
    def __init__(self, model, transactions):
        self.model = model
        self.transactions = transactions

    def amount_of_variables(self):
        return len(self.objective_coefficients())

    def objective_coefficients(self):
        coefficients = [0.0 for _ in self.model.products for _ in self.model.products]
        normalized_sales = self.model.observed_sales/np.sum(self.model.observed_sales)
        return coefficients + list(normalized_sales / self.model.predict_insample_proba())

    def lower_bounds(self):
        lower = [0.0] * self.amount_of_variables()
        return lower

    def upper_bounds(self):
        return [1.0] * self.amount_of_variables()

    def variable_types(self):
        variable_types = 'B' * self.amount_of_variables()
        return variable_types

    def variable_names(self):
        variable_names = ['x_%s_%s' % (i, j) for i in self.model.products for j in self.model.products]
        return variable_names + ['w_%s' % t for t in range(self.model.num_transactions)]

    def constraints(self):
        return MIPMarketExploreConstraints(self.model, self.transactions).constraints()


class MIPMarketExploreConstraints(object):
    def __init__(self, model, transactions):
        self.model = model
        self.transactions = transactions

    def constraints(self):
        independent_terms = []
        names = []
        senses = []
        linear_expressions = []

        self.products_are_ordered_constraints(independent_terms, names, senses, linear_expressions)
        self.transitivity_in_order_constraints(independent_terms, names, senses, linear_expressions)
        if self.model.variant == WITH_NO_CHOICE:
            self.no_purchase_cannot_be_preferred_constraint(independent_terms, names, senses, linear_expressions)
        self.purchase_compatibility_constraints(independent_terms, names, senses, linear_expressions)

        return {'independent_terms': independent_terms, 'names': names,
                'senses': senses, 'linear_expressions': linear_expressions}

    def products_are_ordered_constraints(self, independent_terms, names, senses, linear_expressions):
        # i before j or j before i, never both at the same time
        for (i, j) in itertools.combinations(self.model.products, 2):
            independent_terms.append(1.0)
            names.append('%s_before_%s_or_%s_before_%s' % (i, j, j, i))
            senses.append('E')
            linear_expressions.append([['x_%s_%s' % (i, j), 'x_%s_%s' % (j, i)], [1.0, 1.0]])

    def transitivity_in_order_constraints(self, independent_terms, names, senses, linear_expressions):
        # transitivity constraints to ensure linear ordering among three products
        for (i, j, l) in itertools.permutations(self.model.products, 3):
            independent_terms.append(2.0)
            names.append('transitivity_for_%s_%s_%s' % (i, j, l))
            senses.append('L')
            linear_expressions.append([['x_%s_%s' % (j, i), 'x_%s_%s' % (i, l), 'x_%s_%s' % (l, j)], [1.0, 1.0, 1.0]])

    def no_purchase_cannot_be_preferred_constraint(self, independent_terms, names, senses, linear_expressions):
        # The no-purchase cannot be the preferred option
        independent_terms.append(1.0)
        names.append('no_purchase_is_not_preferred')
        senses.append('G')
        linear_expressions.append([['x_%s_0' % j for j in range(1, len(self.model.products))], [1.0 for _ in range(1, len(self.model.products))]])

    def purchase_compatibility_constraints(self, independent_terms, names, senses, linear_expressions):
        # If I want a profile compatible with transaction t,
        # then all products that are offered must come after the purchased product.
        for t, transaction in enumerate(self.model.offered_products):
            offer_set = transaction.nonzero()[0]
            chosen_prod = self.model.chosen_products[t]
            for i in set(offer_set) - {chosen_prod}:
                independent_terms.append(0.0)
                names.append('product_%s_is_worse_than_purchased_if_type_is_compatible_with_transaction_%s' % (i, t))
                senses += 'L'
                linear_expressions.append([['w_%s' % t, 'x_%s_%s' % (chosen_prod, i)], [1.0, -1.0]])
