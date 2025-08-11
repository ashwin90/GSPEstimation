import itertools
from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed

from estimation.market_explore import MarketExplorer
from optimization.linear import LinearProblem, LinearSolver
from utils import mprint, WITH_NO_CHOICE


class GSPMarketExplorer(MarketExplorer):
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
        print('MIP failed to find improving solution')
        return model.customer_types[:1]


class HeuristicMarketExplorer(GSPMarketExplorer):
    @classmethod
    def code(cls):
        return 'hst'

    def explore_for(self, estimator, model, transactions):
        transactions_probs = model.predict_insample_proba()
        ranked_list = self.construction_phase(model, transactions, transactions_probs)  # Static variant
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

    def __init__(self, max_choice_index=2, max_irr_mass=1.0):
        self.prev_soln = dict()
        self.max_choice_index = max_choice_index
        self.max_irr_mass = max_irr_mass

    def search_ranking_for_given_k(self, choice_index, profiler, model, transactions):
        problem = MIPMarketExploreLinearProblem(model, transactions)
        problem.set_choice_index(choice_index)
        optimum, optimum_values_provider = LinearSolver().solve(problem, profiler, self.prev_soln[choice_index] if choice_index in self.prev_soln else None)
        new_ranked_list = [0] * len(model.products)
        for j in model.products:
            position = sum(optimum_values_provider(['x_%s_%s' % (i, j) for i in model.products if i != j]))
            new_ranked_list[round(position)] = j
        self.prev_soln[choice_index] = optimum_values_provider(problem.variable_names())
        return choice_index, optimum, new_ranked_list

    def explore_for(self, estimator, model, transactions):
        # pool = mp.Pool(5)
        choice_indices_to_search_over = range(1, 1 + self.max_choice_index)
        # results = pool.starmap_async(self.search_ranking_for_given_k, [(cidx, estimator.profiler(), model, transactions) for cidx in choice_indices_to_search_over]).get()
        results = Parallel(n_jobs=len(choice_indices_to_search_over))(delayed(self.search_ranking_for_given_k)(cidx, estimator.profiler(), model, transactions) for cidx in choice_indices_to_search_over)
        # pool.close()
        # pool.join()
        best_choice_index, best_opt, best_ranked_list = max(results, key=lambda w: w[1])
        obj_bound = 1
        if best_opt > obj_bound or np.abs(best_opt - obj_bound) < 1e-6:
            mprint('Obj: {2} Needed: {3} New ranking: {0} and choice index: {1}'.format(best_ranked_list,
                                                                                        best_choice_index,
                                                                                        best_opt, obj_bound))

            types_found = [(best_ranked_list, best_choice_index)]
            # determine if optimal solution has one type or two types
            if (self.max_irr_mass < 1) and (best_choice_index > 1):
                [(rational_type, rational_obj)] = [(w[2], w[1]) for w in results if w[0] == 1]
                if rational_obj + self.max_irr_mass*(best_opt - rational_obj) > obj_bound:
                    types_found.append((rational_type, 1))
                else:
                    print('Did not find a new type!')
                    return NullMarketExplorer().explore_for(estimator, model, transactions)

            if model.variant == WITH_NO_CHOICE:
                # ignore products after 0
                return map(lambda w: (np.array(w[0][:w[0].index(0)+1]), w[1]), types_found)
            else:
                return map(lambda w: (np.array(w[0]), w[1]), types_found)

        return NullMarketExplorer().explore_for(estimator, model, transactions)


class MIPMarketExploreLinearProblem(LinearProblem):
    def __init__(self, model, transactions):
        self.model = model
        self.transactions = transactions
        self.choice_index = 1

    def set_choice_index(self, choice_index):
        self.choice_index = choice_index

    def amount_of_variables(self):
        return len(self.objective_coefficients())

    def objective_coefficients(self):
        coefficients = [0.0 for _ in self.model.products for _ in self.model.products]
        # print(self.model.__repr__())
        normalized_sales = self.model.observed_sales/np.sum(self.model.observed_sales)
        return coefficients + list(normalized_sales/self.model.predict_insample_proba())

    def lower_bounds(self):
        return [0.0] * self.amount_of_variables()

    def upper_bounds(self):
        return [1.0] * self.amount_of_variables()

    def variable_types(self):
        return 'B' * self.amount_of_variables()

    def variable_names(self):
        variable_names = ['x_%s_%s' % (i, j) for i in self.model.products for j in self.model.products]
        return variable_names + ['w_%s' % t for t in range(self.model.num_transactions)]

    def constraints(self):
        return MIPMarketExploreConstraints(self.model, self.transactions, self.choice_index).constraints()


class MIPMarketExploreConstraints(object):
    def __init__(self, model, transactions, choice_index):
        self.model = model
        self.transactions = transactions
        self.choice_index = choice_index

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
        elif self.model.variant == 'gspx':
            self.no_purchase_cannot_be_preferred_constraint(independent_terms, names, senses, linear_expressions)
            self.purchase_compatibility_constraints_v2(independent_terms, names, senses, linear_expressions)
        else:
            self.purchase_compatibility_constraints_v3(independent_terms, names, senses, linear_expressions)

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
        # The no-purchase cannot be before the self.choice_index+1 th position
        independent_terms.append(self.choice_index)
        names.append('no_purchase_is_not_preferred')
        senses.append('G')
        linear_expressions.append([['x_%s_0' % j for j in range(1, len(self.model.products))], [1.0 for _ in range(1, len(self.model.products))]])

    # THIS IS THE VARIANT MENTIONED IN THE PAPER AND THE MOST GENERAL MODEL ALLOWING FOR CONSID SETS
    def purchase_compatibility_constraints(self, independent_terms, names, senses, linear_expressions):
        # If I want a profile compatible with transaction t,
        # then purchased product should be at 'choice_index' position
        for t in range(self.model.num_transactions):
            offered_products = tuple(self.model.offered_products[t].nonzero()[0])
            num_offered = len(offered_products)
            chosen_prod = self.model.chosen_products[t]
            chosen_offered_pairs = ['x_%s_%s' % (chosen_prod, j) for j in set(offered_products) - {chosen_prod}]
            independent_terms.append(0.0)
            names.append(
                'product_%s_is_compatible_with_transaction_%s_G' % (chosen_prod, t))
            senses.append('G')
            linear_expressions.append([['w_%s' % t] + chosen_offered_pairs,
                                       [self.choice_index - num_offered] + [1.0] * len(chosen_offered_pairs)])
            if chosen_prod > 0:
                independent_terms.append(num_offered - 1)
                names.append(
                    'product_%s_is_compatible_with_transaction_%s_L' % (chosen_prod, t))
                senses.append('L')
                linear_expressions.append([['w_%s' % t] + chosen_offered_pairs, [self.choice_index - 1] + [1.0]*len(chosen_offered_pairs)])

                independent_terms.append(0.0)
                names.append(
                    'product_%s_is_ranked_above_0_for_transaction_%s' % (chosen_prod, t))
                senses.append('L')
                linear_expressions.append([['w_%s' % t, 'x_%s_0' % chosen_prod], [1.0, -1.0]])

    # THIS IS A WEIRD VARIANT THAT I TRIED, NOT VERY INTUITIVE SO IGNORE
    def purchase_compatibility_constraints_v2(self, independent_terms, names, senses, linear_expressions):
        # If I want a profile compatible with transaction t,
        # then purchased product should be at 'choice_index' position
        for t in range(self.model.num_transactions):
            offered_products = tuple(self.model.offered_products[t].nonzero()[0])
            num_offered = len(offered_products)
            chosen_prod = self.model.chosen_products[t]
            chosen_offered_pairs = ['x_%s_%s' % (chosen_prod, j) for j in set(offered_products) - {chosen_prod}]
            if chosen_prod == 0:
                for xij in chosen_offered_pairs:
                    independent_terms.append(0.0)
                    names.append('proudcts_%s_are_compatible_with_transaction_%s' % (xij, t))
                    senses.append('L')
                    linear_expressions.append([['w_%s' % t, xij], [1.0, -1.0]])
            else:
                independent_terms.append(0.0)
                names.append(
                    'product_%s_is_ranked_above_0_for_transaction_%s' % (chosen_prod, t))
                senses.append('L')
                linear_expressions.append([['w_%s' % t, 'x_%s_0' % chosen_prod], [1.0, -1.0]])
                if num_offered < self.choice_index + 1:
                    for j in set(offered_products) - {chosen_prod, 0}:
                        independent_terms.append(0.0)
                        names.append('product_%s_is_compatible_with_transaction_%s' % (j, t))
                        senses.append('L')
                        linear_expressions.append([['w_%s' % t, 'x_%s_%s' % (j, chosen_prod)], [1.0, -1.0]])
                else:
                    independent_terms.append(0.0)
                    names.append(
                        'product_%s_is_compatible_with_transaction_%s_G' % (chosen_prod, t))
                    senses.append('G')
                    linear_expressions.append([['w_%s' % t] + chosen_offered_pairs,
                                               [self.choice_index - num_offered] + [1.0] * len(chosen_offered_pairs)])

                    independent_terms.append(num_offered - 1)
                    names.append('product_%s_is_compatible_with_transaction_%s_L' % (chosen_prod, t))
                    senses.append('L')
                    linear_expressions.append([['w_%s' % t] + chosen_offered_pairs, [self.choice_index - 1] + [1.0]*len(chosen_offered_pairs)])

    # THIS IS THE VARIANT WITH FULL RANKINGS WHERE 0 IS JUST ANOTHER PRODUCT
    def purchase_compatibility_constraints_v3(self, independent_terms, names, senses, linear_expressions):
        # If I want a profile compatible with transaction t,
        # then purchased product should be at 'choice_index' position
        for t in range(self.model.num_transactions):
            offered_products = tuple(self.model.offered_products[t].nonzero()[0])
            num_offered = len(offered_products)
            chosen_prod = self.model.chosen_products[t]
            chosen_offered_pairs = ['x_%s_%s' % (chosen_prod, j) for j in set(offered_products) - {chosen_prod}]
            if num_offered <= self.choice_index:
                for xij in chosen_offered_pairs:
                    independent_terms.append(1.0)
                    names.append('products_%s_are_compatible_with_transaction_%s' % (xij, t))
                    senses.append('L')
                    linear_expressions.append([['w_%s' % t, xij], [1.0, 1.0]])
            else:
                independent_terms.append(0.0)
                names.append('product_%s_is_compatible_with_transaction_%s_G' % (chosen_prod, t))
                senses.append('G')
                linear_expressions.append([['w_%s' % t] + chosen_offered_pairs,
                                           [self.choice_index - num_offered] + [1.0] * len(chosen_offered_pairs)])

                independent_terms.append(num_offered - 1)
                names.append('product_%s_is_compatible_with_transaction_%s_L' % (chosen_prod, t))
                senses.append('L')
                linear_expressions.append([['w_%s' % t] + chosen_offered_pairs, [self.choice_index - 1] + [1.0] * len(chosen_offered_pairs)])


