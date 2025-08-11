import itertools

import numpy as np

from estimation.market_explore import MarketExplorer
from optimization.linear import LinearProblem, LinearSolver


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
        return model.ranked_lists[0], model.choice_indices[0]


class MIPMarketExplorer(MarketExplorer):
    @classmethod
    def code(cls):
        return 'mip'

    def __init__(self):
        self.prev_soln = None

    def explore_for(self, estimator, model, transactions):
        problem = MIPMarketExploreLinearProblem(model, transactions)
        optimum, best_opt_values = LinearSolver().solve(problem, estimator.profiler(), self.prev_soln)
        self.prev_soln = best_opt_values(problem.variable_names())
        if optimum > np.sum(model.observed_sales):
            new_ranked_list = [0] * len(model.products)
            for j in model.products:
                position = sum(best_opt_values(['x_%s_%s' % (i, j) for i in model.products if i != j]))
                new_ranked_list[int(position)] = j
            new_choice_index = int(best_opt_values(['k'])[0])
            print('New ranking: {0} and choice index: {1}'.format(new_ranked_list, new_choice_index))
            if model.variant == 'gspc':
                # ignore products after 0
                return np.array(new_ranked_list[:new_ranked_list.index(0)+1]), new_choice_index
            else:
                return np.array(new_ranked_list), new_choice_index

        return NullMarketExplorer().explore_for(estimator, model, transactions)


class MIPMarketExploreLinearProblem(LinearProblem):
    def __init__(self, model, transactions):
        self.model = model
        self.transactions = transactions

    def amount_of_variables(self):
        return len(self.objective_coefficients())

    def objective_coefficients(self):
        # xij coefficients
        coefficients = [0.0 for _ in self.model.products for _ in self.model.products]
        # wt coefs
        coefficients += list(self.model.observed_sales/self.model.predict_insample_proba())
        # zt coefs
        coefficients += [0.0]*self.model.num_transactions
        if self.model.variant == 'gsp':
            # delta coefs
            coefficients += [0.0] * self.model.num_transactions
        # add k coef
        return coefficients + [0.0]

    def lower_bounds(self):
        # 1 is for the choice index 'k'
        return [0.0] * (self.amount_of_variables() - 1) + [1]

    def upper_bounds(self):
        num_prods = len(self.model.products)
        x_ij_bounds = [1.0]*(num_prods**2)
        w_t_bounds = [1.0]*self.model.num_transactions
        z_t_bounds = [num_prods - 1]*self.model.num_transactions
        if self.model.variant == 'gsp':
            delta_t_bounds = [1.0]*self.model.num_transactions
        else:
            delta_t_bounds = []
        k_ub = [num_prods - 1]
        return x_ij_bounds + w_t_bounds + z_t_bounds + delta_t_bounds + k_ub

    def variable_types(self):
        num_prods = len(self.model.products)
        x_ij_types = ['B'] * (num_prods ** 2)
        w_t_types = ['B'] * self.model.num_transactions
        z_t_types = ['C'] * self.model.num_transactions
        if self.model.variant == 'gsp':
            delta_t_types = ['B'] * self.model.num_transactions
        else:
            delta_t_types = []
        k_type = ['I']
        return x_ij_types + w_t_types + z_t_types + delta_t_types + k_type

    def variable_names(self):
        variable_names = ['x_%s_%s' % (i, j) for i in self.model.products for j in self.model.products]
        variable_names += ['w_%s' % t for t in range(self.model.num_transactions)]
        variable_names += ['z_%s' % t for t in range(self.model.num_transactions)]
        if self.model.variant == 'gsp':
            variable_names += ['delta_%s' % t for t in range(self.model.num_transactions)]

        variable_names.append('k')
        return variable_names

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
        if self.model.variant == 'gspc':
            self.no_purchase_cannot_be_preferred_constraint(independent_terms, names, senses, linear_expressions)
            self.purchase_compatibility_constraints_gspc_with_k(independent_terms, names, senses, linear_expressions)
        else:
            self.purchase_compatibility_constraints_gsp_with_k(independent_terms, names, senses, linear_expressions)

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
        independent_terms.append(0.0)
        names.append('no_purchase_is_not_preferred')
        senses.append('G')
        linear_expressions.append([['x_%s_0' % j for j in range(1, len(self.model.products))] + ['k'], [1.0]*(len(self.model.products) - 1) + [-1]])

    # THIS IS THE VARIANT MENTIONED IN THE PAPER AND THE MOST GENERAL MODEL ALLOWING FOR CONSID SETS
    def purchase_compatibility_constraints_gspc_with_k(self, independent_terms, names, senses, linear_expressions):
        # If I want a profile compatible with transaction t,
        # then purchased product should be at 'choice_index' position
        num_products = len(self.model.products) - 1
        for t in range(self.model.num_transactions):
            offered_products = tuple(self.model.offered_products[t].nonzero()[0])
            num_offered = len(offered_products)
            chosen_prod = self.model.chosen_products[t]
            chosen_offered_pairs = ['x_%s_%s' % (chosen_prod, j) for j in set(offered_products) - {chosen_prod}]

            independent_terms.append(0.0)
            names.append('product_%s_is_compatible_with_transaction_%s_L' % (chosen_prod, t))
            senses.append('L')
            linear_expressions.append([['w_%s' % t, 'z_%s' % t] + chosen_offered_pairs,
                                       [num_offered - 1, -1] + [-1.0] * len(chosen_offered_pairs)])

            independent_terms.append(0.0)
            names.append('z_%s_first_constraint' % t)
            senses.append('L')
            linear_expressions.append([['z_%s' %t, 'w_%s' % t],
                                       [1, - num_products]])

            independent_terms.append(-1)
            names.append('z_%s_second_constraint' % t)
            senses.append('L')
            linear_expressions.append([['z_%s' % t, 'k'],
                                       [1, -1]])

            independent_terms.append(-num_products - 1)
            names.append('z_%s_third_constraint' % t)
            senses.append('G')
            linear_expressions.append([['z_%s' % t, 'k', 'w_%s' % t],
                                       [1, -1, num_products]])

            if chosen_prod > 0:
                independent_terms.append(num_offered - 1)
                names.append(
                    'product_%s_is_compatible_with_transaction_%s_L' % (chosen_prod, t))
                senses.append('L')
                linear_expressions.append([['z_%s' % t] + chosen_offered_pairs, [1.0] + [1.0]*len(chosen_offered_pairs)])

                independent_terms.append(0.0)
                names.append(
                    'product_%s_is_ranked_above_0_for_transaction_%s' % (chosen_prod, t))
                senses.append('L')
                linear_expressions.append([['w_%s' % t, 'x_%s_0' % chosen_prod], [1.0, -1.0]])

    # THIS IS THE VARIANT WITH FULL RANKINGS WHERE 0 IS JUST ANOTHER PRODUCT
    def purchase_compatibility_constraints_gsp_with_k(self, independent_terms, names, senses, linear_expressions):
        # If I want a profile compatible with transaction t,
        # then purchased product should be at 'choice_index' position
        num_products = len(self.model.products) - 1
        for t in range(self.model.num_transactions):
            offered_products = tuple(self.model.offered_products[t].nonzero()[0])
            num_offered = len(offered_products)
            chosen_prod = self.model.chosen_products[t]
            chosen_offered_pairs = ['x_%s_%s' % (chosen_prod, j) for j in set(offered_products) - {chosen_prod}]
            independent_terms.append(num_offered-1)
            names.append('product_%s_is_compatible_with_transaction_%s_L' % (chosen_prod, t))
            senses.append('L')
            linear_expressions.append([['w_%s' % t, 'z_%s' % t] + chosen_offered_pairs,
                                       [num_offered - 1, -1] + [1.0] * len(chosen_offered_pairs)])

            independent_terms.append(1-num_offered)
            names.append('product_%s_is_compatible_with_transaction_%s_G' % (chosen_prod, t))
            senses.append('G')
            linear_expressions.append([['w_%s' % t, 'z_%s' % t] + chosen_offered_pairs,
                                       [1 - num_offered, -1] + [1.0] * len(chosen_offered_pairs)])

            independent_terms.append(num_offered)
            names.append('delta_%s_first_constraint' % t)
            senses.append('L')
            linear_expressions.append([['k', 'delta_%s' % t],
                                       [1, num_offered - num_products]])

            independent_terms.append(-1)
            names.append('delta_%s_second_constraint' % t)
            senses.append('L')
            linear_expressions.append([['k', 'delta_%s' % t],
                                       [-1, num_offered - 1]])

            independent_terms.append(num_offered)
            names.append('delta_%s_third_constraint' % t)
            senses.append('L')
            linear_expressions.append([['z_%s' % t, 'k', 'delta_%s' % t],
                                       [1, 1, num_offered - num_products]])

            independent_terms.append(num_offered - 1)
            names.append('delta_%s_fourth_constraint' % t)
            senses.append('L')
            linear_expressions.append([['z_%s' % t, 'delta_%s' % t],
                                       [1, num_offered - 1]])

            independent_terms.append(num_offered)
            names.append('z_%s_lb' % t)
            senses.append('G')
            linear_expressions.append([['z_%s' % t, 'k'],
                                       [1, 1]])
