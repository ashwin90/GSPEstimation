import numpy as np
import pandas as pd
from IPython import embed

from estimation.expectation_maximization import ExpectationMaximizationEstimator
from estimation.k_mnl import GMNLEstimator
from models.k_mnl import GeneralizedMultinomialLogitModel


class GMNLExpectationMaximizationEstimator(ExpectationMaximizationEstimator, GMNLEstimator):
    def can_estimate(self, model):
        return GeneralizedMultinomialLogitModel == model.__class__

    def one_step(self, model, transactions):
        mnl_probs, tr_probs, prod_matix = model.predict_insample_proba(True)
        posterior_mnl_type_probs = model.betas[0]*mnl_probs/tr_probs
        # compute posterior probabilities for highest util product in each offer-set
        posterior_prod_with_highest_util = prod_matix/np.sum(prod_matix, 1, keepdims=True)
        # update betas
        pos_weights = model.observed_sales*posterior_mnl_type_probs
        model.betas[0] = np.sum(pos_weights)/np.sum(model.observed_sales)
        assert 0 <= model.betas[0] <= 1, 'Error in updating GMNL proportions'
        model.betas[0] = max(model.betas[0], 1 - model.irrat_prop_ub)
        model.betas[1] = 1 - model.betas[0]
        # update prod_utils
        # compute weights which get multiplied by observed_sales before fitting mnl model
        dummy_weights = (1 - posterior_mnl_type_probs)[:, np.newaxis]*posterior_prod_with_highest_util
        dummy_weights = dummy_weights[model.em_transaction_indices.nonzero()]
        sales_weights = np.concatenate((posterior_mnl_type_probs, dummy_weights, dummy_weights))
        actual_sales = sales_weights*model.em_observed_sales
        # compute total weighted sales of each product
        sales_df = pd.DataFrame({'sales': actual_sales, 'prods': model.em_chosen_products})
        total_sales_by_prod = np.ravel(sales_df.groupby('prods').sum())
        # implement single MM update (GEM)
        predicted_probs = model.predict_mnl_proba(model.em_offered_products)
        predicted_sales = np.sum(actual_sales[:, np.newaxis]*predicted_probs, 0)
        '''
        try:
            assert np.all(total_sales_by_prod > 0)
        except AssertionError:
            print('Product with zero sales!')
            embed()
        '''
        if len(total_sales_by_prod) < len(model.prod_utils):
            print("Product never offered!")
            embed()
        non_zero_sales = total_sales_by_prod > 0
        model.prod_utils[non_zero_sales] += np.log(total_sales_by_prod[non_zero_sales]/predicted_sales[non_zero_sales])
        model.prod_utils -= model.prod_utils[0]
        return model
