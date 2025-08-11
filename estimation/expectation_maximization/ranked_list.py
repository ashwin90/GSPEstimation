import numpy as np

from estimation.expectation_maximization import ExpectationMaximizationEstimator
from estimation.ranked_list import RankedListEstimator
from models.ranked_list import RankedListModel


class RankedListExpectationMaximizationEstimator(ExpectationMaximizationEstimator, RankedListEstimator):
    def can_estimate(self, model):
        return RankedListModel == model.__class__

    def one_step(self, model, transactions):
        # x = [[0 for _ in transactions] for _ in model.ranked_lists]
        tr_probs = model.predict_insample_proba()
        posterior = model.observed_choice_indicators*model.betas
        pos_weights = model.observed_sales/tr_probs
        posterior = pos_weights[:, np.newaxis]*posterior
        mi = np.sum(posterior, 0)
        model.betas = mi/np.sum(mi)
        '''
        for t, transaction in enumerate(transactions):
            compatibles = model.ranked_lists_compatible_with(transaction)
            den = sum(map(lambda compatible: model.beta_for(compatible[0]), compatibles))
            for i, ranked_list in compatibles:
                x[i][t] = (transaction.sales*model.beta_for(i)) / den
                # x[i][t] = model.beta_for(i) / den

        
        m = [sum(x[i]) for i in range(len(model.ranked_lists))]
        model.set_betas([m[i] / sum(m) for i in range(len(model.ranked_lists))])
        '''
        return model
