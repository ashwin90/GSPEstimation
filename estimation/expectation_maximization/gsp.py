import numpy as np

from estimation.expectation_maximization import ExpectationMaximizationEstimator
from estimation.gsp import GSPEstimator
from models.gsp import GeneralizedStochasticPreferenceModel


class GSPExpectationMaximizationEstimator(ExpectationMaximizationEstimator, GSPEstimator):
    def can_estimate(self, model):
        return GeneralizedStochasticPreferenceModel == model.__class__

    def one_step(self, model, transactions):
        tr_probs = model.predict_insample_proba()
        posterior = model.observed_choice_indicators*model.betas
        pos_weights = model.observed_sales/tr_probs
        posterior = pos_weights[:, np.newaxis]*posterior
        mi = np.sum(posterior, 0)
        # need indicators for which type are rational and which irrational
        rational_betas = mi[model.rational_type_indicators]
        irrational_betas = mi[~model.rational_type_indicators]
        irrational_sum = np.sum(irrational_betas)
        total_sum = np.sum(mi)
        thresh = irrational_sum/total_sum
        delta = min(self.market_explorer.max_irr_mass, thresh)
        rational_betas /= (total_sum - irrational_sum)
        irrational_betas /= irrational_sum
        model.betas = np.zeros_like(mi)
        model.betas[model.rational_type_indicators] = (1 - delta) * rational_betas
        model.betas[~model.rational_type_indicators] = delta*irrational_betas
        assert np.round(np.abs(model.betas.sum() - 1), 7) == 0, 'Proportions not adding to 1'
        return model

