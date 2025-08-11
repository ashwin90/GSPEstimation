from estimation.maximum_likelihood import MaximumLikelihoodEstimator
from models.exponomial import ExponomialModel


class ExponomialMaximumLikelihoodEstimator(MaximumLikelihoodEstimator):
    def can_estimate(self, model):
        return ExponomialModel == model.__class__
