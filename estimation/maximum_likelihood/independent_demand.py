from estimation.maximum_likelihood import MaximumLikelihoodEstimator
from models.independent_demand import IndependentDemandModel


class IndependentDemandMaximumLikelihoodEstimator(MaximumLikelihoodEstimator):
    def can_estimate(self, model):
        return IndependentDemandModel == model.__class__
