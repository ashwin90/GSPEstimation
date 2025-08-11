from estimation.maximum_likelihood import MaximumLikelihoodEstimator
from models.nested_logit import NestedLogitModel


class NestedLogitMaximumLikelihoodEstimator(MaximumLikelihoodEstimator):
    def can_estimate(self, model):
        return NestedLogitModel == model.__class__
