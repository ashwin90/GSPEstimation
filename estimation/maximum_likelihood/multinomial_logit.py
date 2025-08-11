from estimation.maximum_likelihood import MaximumLikelihoodEstimator
from models.multinomial_logit import MultinomialLogitModel


class MultinomialLogitMaximumLikelihoodEstimator(MaximumLikelihoodEstimator):
    def can_estimate(self, model):
        return MultinomialLogitModel == model.__class__
