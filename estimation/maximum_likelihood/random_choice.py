from estimation.maximum_likelihood import MaximumLikelihoodEstimator
from models.random_choice import RandomChoiceModel


class RandomChoiceModelMaximumLikelihoodEstimator(MaximumLikelihoodEstimator):
    def can_estimate(self, model):
        return RandomChoiceModel == model.__class__

    def estimate(self, model, transactions):
        return model
