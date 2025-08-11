from estimation.least_squares import LeastSquaresEstimator
from models import RandomChoiceModel


class RandomChoiceModelLeastSquaresEstimator(LeastSquaresEstimator):
    def can_estimate(self, model):
        return RandomChoiceModel == model.__class__

    def estimate(self, model, transactions):
        return model
