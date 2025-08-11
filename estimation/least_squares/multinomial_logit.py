from estimation.least_squares import LeastSquaresEstimator
from models.multinomial_logit import MultinomialLogitModel


class MultinomialLogitLeastSquaresEstimator(LeastSquaresEstimator):
    def can_estimate(self, model):
        return MultinomialLogitModel == model.__class__
