from estimation.least_squares import LeastSquaresEstimator
from models.exponomial import ExponomialModel


class ExponomialLeastSquaresEstimator(LeastSquaresEstimator):
    def can_estimate(self, model):
        return ExponomialModel == model.__class__
