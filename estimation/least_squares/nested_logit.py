from estimation.least_squares import LeastSquaresEstimator
from models.nested_logit import NestedLogitModel


class NestedLogitLeastSquaresEstimator(LeastSquaresEstimator):
    def can_estimate(self, model):
        return NestedLogitModel == model.__class__
