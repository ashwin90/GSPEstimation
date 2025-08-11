from estimation.least_squares import LeastSquaresEstimator
from models.independent_demand import IndependentDemandModel


class IndependentDemandLeastSquaresEstimator(LeastSquaresEstimator):
    def can_estimate(self, model):
        return IndependentDemandModel == model.__class__
