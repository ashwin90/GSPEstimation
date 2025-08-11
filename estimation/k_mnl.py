from estimation import Estimator
from models.k_mnl import GeneralizedMultinomialLogitModel


class GMNLEstimator(Estimator):
    def __init__(self):
        super(GMNLEstimator, self).__init__()

    def can_estimate(self, model):
        return GeneralizedMultinomialLogitModel == model.__class__

    def estimate(self, model, transactions):
        raise NotImplementedError('Subclass responsibility')

    def estimate_with_em(self, model, transactions):
        model = self.estimate(model, transactions)
        return model, int(self.profiler().duration())

