from utils import Profiler


class EstimationMethod(object):
    """
        Can be Least Squares, Expectation Maximization or Maximum Likelihood for example.
    """
    @classmethod
    def estimators(cls):
        return NotImplementedError('Subclass responsibility')

    @classmethod
    def estimator_for_this(cls, model):
        for estimator_klass in cls.estimators():
            if estimator_klass.can_estimate(model):
                return estimator_klass()
        raise Exception('There is no %s estimator for the model %s.' % (cls.__name__, model.__class__.__name__))


class Estimator(object):
    """
        Estimates a model parameters based on historical transactions data.
    """
    def __init__(self):
        self._profiler = Profiler()

    def profiler(self):
        return self._profiler

    def can_estimate(self, model):
        raise NotImplementedError('Subclass responsibility')

    def estimate(self, model, transactions):
        raise NotImplementedError('Subclass responsibility')
