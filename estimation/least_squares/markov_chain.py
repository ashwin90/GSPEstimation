from estimation.least_squares import LeastSquaresEstimator, LeastSquaresNonLinearProblem
from models.markov_chain import MarkovChainModel


class MarkovChainLeastSquaresEstimator(LeastSquaresEstimator):
    def can_estimate(self, model):
        return MarkovChainModel == model.__class__

    def non_linear_problem_klass(self):
        return MarkovChainLeastSquaresNonLinearProblem


class MarkovChainLeastSquaresNonLinearProblem(LeastSquaresNonLinearProblem):
    def objective_function(self, parameters):
        self.model.update_parameters_from_vector(parameters)
        accum = 0.0
        for offer_set, approximate in self.approximates.items():
            total = sum(approximate.values())
            probabilities = self.model.expected_number_of_visits_if(offer_set)
            for product, amount in approximate.items():
                accum += (((amount / total) - probabilities[product]) ** 2)
        return accum
