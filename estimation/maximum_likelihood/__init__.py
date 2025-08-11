from numpy import array

from estimation import Estimator
from optimization.non_linear import NonLinearProblem, NonLinearSolver


class MaximumLikelihoodEstimator(Estimator):
    def can_estimate(self, model):
        raise NotImplementedError('Subclass responsibility')

    def estimate(self, model, transactions):
        problem = MaximumLikelihoodNonLinearProblem(model, transactions)
        solution = NonLinearSolver.default().solve(problem, self.profiler())
        model.update_parameters_from_vector(solution)
        return model


class MaximumLikelihoodNonLinearProblem(NonLinearProblem):
    def __init__(self, model, transactions):
        self.model = model
        self.transactions = transactions

    def constraints(self):
        return self.model.constraints()

    def objective_function(self, parameters):
        self.model.update_parameters_from_vector(parameters)
        return -self.model.log_likelihood_for(self.transactions)

    def amount_of_variables(self):
        return len(self.model.parameters_vector())

    def initial_solution(self):
        return array(self.model.parameters_vector())
