from numpy import array

from estimation import Estimator
from optimization.non_linear import NonLinearProblem, NonLinearSolver
from transactions.base import Transaction


class LeastSquaresEstimator(Estimator):
    def can_estimate(self, model):
        raise NotImplementedError('Subclass responsibility')

    def approximates_based_on(self, transactions):
        approximates = {}
        for transaction in transactions:
            offer_set = tuple(sorted(transaction.offered_products))
            if offer_set not in approximates:
                approximates[offer_set] = {product: 0.0 for product in offer_set}
            approximates[offer_set][transaction.product] += 1.0
        return approximates

    def estimate(self, model, transactions):
        approximates = self.approximates_based_on(transactions)
        non_linear_problem_klass = self.non_linear_problem_klass()
        problem = non_linear_problem_klass(model, approximates)
        solution = NonLinearSolver.default().solve(problem, self.profiler())
        model.update_parameters_from_vector(solution)
        return model

    def non_linear_problem_klass(self):
        return LeastSquaresNonLinearProblem


class LeastSquaresNonLinearProblem(NonLinearProblem):
    def __init__(self, model, approximates):
        self.model = model
        self.approximates = approximates

    def constraints(self):
        return self.model.constraints()

    def objective_function(self, parameters):
        self.model.update_parameters_from_vector(parameters)
        accum = 0.0
        for offer_set, approximate in self.approximates.items():
            total = sum(approximate.values())
            for product, amount in approximate.items():
                transaction = Transaction(product, list(offer_set))
                accum += (((amount / total) - self.model.probability_of(transaction)) ** 2)
        return accum

    def amount_of_variables(self):
        return len(self.model.parameters_vector())

    def initial_solution(self):
        return array(self.model.parameters_vector())
