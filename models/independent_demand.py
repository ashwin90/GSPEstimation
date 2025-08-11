from numpy import ones, array, zeros

from models import Model
from optimization.non_linear import Constraints
from utils import generate_n_equal_numbers_that_sum_one, generate_n_random_numbers_that_sum_one, ZERO_LOWER_BOUND, \
    ONE_UPPER_BOUND


class IndependentDemandModel(Model):
    @classmethod
    def code(cls):
        return 'id'

    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['betas'])

    @classmethod
    def default_random(cls, products):
        betas = generate_n_random_numbers_that_sum_one(len(products))[1:]
        return cls(products, betas)

    @classmethod
    def default_naive(cls, products):
        betas = generate_n_equal_numbers_that_sum_one(len(products))[1:]
        return cls(products, betas)

    def __init__(self, products, betas):
        super(IndependentDemandModel, self).__init__(products)
        if len(betas) + 1 != len(products):
            raise Exception('Betas should be one less than amount of products.')
        if not (sum(betas) < 1 and all([beta >= 0 for beta in betas])):
            raise Exception('Betas should be less than one and positive.')
        self.betas = betas

    def probability_of(self, transaction):
        if transaction.product == 0:
            not_offered = [p for p in self.products if p not in transaction.offered_products]
            return 1 - sum(self.betas) + sum([self.betas[p - 1] for p in not_offered])
        return self.betas[transaction.product - 1]

    def parameters_vector(self):
        return self.betas

    def update_parameters_from_vector(self, parameters):
        self.betas = list(parameters)

    def constraints(self):
        return IndependentDemandModelConstraints(self)

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'betas': self.betas,
        }


class IndependentDemandModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return ones(len(self.model.parameters_vector()))

    def amount_of_constraints(self):
        return 1

    def lower_bounds_over_constraints_vector(self):
        return array([ZERO_LOWER_BOUND])

    def upper_bounds_over_constraints_vector(self):
        return array([ONE_UPPER_BOUND])

    def non_zero_parameters_on_constraints_jacobian(self):
        return len(self.model.betas)

    def constraints_evaluator(self):
        def evaluator(x):
            return array([sum(x)])
        return evaluator

    def constraints_jacobian_evaluator(self):
        def jacobian_evaluator(x, flag):
            if flag:
                return zeros(len(self.model.betas)), array(range(len(self.model.betas)))
            else:
                return ones(len(self.model.betas))
        return jacobian_evaluator
