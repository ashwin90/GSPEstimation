# coding=utf-8
from math import log

from numpy import ones

from models import Model
from optimization.non_linear import Constraints
from utils import generate_n_equal_numbers_that_sum_one, generate_n_random_numbers_that_sum_m, ZERO_LOWER_BOUND


class MultinomialLogitModel(Model):
    @classmethod
    def code(cls):
        return 'mnl'

    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['etas'])

    @classmethod
    def default_naive(cls, products):
        return cls(products, generate_n_equal_numbers_that_sum_one(len(products) - 1))

    @classmethod
    def default_random(cls, products):
        return cls(products, generate_n_random_numbers_that_sum_m(len(products) - 1, 2))

    def __init__(self, products, etas):
        super(MultinomialLogitModel, self).__init__(products)
        if len(etas) != len(products) - 1:
            info = (len(etas), len(products))
            raise Exception('Incorrect amount of etas (%s) for amount of products (%s)' % info)
        self.etas = etas

    def probability_of(self, transaction):
        if transaction.product not in transaction.offered_products:
            return 0
        den = sum([self.utility_of(product) for product in transaction.offered_products])
        return self.utility_of(transaction.product) / den

    def log_probability_of(self, transaction):
        den = sum([self.utility_of(product) for product in transaction.offered_products])
        return log(self.utility_of(transaction.product)) - log(den)

    def utility_of(self, product):
        return 1.0 if product == 0 else self.etas[product - 1]

    def parameters_vector(self):
        return self.etas

    def update_parameters_from_vector(self, parameters):
        self.etas = list(parameters)

    def constraints(self):
        return MultinomialLogitModelConstraints(self)

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'etas': self.etas,
        }

    def __repr__(self):
        return '<Products: %s ; Etas: %s >' % (self.products, self.etas)


class MultinomialLogitModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * utils.NLP_UPPER_BOUND_INF
