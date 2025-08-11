import numpy
from numpy import ones, array
from src import utils

from models import Model
from optimization.non_linear import Constraints
from utils import generate_n_equal_numbers_that_sum_one, generate_n_random_numbers_that_sum_m, ONE_UPPER_BOUND


class NestedLogitModel(Model):
    @classmethod
    def code(cls):
        return 'nl'

    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['nests'], data['etas'])

    @classmethod
    def default_naive(cls, products, nests=None):
        if not nests:
            nests = [{'products': filter(lambda p: p % 2 == 1, products)},
                     {'products': filter(lambda p: p % 2 == 0, products)}]

        for nest in nests:
            nest['lambda'] = 0.8
        etas = generate_n_equal_numbers_that_sum_one(len(products))
        return cls(products, nests, etas)

    @classmethod
    def default_random(cls, products, nests):
        for nest in nests:
            nest['lambda'] = numpy.random.uniform(0.6, 1)
        etas = generate_n_random_numbers_that_sum_m(len(products), 2)
        return cls(products, nests, etas)

    def constraints(self):
        return NestedLogitModelConstraints(self)

    def __init__(self, products, nests, etas):
        """
            nests: [{'lambda': 0.8, 'products': [3, 2, 0, 4]}, ...]
        """
        super(NestedLogitModel, self).__init__(products)
        if len(products) != len(etas):
            info = (len(products), len(etas))
            raise Exception('Amount of products (%s) should be equal to amount of utilities "eta" (%s)' % info)
        if products != sorted([product for nest in nests for product in nest['products']]):
            info = (products, nests)
            raise Exception('All product should be inside a nest.\nProducts: %s\nNests: %s' % info)
        self.etas = etas
        self.nests = nests

    def utility_for(self, product):
        return self.etas[product]

    def nest_for(self, product):
        for nest in self.nests:
            if product in nest['products']:
                return nest
        raise Exception('Not nest found for product %s in nests %s.' % (product, self.nests))

    def probability_of(self, transaction):
        if transaction.product not in transaction.offered_products:
            return 0

        denominator = 0.0
        for nest in self.nests:
            term = 0.0
            for product in filter(lambda p: p in transaction.offered_products, nest['products']):
                term += (self.utility_for(product) ** (1.0 / nest['lambda']))
            denominator += (term ** nest['lambda'])

        nest = self.nest_for(transaction.product)
        numerator = 0.0
        for product in filter(lambda p: p in transaction.offered_products, nest['products']):
            numerator += (self.utility_for(product) ** (1.0 / nest['lambda']))
        numerator **= (nest['lambda'] - 1.0)
        numerator *= (self.utility_for(transaction.product) ** (1.0 / nest['lambda']))

        return numerator / denominator

    def parameters_vector(self):
        return self.etas + map(lambda d: d['lambda'], self.nests)

    def update_parameters_from_vector(self, parameters):
        self.etas = list(parameters)[:len(self.etas)]
        for i, nest in enumerate(self.nests):
            nest['lambda'] = list(parameters)[len(self.etas) + i]

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'nests': self.nests,
            'etas': self.etas
        }

    def __repr__(self):
        return '<Products: %s Nests: %s Etas: %s>' % (self.products, self.nests, self.etas)


class NestedLogitModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * 0.1

    def upper_bounds_vector(self):
        return array([utils.NLP_UPPER_BOUND_INF] * len(self.model.etas) + [ONE_UPPER_BOUND] * len(self.model.nests))
