from functools import reduce

from numpy import ones, array, zeros

from models import Model
from models.multinomial_logit import MultinomialLogitModel
from optimization.non_linear import Constraints
from utils import generate_n_equal_numbers_that_sum_one, generate_n_random_numbers_that_sum_one, ZERO_LOWER_BOUND, \
    NLP_UPPER_BOUND_INF


class LatentClassModel(Model):
    @classmethod
    def code(cls):
        return 'lc'

    @classmethod
    def from_data(cls, data):
        multi_logit_models = [Model.from_data(multi_logit_model) for multi_logit_model in data['multi_logit_models']]
        return cls(data['products'], data['gammas'], multi_logit_models)

    @classmethod
    def default_naive(cls, products, amount_classes=3):
        gammas = generate_n_equal_numbers_that_sum_one(amount_classes)
        multi_logit_models = [MultinomialLogitModel.default_naive(products) for i in range(amount_classes)]
        return cls(products, gammas, multi_logit_models)

    @classmethod
    def default_random(cls, products, amount_classes):
        gammas = generate_n_random_numbers_that_sum_one(amount_classes)
        multi_logit_models = [MultinomialLogitModel.default_random(products) for i in range(amount_classes)]
        return cls(products, gammas, multi_logit_models)

    def __init__(self, products, gammas, multi_logit_models):
        super(LatentClassModel, self).__init__(products)
        if len(gammas) != len(multi_logit_models):
            info = (len(gammas), len(multi_logit_models))
            raise Exception('Amount of gammas (%s) should be equal to amount of MNL models (%s).' % info)
        self.gammas = gammas
        self.multi_logit_models = multi_logit_models

    def probability_of(self, transaction):
        probability = 0.0
        for gamma, model in zip(self.gammas, self.multi_logit_models):
            probability += (gamma * model.probability_of(transaction))
        return probability

    def mnl_models(self):
        return self.multi_logit_models

    def amount_of_classes(self):
        return len(self.gammas)

    def add_new_class(self):
        percentage = 1.0 / (len(self.gammas) + 1.0)
        new_gamma = sum([gamma * percentage for gamma in self.gammas])
        self.gammas = [gamma * (1.0 - percentage) for gamma in self.gammas] + [new_gamma]
        self.multi_logit_models.append(MultinomialLogitModel.default_naive(self.products))

    def add_new_class_with(self, mnl_model):
        percentage = 1.0 / (len(self.gammas) + 1.0)
        new_gamma = sum([gamma * percentage for gamma in self.gammas])
        self.gammas = [gamma * (1.0 - percentage) for gamma in self.gammas] + [new_gamma]
        self.multi_logit_models.append(mnl_model)

    def update_gammas_from(self, gammas):
        self.gammas = list(gammas)

    def parameters_vector(self):
        etas = reduce(lambda x, y: x + y, [mnl.etas for mnl in self.mnl_models()], [])
        return self.gammas + etas

    def update_parameters_from_vector(self, parameters):
        self.gammas = list(parameters)[:len(self.gammas)]
        for i, mnl_model in enumerate(self.mnl_models()):
            offset = len(self.gammas) + (i * len(mnl_model.etas))
            mnl_model.etas = list(parameters)[offset:offset + len(mnl_model.etas)]

    def constraints(self):
        return LatentClassModelConstraints(self)

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'gammas': self.gammas,
            'multi_logit_models': [model.data() for model in self.multi_logit_models]
        }

    def __repr__(self):
        return '<Products: %s ; Gammas: %s >' % (self.products, self.gammas)


class LatentClassModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * NLP_UPPER_BOUND_INF

    def amount_of_constraints(self):
        return 1

    def lower_bounds_over_constraints_vector(self):
        return array([1.0])

    def upper_bounds_over_constraints_vector(self):
        return array([1.0])

    def non_zero_parameters_on_constraints_jacobian(self):
        return len(self.model.gammas)

    def constraints_evaluator(self):
        def evaluator(x):
            return array([sum(x[:len(self.model.gammas)])])
        return evaluator

    def constraints_jacobian_evaluator(self):
        def jacobian_evaluator(x, flag):
            if flag:
                return (zeros(len(self.model.gammas)),
                        array(range(len(self.model.gammas))))
            else:
                return ones(len(self.model.gammas))
        return jacobian_evaluator
