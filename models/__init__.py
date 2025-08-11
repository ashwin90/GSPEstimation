from math import sqrt

from transactions.base import Transaction
from utils import *


class Model(object):
    """
        Represents a mathematical model for Discrete Choice Consumer Decision.
    """
    def __init__(self, products):
        if products != range(len(products)):
            raise Exception('Products should be entered as an ordered consecutive list.')
        self.products = products

    @classmethod
    def code(cls):
        raise NotImplementedError('Subclass responsibility')

    @classmethod
    def from_data(cls, data):
        for klass in cls.__subclasses__():
            if data['code'] == klass.code():
                return klass.from_data(data)
        raise Exception('No model can be created from data %s')

    @classmethod
    def default_naive(cls, *args, **kwargs):
        """
            Must return a default model with naive pdf parameters to use as an initial solution for estimators.
        """
        raise NotImplementedError('Subclass responsibility')

    @classmethod
    def default_random(cls, *args, **kwargs):
        """
            Must return a default model with random pdf parameters to use as a ground model.
        """
        raise NotImplementedError('Subclass responsibility')

    def probability_of(self, transaction):
        """
            Must return the probability of a transaction.
        """
        raise NotImplementedError('Subclass responsibility')

    def log_probability_of(self, transaction):
        return safe_log(self.probability_of(transaction))

    def probability_distribution_over(self, offered_products):
        distribution = []
        for product in range(len(self.products)):
            transaction = Transaction(product, offered_products)
            distribution.append(self.probability_of(transaction))
        return distribution

    def log_likelihood_for(self, transactions):
        result = 0
        den_res = 0
        for transaction in transactions:
            result += transaction.sales*self.log_probability_of(transaction)
            den_res += transaction.sales
        # return result / len(transactions)
        return result / den_res

    def metric_for(self, model_probs, prod_sales, offerset_sales, n_obs_per_os, metric=RMSE_METRIC):
        data_probs = prod_sales / offerset_sales
        if metric == NLL_METRIC:
            return np.dot(prod_sales, np.ma.log(data_probs).filled(0.) - np.ma.log(model_probs).filled(0.))
            # print(data_probs[model_probs < 1e-15])
            # print('Max log argument: {0}'.format(np.max(data_probs/model_probs)))
            # return np.dot(prod_sales, np.ma.log(data_probs/model_probs).filled(0.))
        elif metric == RMSE_METRIC:
            return np.sum(np.square(offerset_sales*model_probs-prod_sales))
            #return np.mean(np.square(model_probs-data_probs))
        elif metric == CHISQ_METRIC:
            return np.mean(np.square(model_probs-data_probs)/(model_probs + 0.5))
        elif metric == HIT_RATE_METRIC:
            start_id = 0
            hits = 0.
            for n_obs in n_obs_per_os:
                prod_sales_obs = prod_sales[start_id:start_id + n_obs]
                hits += prod_sales_obs[np.argmax(model_probs[start_id:start_id + n_obs])]
                start_id += n_obs
            return hits / np.sum(prod_sales)
        elif metric == MAPE_METRIC:
            return np.mean(np.abs(model_probs - data_probs)/data_probs)
        elif metric == L1_METRIC:
            return np.sum(np.abs(offerset_sales*model_probs-prod_sales))

    def rmse_for(self, transactions):
        rmse = 0.0
        amount_terms = 0
        for transaction in transactions:
            for product in transaction.offered_products:
                probability = self.probability_of(Transaction(product, transaction.offered_products))
                rmse += transaction.sales*((probability - float(product == transaction.product)) ** 2)
                amount_terms += transaction.sales
        return sqrt(rmse / float(amount_terms))

    def rmse_known_ground(self, ground_model, transactions):
        rmse = 0.0
        amount_terms = 0
        for transaction in transactions:
            for product in transaction.offered_products:
                probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
                probability_2 = ground_model.probability_of(Transaction(product, transaction.offered_products))
                rmse += ((probability_1 - probability_2) ** 2)
                amount_terms += 1
        return sqrt(rmse / float(amount_terms))

    def hit_rate_for(self, transactions):
        hit_rate = 0
        for transaction in transactions:
            probabilities = []
            for product in transaction.offered_products:
                probabilities.append((product, self.probability_of(Transaction(product, transaction.offered_products))))
            most_probable = max(probabilities, key=lambda p: p[1])[0]
            hit_rate += int(most_probable == transaction.product)
        return float(hit_rate) / float(len(transactions))

    def soft_chi_squared_score_for(self, ground_model, transactions):
        expected_purchases = [0.0 for _ in self.products]
        observed_purchases = [0.0 for _ in self.products]

        for transaction in transactions:
            for product in transaction.offered_products:
                observed_purchases[product] += ground_model.probability_of(Transaction(product, transaction.offered_products))
                expected_purchases[product] += self.probability_of(Transaction(product, transaction.offered_products))

        score = 0.0
        for p in self.products:
            score += (((expected_purchases[p] - observed_purchases[p]) ** 2) / (expected_purchases[p] + 0.5))
        return score / float(len(self.products))

    def hard_chi_squared_score_for(self, transactions):
        expected_purchases = [0.0 for _ in self.products]
        observed_purchases = [0.0 for _ in self.products]

        for transaction in transactions:
            observed_purchases[transaction.product] += 1.0
            for product in transaction.offered_products:
                expected_purchases[product] += self.probability_of(Transaction(product, transaction.offered_products))

        score = 0.0
        for p in self.products:
            score += (((expected_purchases[p] - observed_purchases[p]) ** 2) / (expected_purchases[p] + 0.5))
        return score / float(len(self.products))

    def aic_for(self, transactions):
        k = self.amount_of_parameters()
        amount_samples = len(transactions)
        l = self.log_likelihood_for(transactions) * len(transactions)
        return 2 * (k - l + (k * (k + 1) / (amount_samples - k - 1)))

    def bic_for(self, transactions):
        k = self.amount_of_parameters()
        amount_samples = len(transactions)
        l = self.log_likelihood_for(transactions) * len(transactions)
        return -2 * l + (k * log(amount_samples))

    def amount_of_parameters(self):
        return len(self.parameters_vector())

    def parameters_vector(self):
        """
            Vector of parameters that define the model. For example lambdas and ros in Markov Chain.
        """
        return []

    def update_parameters_from_vector(self, parameters):
        """
            Updates current parameters from input parameters vector
        """
        pass

    def constraints(self):
        raise NotImplementedError('Subclass responsibility')

    def data(self):
        raise NotImplementedError('Subclass responsibility')


from models.random_choice import RandomChoiceModel
