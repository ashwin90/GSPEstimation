import numpy as np

from estimation import Estimator
from utils import KLDIV_REL_CHANGE, KLDIV_CUTOFF


class ExpectationMaximizationEstimator(Estimator):
    def can_estimate(self, model):
        raise NotImplementedError('Subclass responsibility')

    def estimate(self, model, transactions):
        self.profiler().reset_convergence_criteria()
        self.profiler().update_time()
        model = self.custom_initial_solution(model, transactions)
        data_neg_entropy = np.dot(model.observed_sales, np.ma.log(model.empirical_choice_probs).filled(0.)) / np.sum(model.observed_sales)
        kldiv_prev = data_neg_entropy - model.log_likelihood_for(transactions)
        while True:
            self.profiler().start_iteration()
            model = self.one_step(model, transactions)
            likelihood = model.log_likelihood_for(transactions)
            self.profiler().stop_iteration(likelihood)
            kldiv = data_neg_entropy - likelihood
            if kldiv <= KLDIV_CUTOFF or np.abs(kldiv_prev - kldiv)/kldiv_prev < KLDIV_REL_CHANGE:
                break
            kldiv_prev = kldiv

        #elif self.profiler().should_stop():
        # break

        return model

    def one_step(self, model, transactions):
        raise NotImplementedError('Subclass responsibility')

    def custom_initial_solution(self, model, transactions):
        return model
