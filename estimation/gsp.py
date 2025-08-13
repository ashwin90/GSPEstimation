import pickle
from copy import deepcopy

import numpy as np
from scipy.stats import chi2

from estimation import Estimator
from estimation.market_explore.gsp import NullMarketExplorer
from models.gsp import GeneralizedStochasticPreferenceModel
from utils import ESTIMATOR_TIME_LIMIT, KLDIV_CUTOFF, KLDIV_REL_CHANGE


class GSPEstimator(Estimator):
    @classmethod
    def with_this(cls, market_explorer):
        return cls(market_explorer)

    def __init__(self, market_explorer=NullMarketExplorer()):
        super(GSPEstimator, self).__init__()
        self.market_explorer = market_explorer

    def can_estimate(self, model):
        return GeneralizedStochasticPreferenceModel == model.__class__

    def estimate(self, model, transactions):
        raise NotImplementedError('Subclass responsibility')

    def estimate_with_market_discovery(self, model, transactions, dataset, save_path, to_save=True, gpt_exp=False):
        data_neg_entropy = np.dot(model.observed_sales, np.ma.log(model.empirical_choice_probs).filled(0.)) / np.sum(model.observed_sales)
        model = self.estimate(model, transactions)
        add = True
        while add and int(self.profiler().duration()) < ESTIMATOR_TIME_LIMIT:
            new_types = self.market_explorer.explore_for(self, model, transactions)
            add, model = self.is_worth_adding(model, new_types, transactions, data_neg_entropy, gpt_exp)
            model.simplify()
            # print(f'Current log-ll={model.log_likelihood_for(transactions):.4f}')
        runtime = int(self.profiler().duration())
        if to_save:
            pickle.dump((model.betas, model.customer_types, runtime), open('{1}/GSP_{0}_maxk={2}_maxirrmass={3}.final_model'.format(dataset, save_path, self.market_explorer.max_choice_index, self.market_explorer.max_irr_mass), 'wb'))
        return model, runtime

    def is_worth_adding(self, model, new_types, transactions, data_neg_entropy, gpt_exp):
        new_model = deepcopy(model)
        # add each type and update type indicators
        # make sure new masses respect constraints
        for (new_ranked_list, new_choice_index) in new_types:
            new_model.add_ranked_list_and_index(new_ranked_list, new_choice_index, self.market_explorer.max_irr_mass)
        new_model = self.estimate(new_model, transactions)
        if gpt_exp:
            return self.compare_statistical_significance(model, new_model, transactions), new_model
        else:
            kldiv_1 = data_neg_entropy - model.log_likelihood_for(transactions)
            kldiv_2 = data_neg_entropy - new_model.log_likelihood_for(transactions)
            is_worth_adding = True if (kldiv_2 > KLDIV_CUTOFF) and (np.abs(kldiv_2 - kldiv_1)/kldiv_1 > KLDIV_REL_CHANGE) else False
            return is_worth_adding, new_model

    def compare_statistical_significance(self, model, new_model, transactions):
        likelihood_1 = model.log_likelihood_for(transactions)
        likelihood_2 = new_model.log_likelihood_for(transactions)
        likelihood_ratio = -2.0 * (likelihood_1 - likelihood_2)
        dimensionality_difference = len(new_model.betas) - len(model.betas)
        return likelihood_ratio > chi2.isf(q=0.05, df=dimensionality_difference)
        # mprint('likelihood ratio: {0}'.format(likelihood_ratio))
        # return likelihood_ratio > LIKELIHOOD_RATIO_THRESH
