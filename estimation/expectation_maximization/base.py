from estimation import EstimationMethod
from estimation.expectation_maximization.markov_chain import MarkovChainExpectationMaximizationEstimator

from estimation.expectation_maximization.ranked_list import RankedListExpectationMaximizationEstimator


class ExpectationMaximization(EstimationMethod):
    @classmethod
    def estimators(cls):
        return [MarkovChainExpectationMaximizationEstimator(),
                RankedListExpectationMaximizationEstimator()]