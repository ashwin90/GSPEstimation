from estimation.maximum_likelihood.markov_chain import MarkovChainMaximumLikelihoodEstimator
from estimation.maximum_likelihood.ranked_list import RankedListMaximumLikelihoodEstimator

from estimation import EstimationMethod
from estimation.maximum_likelihood.exponomial import ExponomialMaximumLikelihoodEstimator
from estimation.maximum_likelihood.latent_class import LatentClassMaximumLikelihoodEstimator
from estimation.maximum_likelihood.multinomial_logit import MultinomialLogitMaximumLikelihoodEstimator
from estimation.maximum_likelihood.nested_logit import NestedLogitMaximumLikelihoodEstimator


class MaximumLikelihood(EstimationMethod):
    @classmethod
    def estimators(cls):
        return [ExponomialMaximumLikelihoodEstimator(),
                LatentClassMaximumLikelihoodEstimator(),
                MarkovChainMaximumLikelihoodEstimator(),
                MultinomialLogitMaximumLikelihoodEstimator(),
                NestedLogitMaximumLikelihoodEstimator(),
                RankedListMaximumLikelihoodEstimator()]
