from estimation import EstimationMethod
from estimation.least_squares.exponomial import ExponomialLeastSquaresEstimator
from estimation.least_squares.markov_chain import MarkovChainLeastSquaresEstimator
from estimation.least_squares.multinomial_logit import MultinomialLogitLeastSquaresEstimator
from estimation.least_squares.nested_logit import NestedLogitLeastSquaresEstimator


class LeastSquares(EstimationMethod):
    def estimators(self):
        return [ExponomialLeastSquaresEstimator(),
                LatentClassLeastSquaresEstimator(),
                MarkovChainLeastSquaresEstimator(),
                MultinomialLogitLeastSquaresEstimator(),
                NestedLogitLeastSquaresEstimator(),
                RankedListLeastSquaresEstimator()]
