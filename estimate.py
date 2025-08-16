import json
import sys
from pprint import pprint

import numpy as np

from estimation.expectation_maximization.gsp import GSPExpectationMaximizationEstimator
from estimation.market_explore.gsp import MIPMarketExplorer as GSPMarketExplorer
from models.gsp import GeneralizedStochasticPreferenceModel
from transactions.base import Transaction
from utils import WITH_NO_CHOICE


def predict_for_out_of_sample_transactions(gsp_model, transactions, num_prods):
    offered_products = []
    chosen_products = []
    for tr in transactions:
        os = np.zeros(num_prods, dtype=int)
        os[tr.offered_products] = 1
        chosen_products.append(tr.product)
        offered_products.append(os)

    test_pred_probs = gsp_model.predict_proba(np.array(offered_products), np.array(chosen_products))
    print("Predicted choice probabilities for out_of_sample transactions:")
    pprint(test_pred_probs)


def estimate_GSP_model(input_data, num_products, maxk, irrat_prop_ub):
    products = range(num_products)
    in_sample_transactions = Transaction.from_json(input_data['transactions']['in_sample'])
    gsp_model = GeneralizedStochasticPreferenceModel.default_naive(products, variant=WITH_NO_CHOICE)
    gsp_model.populate(in_sample_transactions)
    gsp_market_explorer = GSPMarketExplorer(maxk, irrat_prop_ub)
    gsp_est = GSPExpectationMaximizationEstimator(gsp_market_explorer)
    gsp_model, gsp_runtime = gsp_est.estimate_with_market_discovery(gsp_model, None, "", ".", to_save=False)
    # print estimated model
    pprint(gsp_model.data())
    # predicted probs on out-of-sample transactions
    out_of_sample_transactions = Transaction.from_json(input_data['transactions']['out_of_sample'])
    predict_for_out_of_sample_transactions(gsp_model, out_of_sample_transactions, num_products)


if __name__ == "__main__":
    max_choice_index_for_non_standard_types = int(sys.argv[2])
    max_proportion_of_non_standard_types = float(sys.argv[3])
    input_file = open(sys.argv[1], 'r')
    input_data = json.loads(input_file.read())
    input_file.close()
    num_products = input_data['amount_products'] # including no-purchase option
    assert 0 <= max_proportion_of_non_standard_types <= 1, f'upper bound on total proportion of non-standard types must be between 0 and 1, given {max_proportion_of_non_standard_types}'
    assert 1 <= max_choice_index_for_non_standard_types <= num_products-1, f'maximum choice index for non-standard type must be between 1 and {num_products-1}, given {max_choice_index_for_non_standard_types}'
    estimate_GSP_model(input_data, num_products, max_choice_index_for_non_standard_types, max_proportion_of_non_standard_types)
