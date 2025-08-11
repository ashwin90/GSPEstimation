import json

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


def estimate_GSP_model(file_name, maxk, irrat_prop_ub):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    num_products = data['amount_products']
    products = range(num_products)
    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    gsp_model = GeneralizedStochasticPreferenceModel.default_naive(products, variant=WITH_NO_CHOICE)
    gsp_model.populate(in_sample_transactions)
    gsp_market_explorer = GSPMarketExplorer(maxk, irrat_prop_ub)
    gsp_est = GSPExpectationMaximizationEstimator(gsp_market_explorer)
    gsp_model, gsp_runtime = gsp_est.estimate_with_market_discovery(gsp_model, None, file_name, ".", to_save=False)

    # predicted probs on out-of-sample transactions
    out_of_sample_transactions = Transaction.from_json(data['transactions']['out_of_sample'])
    predict_for_out_of_sample_transactions(gsp_model, out_of_sample_transactions, num_products)


if __name__ == "_main__":
    transaction_filename = ""
    estimate_GSP_model(transaction_filename, maxk=2, irrat_prop_ub=0.5)
