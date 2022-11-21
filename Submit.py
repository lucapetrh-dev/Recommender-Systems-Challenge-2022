import os
import numpy as np
import pandas as pd

from Recommenders.Hybrids.Hybrid_SlimElastic_Rp3_PureSVD import Hybrid_SlimElastic_Rp3_PureSVD
from Recommenders.Recommender_import_list import *
from Run import _get_params
from Utils.DataReader import load_urm, load_target, load_icm

res_dir = 'Experiments/csv'
output_root_path = "Experiments/"

# If directory does not exist, create
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)


def run_prediction_on_target(URM_all):
    recommender = Hybrid_SlimElastic_Rp3_PureSVD(URM_all)
    fit_params = _get_params(recommender)
    recommender.fit(**fit_params)

    test_users = load_target()

    user_id = test_users['user_id']
    recommendations = []
    for user in user_id:
        recommendations.append(recommender.recommend(user, cutoff=10))

    for index in range(len(recommendations)):
        recommendations[index] = np.array(recommendations[index])

    test_users['item_list'] = recommendations
    test_users['item_list'] = pd.DataFrame(
        [str(line).strip('[').strip(']').replace("'", "") for line in test_users['item_list']])
    test_users.to_csv('Submissions\Submission_SlimElastic_rp3_PureSVD.csv', index=False)


if __name__ == '__main__':
    URM_all = load_urm()
    target_ids = load_target()
    run_prediction_on_target(URM_all)
