#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/2018

@author: Maurizio Ferrari Dacrema
"""

import traceback, os, shutil

from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender

from Evaluation.Evaluator import EvaluatorHoldout, EvaluatorNegativeItemSample
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Utils.DataReader import load_urm, load_icm, load_target
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping


def write_log_string(log_file, string):
    log_file.write(string)
    log_file.flush()


def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):

    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)

    return recommender_object


def run_recommender(recommender_class):
    temp_save_file_folder = "./result_experiments/__temp_model/"

    if not os.path.isdir(temp_save_file_folder):
        os.makedirs(temp_save_file_folder)

    try:
        URM_all = load_urm()
        URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.85)
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.85)
        ICM_all = load_icm()

        write_log_string(log_file, "On Recommender {}\n".format(recommender_class))

        recommender_object = _get_instance(recommender_class, URM_train, ICM_all)

        if isinstance(recommender_object, Incremental_Training_Early_Stopping):
            fit_params = {"epochs": 15}
        else:
            fit_params = {}

        recommender_object.fit(**fit_params)

        write_log_string(log_file, "Fit OK, ")



        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen = True)
        results_df, results_run_string = evaluator.evaluateRecommender(recommender_object)

        write_log_string(log_file, "EvaluatorHoldout OK, ")



        evaluator = EvaluatorNegativeItemSample(URM_test, URM_train, [5], exclude_seen = True)
        _, _ = evaluator.evaluateRecommender(recommender_object)

        write_log_string(log_file, "EvaluatorNegativeItemSample OK, ")



        recommender_object.save_model(temp_save_file_folder, file_name="temp_model")

        write_log_string(log_file, "save_model OK, ")



        recommender_object = _get_instance(recommender_class, URM_train, ICM_all)
        recommender_object.load_model(temp_save_file_folder, file_name="temp_model")

        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen = True)
        result_df_load, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)

        assert results_df.equals(result_df_load), "The results of the original model should be equal to that of the loaded one"

        write_log_string(log_file, "load_model OK, ")



        shutil.rmtree(temp_save_file_folder, ignore_errors = True)

        write_log_string(log_file, " PASS\n")
        write_log_string(log_file, results_run_string + "\n\n")



    except Exception as e:

        print("On Recommender {} Exception {}".format(recommender_class, str(e)))
        log_file.write("On Recommender {} Exception {}\n\n\n".format(recommender_class, str(e)))
        log_file.flush()

        traceback.print_exc()


from Recommenders.Recommender_import_list import *


if __name__ == '__main__':


    log_file_name = "Experiments/run_test_recommender.txt"


    recommender_list = [
        Random,
        TopPop,
        GlobalEffects,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        ItemKNNCBFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        SLIM_BPR_Cython,
        SLIMElasticNetRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        MatrixFactorization_AsySVD_Cython,
        PureSVDRecommender,
        IALSRecommender,
        EASE_R_Recommender,
    ]

    log_file = open(log_file_name, "w")



    for recommender_class in recommender_list:
        run_recommender(recommender_class)
    #
    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(run_dataset, dataset_list)


#%%

#%%
