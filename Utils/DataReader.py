import os

import pandas as pd
import scipy.sparse as sps
import numpy as np


def load_urm():
    df_original = load_urm_df()

    user_id_list = df_original['UserID'].values
    item_id_list = df_original['ItemID'].values
    rating_id_list = df_original['Data'].values

    csr_matrix = sps.csr_matrix((rating_id_list, (user_id_list, item_id_list)))
    csr_matrix = csr_matrix.astype(dtype=np.int32)

    return csr_matrix


def load_target():
    df_original = pd.read_csv('Dataset/data_target_users_test.csv')
    df_original.columns = ['UserID']
    return df_original


def load_icm():
    df_original = load_icm_df()

    item_id_list = df_original['ItemID'].values
    feature_id_list = df_original['FeatureID'].values
    data_list = df_original['Data'].values

    csr_matrix = sps.csr_matrix((data_list, (item_id_list, feature_id_list)))
    csr_matrix = csr_matrix.astype(dtype=np.int32)

    return csr_matrix


def load_merged_icm():
    df_original = load_merged_icm_df()

    item_id_list = df_original['item_id'].values
    type_list = df_original['type'].values
    length_list = df_original['length'].values

    csr_matrix = sps.csr_matrix((length_list, (item_id_list, type_list)))
    csr_matrix = csr_matrix.astype(dtype=np.int32)

    return csr_matrix


def load_icm_df():
    df_original = pd.read_csv("Dataset/data_ICM_length.csv",
                              header=1,
                              dtype={0: np.int32,
                                     1: np.int32,
                                     2: np.int32})
    df_original.columns = ["ItemID", "FeatureID", "Data"]
    # print("Resultant CSV after joining all CSV files at a particular location...")
    return df_original


def load_merged_icm_df():
    ICM_length = pd.read_csv("Dataset/data_ICM_length.csv",
                             header=1,
                             dtype={0: np.int32,
                                    1: np.int32,
                                    2: np.int32})
    ICM_length.columns = ["item_id", "feature_id", "length"]
    ICM_type = pd.read_csv("Dataset/data_ICM_type.csv",
                           header=1,
                           dtype={0: np.int32,
                                  1: np.int32,
                                  2: np.int32})
    ICM_type.columns = ["item_id", "type", "data"]

    # print("Resultant CSV after joining all CSV files at a particular location...")

    ICM_all = pd.merge(ICM_length, ICM_type, how='inner', on='item_id')
    ICM_all.drop(['feature_id', 'data'], axis=1, inplace=True)
    print(ICM_all)

    return ICM_all


def load_urm_df():
    urm_path = os.path.join(os.path.dirname(__file__), '../Dataset/URM_Binary_Ratings.csv')
    df_original = pd.read_csv(filepath_or_buffer=urm_path)

    df_original.columns = ["UserID", "ItemID", "Data"]

    return df_original


def load_inverted_urm_df():
    urm_path = os.path.join(os.path.dirname(__file__), 'Dataset/inverted_interactions.csv')
    df_original = pd.read_csv(filepath_or_buffer=urm_path,
                              header=1,
                              dtype={0: np.int32,
                                     1: np.int32,
                                     2: str,
                                     3: np.float16})

    df_original.columns = ["UserID", "ItemID", "Impression", "Data"]

    return df_original
# %%
